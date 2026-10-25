"""RL fine-tuning of the learned placer (10-08): group-relative policy gradient.

The placer's outputs become the means of a Gaussian policy (learned global
spreads for centres, in pin-box units, and log aspects). For each training
design K arrangements are sampled, each packed exactly from the origin
(rl/learned_placer.py to_layout) and scored with the contest cost against the
design's ground-truth baseline; the advantage of a sample is its reward
relative to the group's mean (std-normalized), so a design's difficulty
cancels out. A behaviour-cloning term (the supervised L1 to the ground truth)
keeps the policy near what it already knows.

  python scripts/rl_placer.py --init checkpoints/placer_c.pt --hours 4 --out checkpoints/placer_rl.pt
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch

PENALTY = 3.0          # cost of a sample that collides with a preplaced block / is infeasible


def contest_cost(inst, pos):
    from polish_bench import _score
    s = _score(inst, pos)
    return s.cost if s.is_feasible else PENALTY


def layout_cost(inst, net, pred):
    """Fast contest cost (rl/packer.py _Batch, HPWL / area relative to the ground
    truth) of the packed sample after the fast collision repair
    (rl/packer.py _origin_repair), as the real pipeline would see it; 10 if the
    repair fails. (A flat penalty gave no signal: 98% of raw samples collide;
    a graded one saturated.)"""
    from rl import learned_placer as L
    from rl import packer as K
    from rl import repack as R
    pos, (gp, gm), over = L.to_layout(inst, net, pred=pred)
    P = R._problem(inst, pos)
    bm = inst.baseline_metrics
    hb, ab = max(bm["hpwl_baseline"], 1e-9), max(bm["area_baseline"], 1e-9)
    Bt = K._Batch(P, hb, ab, (0.0, 0.0))
    p = P["pos"]
    w, h = p[:, 2].copy(), p[:, 3].copy()
    x, y = p[:, 0], p[:, 1]
    if over > 1e-6:
        fx = K._origin_repair(Bt, np.asarray(gp), np.asarray(gm), w, h, max_moves=2 * P["n"])
        if fx is None:
            return 10.0
        gp, gm = fx
        xs, ys, o2 = Bt.pack(gm[None], np.argsort(gp)[None], np.argsort(gm)[None], w[None], h[None])
        if o2[0] > 1e-6:
            return 10.0
        x, y = xs[0], ys[0]
    tmp = R._Cost(P, 1.0, 1.0)
    Bt.mib_v = sum(len({(round(w[i], 4), round(h[i], 4)) for i in g}) - 1 for g in tmp.mibs)
    return min(float(Bt.cost(x[None], y[None], w[None], h[None], np.zeros(1))[0]), 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--data-path", default="../")
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bc", type=float, default=0.5)
    ap.add_argument("--sigma-c", type=float, default=0.02)
    ap.add_argument("--sigma-a", type=float, default=0.2)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    import polish_bench as PB
    PB._init()
    from rl import learned_placer as L
    from train_placer import iter_training
    ck = torch.load(a.init, map_location="cpu")
    cfg = ck.get("cfg", {})
    net = L.PlacerNet(**cfg)
    net.load_state_dict(ck["state"])
    net.train()
    log_sc = torch.tensor(math.log(a.sigma_c), requires_grad=True)
    log_sa = torch.tensor(math.log(a.sigma_a), requires_grad=True)
    opt = torch.optim.Adam([{"params": net.parameters(), "lr": a.lr},
                            {"params": [log_sc, log_sa], "lr": 1e-3}])

    from rl.data import from_validation_sample
    val = [from_validation_sample(PB.EV, t) for t in range(100)]
    Wv = [math.exp(v.block_count / 12) for v in val]

    def validate():
        net.eval()
        costs = []
        with torch.no_grad():
            for inst in val:
                costs.append(layout_cost(inst, net, L.predict(net, inst)))
        net.train()
        w = sum(c * x for c, x in zip(costs, Wv)) / sum(Wv)
        return w, sum(c >= 10.0 for c in costs)

    best, nfail = validate()
    print(f"start: val weighted raw cost {best:.4f} (collisions/infeasible {nfail}); sigma_c {a.sigma_c} sigma_a {a.sigma_a}", flush=True)
    torch.save({"state": net.state_dict(), "cfg": cfg, "val_raw": best}, a.out)
    data = iter_training(a.data_path, a.seed)
    t0, end, seen = time.time(), time.time() + a.hours * 3600, 0
    run_c, run_m, run_f = [], [], []
    opt.zero_grad()
    while time.time() < end:
        inst, sol = next(data)
        n = inst.block_count
        if n < 3:
            continue
        feats, pin_feats, b2b, p2b, (PX, PY) = L.features(inst)
        mu_c, mu_a = net(feats, pin_feats, b2b, p2b)
        con = inst.constraints[:n]
        movable = (con[:, 1] == 0) if con.shape[1] > 1 else torch.ones(n, dtype=torch.bool)
        soft = movable & (con[:, 0] == 0)
        sc, sa = log_sc.exp(), log_sa.exp()
        eps_c = torch.randn(a.k, n, 2)
        eps_a = torch.randn(a.k, n)
        c_s = (mu_c.detach() + sc.detach() * eps_c)
        a_s = (mu_a.detach() + sa.detach() * eps_a)
        rewards = []
        for k in range(a.k):
            pred = ((c_s[k, :, 0] * PX).numpy(), (c_s[k, :, 1] * PY).numpy(), a_s[k].numpy())
            rewards.append(-layout_cost(inst, net, pred))
        R = torch.tensor(rewards)
        adv = (R - R.mean()) / (R.std() + 1e-6) if R.std() > 1e-9 else torch.zeros_like(R)
        # Gaussian log-likelihood of the sampled actions under the current policy
        lp_c = (-0.5 * ((c_s - mu_c) / sc) ** 2 - log_sc).sum(-1)          # [k, n]
        lp_a = -0.5 * ((a_s - mu_a) / sa) ** 2 - log_sa                   # [k, n]
        logp = (lp_c * movable).sum(1) + (lp_a * soft).sum(1)
        pg = -(adv * logp).mean() / max(int(movable.sum()), 1)
        bc, lc, _ = L.loss_fn(net, inst, sol)
        ((pg + a.bc * bc) / a.batch).backward()
        run_c.append(-float(R.mean()))
        run_m.append(-float(R.max()))
        run_f.append(0.0)
        seen += 1
        if seen % a.batch == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            with torch.no_grad():
                log_sc.clamp_(math.log(0.003), math.log(0.1))
                log_sa.clamp_(math.log(0.02), math.log(0.6))
        if seen % 200 == 0:
            print(f"{seen} designs, {(time.time() - t0) / 60:.0f} min: sample cost mean {np.mean(run_c):.4f} best-of-k "
                  f"{np.mean(run_m):.4f} collisions {np.mean(run_f):.2f} | sigma_c {float(log_sc.exp()):.4f} "
                  f"sigma_a {float(log_sa.exp()):.3f}", flush=True)
            run_c, run_m, run_f = [], [], []
        if seen % a.val_every == 0:
            v, nf = validate()
            tag = ""
            if v < best:
                best = v
                torch.save({"state": net.state_dict(), "cfg": cfg, "val_raw": v, "seen": seen}, a.out)
                tag = " (saved)"
            print(f"  val weighted raw cost {v:.4f} (collisions/infeasible {nf}){tag}", flush=True)
    v, nf = validate()
    if v < best:
        torch.save({"state": net.state_dict(), "cfg": cfg, "val_raw": v, "seen": seen}, a.out)
    print(f"done: {seen} designs, final val {v:.4f}, best {min(best, v):.4f}", flush=True)


if __name__ == "__main__":
    main()

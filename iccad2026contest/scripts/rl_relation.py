"""RL fine-tuning of the relation placer (10-11): group-relative policy gradient
over sampled sequence pairs (rl/placer_rl.py).

Per training design: K sequence pairs sampled from the Plackett-Luce policy,
each scored by the solver's path (repair, fit, short packer) against the
design's baseline; advantage = (group mean cost - cost) / group std. The
supervised loss (ground-truth centres + relations) at weight --bc keeps the
network near what it learned. Validation: the greedy arrangement of every
validation design through the same path, weighted like the contest; the
best checkpoint is kept.

  python scripts/rl_relation.py --init checkpoints/placer_r.pt --hours 8 --out checkpoints/placer_rl2.pt
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--data-path", default="../")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bc", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=1.0, help="Plackett-Luce logit scale on the scores")
    ap.add_argument("--fit", type=int, default=200)
    ap.add_argument("--pack", type=float, default=1.0)
    ap.add_argument("--min-blocks", type=int, default=30)
    ap.add_argument("--val-every", type=int, default=400)
    ap.add_argument("--val-pack", type=float, default=0.0,
                    help="packer seconds in validation (0: fit only, repeatable -- the packer stops on a clock)")
    ap.add_argument("--val-n", type=int, default=100, help="validation designs (fewer for a smoke test)")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    import multiprocessing as mp
    import polish_bench as PB
    PB._init()
    from rl import learned_placer as L
    from rl.packer import init_worker
    from rl.placer_rl import pl_logp, pl_sample, sample_cost, scores
    from rl.data import from_validation_sample
    from train_placer import iter_training
    pool = mp.get_context("forkserver").Pool(a.procs, initializer=init_worker)
    ck = torch.load(a.init, map_location="cpu")
    cfg = ck.get("cfg", {})
    net = L.PlacerNet(**cfg)
    net.load_state_dict(ck["state"])
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    val = [from_validation_sample(PB.EV, t) for t in range(a.val_n)]
    Wv = np.array([math.exp(v.block_count / 12) for v in val])

    def greedy_job(inst, seed):
        net.eval()
        with torch.no_grad():
            pred = L.predict(net, inst)
            gp, gm = L.rel_sequences(net, inst)
        net.train()
        return (inst, pred, (gp, gm), a.fit, a.val_pack, seed)

    def validate():
        res = pool.map(sample_cost, [greedy_job(v, 0) for v in val])
        c = np.array([r[0] for r in res])
        return float((c * Wv).sum() / Wv.sum()), float(c.mean()), int(sum(not r[1] for r in res))

    best, vm, nf = validate()
    print(f"start: val weighted {best:.4f} mean {vm:.4f} infeasible {nf}", flush=True)
    torch.save({"state": net.state_dict(), "cfg": cfg, "val_rl": best, "seen": 0}, a.out)
    data = iter_training(a.data_path, a.seed)
    t0, end, seen = time.time(), time.time() + a.hours * 3600, 0
    log = []
    net.train()
    opt.zero_grad()
    while time.time() < end:
        inst, sol = next(data)
        n = inst.block_count
        if n < a.min_blocks:
            continue
        sp, sm = scores(net, inst)
        u, v = a.beta * sp, a.beta * sm
        un, vn = u.detach().numpy(), v.detach().numpy()
        with torch.no_grad():
            pred = L.predict(net, inst)
        orders = [(pl_sample(un, rng), pl_sample(vn, rng)) for _ in range(a.k)]
        greedy = (np.argsort(-un, kind="stable"), np.argsort(-vn, kind="stable"))
        jobs = [(inst, pred, o, a.fit, a.pack, int(rng.integers(1 << 30))) for o in orders + [greedy]]
        res = pool.map(sample_cost, jobs)
        cost = np.array([r[0] for r in res[:-1]])
        g_cost = res[-1][0]
        if cost.std() > 1e-6:
            adv = torch.tensor((cost.mean() - cost) / (cost.std() + 1e-6), dtype=torch.float32)
            logp = torch.stack([pl_logp(u, gp) + pl_logp(v, gm) for gp, gm in orders])
            pg = -(adv * logp).mean() / n
        else:
            pg = torch.zeros(())
        bc, _, _ = L.loss_fn(net, inst, sol)
        ((pg + a.bc * bc) / a.batch).backward()
        seen += 1
        log.append((cost.mean(), cost.min(), g_cost, float((cost < g_cost).mean())))
        if seen % a.batch == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        if seen % 50 == 0:
            m = np.array(log)
            print(f"{seen} designs, {(time.time() - t0) / 60:.0f} min: sample mean {m[:, 0].mean():.4f} best-of-k "
                  f"{m[:, 1].mean():.4f} greedy {m[:, 2].mean():.4f} share of samples beating greedy {m[:, 3].mean():.2f}",
                  flush=True)
            log = []
        if seen % a.val_every == 0:
            vw, vm, nf = validate()
            tag = ""
            snap = Path(a.out).with_name(Path(a.out).stem + f"_{seen:06d}.pt")
            torch.save({"state": net.state_dict(), "cfg": cfg, "val_rl": vw, "seen": seen}, snap)
            if vw < best:
                best = vw
                torch.save({"state": net.state_dict(), "cfg": cfg, "val_rl": vw, "seen": seen}, a.out)
                tag = " (saved)"
            print(f"  val weighted {vw:.4f} mean {vm:.4f} infeasible {nf}{tag}", flush=True)
    vw, vm, nf = validate()
    if vw < best:
        best = vw
        torch.save({"state": net.state_dict(), "cfg": cfg, "val_rl": vw, "seen": seen}, a.out)
    print(f"done: {seen} designs, final val {vw:.4f}, best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()

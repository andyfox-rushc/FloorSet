"""Train the repair net (10-16): the relation placer (rl/learned_placer.py) with
each block's current position and size as extra inputs, trained to predict
the ground truth's centres, shapes and pairwise relations from a flawed layout.

Why: started from the ground truth's sequence pair the packer reaches ~1.00,
but 2 randomly misplaced blocks already cost 0.02-0.30 and 94% pair accuracy
(the relation nets) is about 5 of them; the packer can't undo such errors.

Input layouts, mixed per design:
  corrupted  ground-truth sequence pair, k = 0..8 blocks moved to random places,
             soft shapes x exp(N(0, 0.2)), packed from the origin
  own        the base net's own prediction packed (what the solver hands it)
Starts from --init (placer_r.pt) with the new input weights at zero, so step 0
is exactly the base net.

  python scripts/train_repair.py --hours 8 --threads 6 --out checkpoints/repair.pt
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


def gt_positions(sol, n):
    s = sol[:n].double().numpy()
    return [(float(s[i, 2]), float(s[i, 3]), float(s[i, 0]), float(s[i, 1])) for i in range(n)]


def corrupted_layout(inst, sol, rng, kmax=8, sigma=0.2, k=None, full=False):
    """k = exact number of moved blocks (else 0..kmax); full also returns ((gp, gm), overlap)."""
    from rl import packer as K, repack as R
    n = inst.block_count
    P = R._problem(inst, gt_positions(sol, n))
    p = P["pos"]
    gp, gm = K.sequence_pair(p[:, 0], p[:, 1], p[:, 2], p[:, 3])
    gp, gm = list(gp), list(gm)
    mov = np.where(~P["preplaced"])[0]
    for _ in range((int(rng.integers(0, kmax + 1)) if k is None else k) if len(mov) else 0):
        a = int(rng.choice(mov))
        for s in (gp, gm):
            s.remove(a)
            s.insert(int(rng.integers(0, n)), a)
    gp, gm = np.array(gp), np.array(gm)
    soft = ~P["fixed"] & ~P["preplaced"]
    f = np.exp(rng.normal(0, sigma, n))
    w = np.where(soft, p[:, 2] * f, p[:, 2])
    h = np.where(soft, np.maximum(P["area"], 0) / np.maximum(w, 1e-9), p[:, 3])
    Bt = K._Batch(P, 1.0, 1.0, (0.0, 0.0))
    x, y, over = Bt.pack(gm[None], np.argsort(gp)[None], np.argsort(gm)[None], w[None], h[None])
    pos = [(float(x[0, i]), float(y[0, i]), float(w[i]), float(h[i])) for i in range(n)]
    return (pos, (gp, gm), float(over[0])) if full else pos


def own_layout(inst, base, use_rel=True):
    from rl import learned_placer as L
    pos, _, _ = L.to_layout(inst, base, use_rel=use_rel and base.rel)
    return pos


def widen_init(net, path):
    """Load a layout-free checkpoint into the layout net: wider input layers get zero
    columns for the new inputs, new modules start at zero (step 0 = the old net)."""
    st = torch.load(path, map_location="cpu")["state"]
    own = net.state_dict()
    for k, v in own.items():
        if k not in st:
            st[k] = torch.zeros_like(v)
        elif st[k].shape != v.shape:
            new = torch.zeros_like(v)
            new[:, :st[k].shape[1]] = st[k]
            st[k] = new
    net.load_state_dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="../")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default="checkpoints/placer_r.pt")
    ap.add_argument("--own", type=float, default=0.3, help="share of designs fed the base net's own layout")
    ap.add_argument("--sources", default="",
                    help="comma list of placer checkpoints whose packed layouts are the 'own' inputs (default: --init)")
    ap.add_argument("--rel-geom", action="store_true", help="pair geometry of the input layout into the relation head")
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="checkpoints/repair.pt")
    ap.add_argument("--resume", action="store_true", help="continue from <out>.state.pt (saved every --val-every)")
    ap.add_argument("--warm", default="", help="start from these layout-net weights instead of --init")
    ap.add_argument("--warm-min", type=float, default=0.0, help="minutes already trained (lr schedule offset)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    from rl.learned_placer import PlacerNet, loss_fn, load
    from train_placer import iter_training, val_set
    base = load(a.init)
    sources = [load(p) for p in a.sources.split(",") if p] or [base]
    cfg = dict(hidden=base.encoder.hidden_dim, layers=len(base.encoder.layers), norm=base.encoder.norm,
               rel=True, layout=True, rel_geom=a.rel_geom)
    net = PlacerNet(**cfg)
    widen_init(net, a.warm or a.init)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    state_out = a.out.replace(".pt", ".state.pt")
    rs = torch.load(state_out, map_location="cpu", weights_only=False) if a.resume else None
    if rs:
        net.load_state_dict(rs["state"])
        opt.load_state_dict(rs["opt"])
        rng.bit_generator.state = rs["rng"]
        torch.set_rng_state(rs["torch_rng"])
    val = val_set()
    vrng = np.random.default_rng(12345)
    val_own = [(i, s, own_layout(i, base)) for i, s in val]
    val_cor = [(i, s, corrupted_layout(i, s, vrng, kmax=5)) for i, s in val]

    def evaluate():
        net.eval()
        out = {}
        with torch.no_grad():
            for name, vs in (("own", val_own), ("corrupted", val_cor)):
                np.random.seed(0)       # same relation pairs every time
                r, acc = [], []
                for i, s, lay in vs:
                    r.append(loss_fn(net, i, s, lay))
                    acc.append(net._last_rel[1])
                out[name] = (sum(x[1] for x in r) / len(r), sum(acc) / len(acc))
        net.train()
        print("  val " + "  ".join(f"{k}: centre L1 {v[0]:.4f} rel acc {v[1]:.4f}" for k, v in out.items()), flush=True)
        return out

    cor_out = a.out.replace(".pt", "_cor.pt")
    if rs:      # resumed: best files on disk are already current
        best, best_cor, seen, done_s = rs["best"], rs["best_cor"], rs["seen"], rs["elapsed"]
        print(f"resumed at {seen} designs, {done_s / 60:.0f} min", flush=True)
    else:
        v = evaluate()
        best, best_cor, seen, done_s = v["own"][1], v["corrupted"][1], 0, a.warm_min * 60
        if a.warm:  # keep the earlier run's bests (and its design count)
            seen = torch.load(a.warm, map_location="cpu", weights_only=False).get("seen", 0)
            for f, k in ((a.out, "own"), (cor_out, "corrupted")):
                if Path(f).exists():
                    old = torch.load(f, map_location="cpu", weights_only=False)["val"][k][1]
                    best, best_cor = (max(best, old), best_cor) if k == "own" else (best, max(best_cor, old))
        else:
            torch.save({"state": net.state_dict(), "cfg": cfg, "seen": 0, "val": v}, a.out)
            torch.save({"state": net.state_dict(), "cfg": cfg, "seen": 0, "val": v}, cor_out)
    t0 = time.time() - done_s
    end = t0 + a.hours * 3600
    run = []
    # resumed runs reshuffle with a seed derived from progress instead of replaying the start
    data = iter_training(a.data_path, a.seed + 1000 * (seen // 1000) if rs else a.seed)
    opt.zero_grad()
    while time.time() < end:
        inst, sol = next(data)
        if inst.block_count < 2:
            continue
        own = rng.random() < a.own
        lay = own_layout(inst, sources[int(rng.integers(len(sources)))]) if own else corrupted_layout(inst, sol, rng)
        loss, c, asp = loss_fn(net, inst, sol, lay)
        (loss / a.batch).backward()
        run.append((c, net._last_rel[1], own))
        seen += 1
        if seen % a.batch == 0:
            frac = min((time.time() - t0) / (a.hours * 3600), 1.0)
            for g in opt.param_groups:
                g["lr"] = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        if seen % 1000 == 0:
            o = [x for x in run if x[2]]
            cr = [x for x in run if not x[2]]
            m = lambda xs, k: sum(x[k] for x in xs) / max(len(xs), 1)
            print(f"{seen} designs, {(time.time() - t0) / 60:.0f} min: train centre L1 {m(run, 0):.4f} "
                  f"rel acc own {m(o, 1):.3f} corrupted {m(cr, 1):.3f}", flush=True)
            run = []
        if seen % a.val_every == 0:
            v = evaluate()
            if v["own"][1] > best:
                best = v["own"][1]
                torch.save({"state": net.state_dict(), "cfg": cfg, "seen": seen, "val": v}, a.out)
                print("  (saved)", flush=True)
            if v["corrupted"][1] > best_cor:
                best_cor = v["corrupted"][1]
                torch.save({"state": net.state_dict(), "cfg": cfg, "seen": seen, "val": v}, cor_out)
                print("  (saved corrupted-best)", flush=True)
            torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "seen": seen, "elapsed": time.time() - t0,
                        "best": best, "best_cor": best_cor}, state_out)
    v = evaluate()
    if v["own"][1] > best:
        torch.save({"state": net.state_dict(), "cfg": cfg, "seen": seen, "val": v}, a.out)
    print(f"done: {seen} designs, best own rel acc {max(best, v['own'][1]):.4f}", flush=True)


if __name__ == "__main__":
    main()

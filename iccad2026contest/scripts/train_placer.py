"""Train the learned placer (rl/learned_placer.py) on ground-truth layouts.

  python scripts/train_placer.py --hours 6 --threads 4 --out checkpoints/placer.pt

Shuffled training files (rl/train.py's buffering idea, here also yielding
the ground truth), Adam with cosine decay, gradient over --batch designs;
every --val-every designs the loss on the 100 validation designs is
measured and the best checkpoint kept.
"""
import argparse
import glob
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import torch


def iter_training(data_path, seed, buffer_files=32):
    from rl.data import from_training_batch_item
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(data_path, "floorset_lite", "worker_*", "layouts*")))
    while True:
        rng.shuffle(files)
        buffer, nxt, since = [], 0, 0

        def load_next():
            nonlocal nxt
            c = torch.load(files[nxt])
            nxt += 1
            for li in range(len(c[0])):
                buffer.append((c[0][li][:, 0], c[1][li], c[2][li], c[3][li], c[0][li][:, 1:], c[6][li], c[5][li]))

        while nxt < min(buffer_files, len(files)):
            load_next()
        while buffer:
            j = rng.randrange(len(buffer))
            buffer[j], buffer[-1] = buffer[-1], buffer[j]
            area, b2b, p2b, pins, cons, metrics, sol = buffer.pop()
            inst = from_training_batch_item(area, b2b, p2b, pins, cons, metrics, sol)
            yield inst, sol
            since += 1
            if since >= 112 and nxt < len(files):
                load_next()
                since = 0


def val_set():
    import polish_bench as PB
    from rl.data import from_validation_sample
    PB._init()
    out = []
    for t in range(100):
        inst = from_validation_sample(PB.EV, t)
        _, gt = PB.EV._extract_baseline(t, PB.EV.dataset[t]["label"], inst.b2b_connectivity,
                                        inst.p2b_connectivity, inst.pins_pos, inst.block_count)
        sol = torch.tensor([[float(q[2]), float(q[3]), float(q[0]), float(q[1])] for q in gt])
        out.append((inst, sol))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="../")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--norm", action="store_true", help="normalized message passing + LayerNorm (10-08)")
    ap.add_argument("--rel", action="store_true", help="pairwise sequence-pair relation head (10-09)")
    ap.add_argument("--val-every", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default="checkpoints/placer.pt")
    ap.add_argument("--near-frac", type=float, default=None,
                    help="share of relation pairs from (nearly) touching blocks (10-18)")
    ap.add_argument("--select", default="centre", choices=["centre", "near"],
                    help="checkpoint selection: val centre L1, or val relation accuracy on near pairs")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    from rl import learned_placer as LP
    from rl.learned_placer import PlacerNet, loss_fn
    if a.near_frac is not None:
        LP.NEAR_FRAC = a.near_frac
    cfg = dict(hidden=a.hidden, layers=a.layers, norm=a.norm, rel=a.rel)
    net = PlacerNet(**cfg)
    if a.resume:      # strict=False: a new head (e.g. --rel) starts fresh on old weights
        missing = net.load_state_dict(torch.load(a.resume)["state"], strict=False)
        print("resume: missing", missing.missing_keys[:6], "unexpected", missing.unexpected_keys[:6], flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    val = val_set()
    t0, end = time.time(), time.time() + a.hours * 3600
    seen, best, run = 0, float("inf"), []

    near_acc = [0.0]

    def evaluate():
        net.eval()
        rel, full = [], []
        with torch.no_grad():
            r = []
            for i, s in val:
                r.append(loss_fn(net, i, s))
                if a.rel:
                    rel.append(net._last_rel[1])
                    if a.select == "near":
                        full.append(LP.rel_accuracy(net, i, s))
        net.train()
        if rel:
            print(f"  val relation accuracy {sum(rel) / len(rel):.3f}", flush=True)
        if full:
            near_acc[0] = sum(x[1] for x in full) / len(full)
            print(f"  val relation accuracy, all pairs {sum(x[0] for x in full) / len(full):.4f} "
                  f"near pairs {near_acc[0]:.4f}", flush=True)
        return sum(x[1] for x in r) / len(r), sum(x[2] for x in r) / len(r)

    lc, la = evaluate()
    print(f"start: val centre L1 {lc:.4f} aspect L1 {la:.4f}", flush=True)
    if a.select == "near":
        best = -near_acc[0]
    data = iter_training(a.data_path, a.seed)
    opt.zero_grad()
    while time.time() < end:
        inst, sol = next(data)
        if inst.block_count < 2:
            continue
        loss, c, asp = loss_fn(net, inst, sol)
        (loss / a.batch).backward()
        run.append((c, asp) + (net._last_rel if a.rel else ()))
        seen += 1
        if seen % a.batch == 0:
            frac = min((time.time() - t0) / (a.hours * 3600), 1.0)
            for g in opt.param_groups:
                g["lr"] = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        if seen % 1000 == 0:
            extra = (f" relation loss {sum(x[2] for x in run) / len(run):.3f} acc {sum(x[3] for x in run) / len(run):.3f}"
                     if a.rel else "")
            print(f"{seen} designs, {(time.time() - t0) / 60:.0f} min: train centre L1 "
                  f"{sum(x[0] for x in run) / len(run):.4f} aspect {sum(x[1] for x in run) / len(run):.4f}{extra}", flush=True)
            run = []
        if seen % a.val_every == 0:
            lc, la = evaluate()
            tag = ""
            score = -near_acc[0] if a.select == "near" else lc
            if score < best:
                best = score
                torch.save({"state": net.state_dict(), "cfg": cfg, "seen": seen, "val_centre_l1": lc}, a.out)
                tag = " (saved)"
            print(f"  val centre L1 {lc:.4f} aspect L1 {la:.4f}{tag}", flush=True)
    lc, la = evaluate()
    if (-near_acc[0] if a.select == "near" else lc) < best:
        torch.save({"state": net.state_dict(), "cfg": cfg, "seen": seen, "val_centre_l1": lc}, a.out)
    print(f"done: {seen} designs, final val centre L1 {lc:.4f}, best {min(best, lc):.4f}", flush=True)


if __name__ == "__main__":
    main()

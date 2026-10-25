"""Train the FloorSet-Prime centroid predictor (rl/prime_placer.py) on the
Prime training set (../PrimeTensorData, 1000 files x 1000 designs).

  python scripts/train_prime_placer.py --hours 8 --threads 3 --out checkpoints/prime_placer.pt

File 10 of every 10th config is held out: 200 of its designs are the
validation set used to keep the best checkpoint (the 100 test designs stay
unseen). Warm start from the Lite placer checkpoints/placer_d.pt by default.
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
import torch

from rl.learned_placer import PlacerNet
from rl.prime_placer import frame_of, lite_instance, loss_fn, targets


def _designs(path):
    data = torch.load(path, weights_only=False)
    labels = torch.load(path.replace("primedata_", "primelabel_"), weights_only=False)
    for d, l in zip(data, labels):
        area, cons = d[0][:, 0], d[0][:, 1:]
        yield area, d[1], d[2], d[3], cons, l[1]


def _sample(area, b2b, p2b, pins, cons, polys):
    n = len(area)
    tp = []
    for i in range(n):
        v = [(float(x), float(y)) for x, y in polys[i].tolist() if not (x == -1 and y == -1)]
        if cons[i, 1]:
            tp.append(v)
        elif cons[i, 0]:
            x0, y0 = min(p[0] for p in v), min(p[1] for p in v)
            tp.append([(x - x0, y - y0) for x, y in v])
        else:
            tp.append(None)
    inst = lite_instance(n, area.float(), b2b.float(), p2b.float(), pins.float(), cons.float(), tp)
    W, H = frame_of(pins.float())
    tgt = targets(polys, W, H)
    movable = cons[:, 1] == 0
    return inst, n, tgt, movable


def held_out(files):
    return [f for f in files if f.endswith("primedata_10.pth") and int(f.split("config_")[1].split("/")[0]) % 10 == 0]


def iter_training(files, seed, buffer_files=8):
    rng = random.Random(seed)
    while True:
        rng.shuffle(files)
        buf, nxt = [], 0
        while nxt < len(files) or buf:
            while len(buf) < buffer_files * 1000 and nxt < len(files):
                buf.extend(_designs(files[nxt]))
                nxt += 1
            j = rng.randrange(len(buf))
            buf[j], buf[-1] = buf[-1], buf[j]
            yield buf.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../PrimeTensorData")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", default="checkpoints/placer_d.pt")
    ap.add_argument("--val-every", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="checkpoints/prime_placer.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    files = sorted(glob.glob(os.path.join(a.data, "config_*", "primedata_*.pth")))
    hold = set(held_out(files))
    train = [f for f in files if f not in hold]
    val = []
    for f in sorted(hold):
        for k, d in enumerate(_designs(f)):
            if k >= 20:
                break
            val.append(_sample(*d))
    print(f"{len(train)} training files, {len(val)} validation designs", flush=True)
    ck = torch.load(a.resume, weights_only=False) if a.resume else None
    cfg = dict(ck["cfg"]) if ck else dict(hidden=256, layers=8, norm=True)
    cfg.pop("rel", None)
    net = PlacerNet(**cfg)
    if ck:
        net.load_state_dict(ck["state"], strict=False)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    def evaluate():
        net.eval()
        with torch.no_grad():
            v = sum(float(loss_fn(net, *s)) for s in val) / len(val)
        net.train()
        return v

    best = evaluate()
    print(f"start: val centroid L1 {best:.4f} (frame units, x + y)", flush=True)
    t0, end = time.time(), time.time() + a.hours * 3600
    seen, run = 0, []
    data = iter_training(train, a.seed)
    opt.zero_grad()
    while time.time() < end:
        s = _sample(*next(data))
        loss = loss_fn(net, *s)
        (loss / a.batch).backward()
        run.append(float(loss))
        seen += 1
        if seen % a.batch == 0:
            frac = min((time.time() - t0) / (a.hours * 3600), 1.0)
            for g in opt.param_groups:
                g["lr"] = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        if seen % 1000 == 0:
            print(f"{seen} designs, {(time.time() - t0) / 60:.0f} min: train L1 {sum(run) / len(run):.4f}", flush=True)
            run = []
        if seen % a.val_every == 0:
            v = evaluate()
            tag = ""
            if v < best:
                best = v
                torch.save({"state": net.state_dict(), "cfg": cfg, "seen": seen, "val_l1": v}, a.out)
                tag = " (saved)"
            print(f"  val centroid L1 {v:.4f}{tag}", flush=True)
    v = evaluate()
    if v < best:
        torch.save({"state": net.state_dict(), "cfg": cfg, "seen": seen, "val_l1": v}, a.out)
    print(f"done: {seen} designs, final val {v:.4f}, best {min(best, v):.4f}", flush=True)


if __name__ == "__main__":
    main()

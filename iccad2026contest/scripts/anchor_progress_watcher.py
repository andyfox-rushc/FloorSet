#!/usr/bin/env python3
"""
Watches checkpoints/history/ for new checkpoints at a fixed iteration
interval and, for each one, solves test-0/49/98 and saves
floorplan_test{N}_training_{Y}.jpg (Y = iteration number) -- a visual
convergence-over-training record, run alongside `python -m rl.train`
without modifying it (decoupled so a bug here can never affect training).

Usage:
    python scripts/anchor_progress_watcher.py --interval 1000 --poll-seconds 30
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image

from eval_checkpoint import solve_one, _regenerate_image
from iccad2026_evaluate import ContestEvaluator
from rl.networks import ActorCritic

ANCHOR_CASES = [0, 49, 98]
CKPT_RE = re.compile(r"policy_iter(\d+)\.pt$")


def find_new_iters(history_dir: Path, interval: int, done: set) -> list:
    found = []
    for f in history_dir.glob("policy_iter*.pt"):
        m = CKPT_RE.search(f.name)
        if not m:
            continue
        it = int(m.group(1))
        if it % interval == 0 and it not in done:
            found.append(it)
    return sorted(found)


def process_iter(ev, history_dir: Path, out_dir: Path, it: int, time_budget: float, anneal_budget: float):
    ckpt_path = history_dir / f"policy_iter{it:06d}.pt"
    net = ActorCritic()
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    print(f"[iter {it}] loaded {ckpt_path}")
    stranded_summary = {}
    for tid in ANCHOR_CASES:
        positions, m, stranded = solve_one(net, ev, tid, time_budget, anneal_budget, seed=tid)
        stranded_summary[tid] = stranded['stranded']
        print(f"  case{tid}: hpwl_gap={m.hpwl_gap:.3f} area_gap={m.area_gap:.3f} "
              f"vrel={m.violations_relative:.3f} cost={m.cost:.3f} "
              f"feasible={m.is_feasible} stranded={stranded['stranded']}")
        tag = f"training_{it}"
        _regenerate_image(ev, tid, positions, m, tag)
        png = out_dir / f"floorplan_test{tid}_{tag}.png"
        jpg = out_dir / f"case{tid}_iteration{it}.jpeg"
        Image.open(png).convert("RGB").save(jpg, "JPEG", quality=95)
        print(f"  saved {jpg}")
    print(f"[iter {it}] stranded summary: {stranded_summary}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history-dir", default="checkpoints/history")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--data-path", default="../")
    p.add_argument("--interval", type=int, default=1000)
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--time-budget", type=float, default=15.0)
    p.add_argument("--anneal-budget", type=float, default=8.0)
    p.add_argument("--stop-after-iter", type=int, default=None,
                    help="Exit once this iteration has been processed (default: run forever).")
    args = p.parse_args()

    history_dir = Path(args.history_dir)
    out_dir = Path(args.out_dir)
    ev = ContestEvaluator(args.data_path, verbose=False)
    ev._load_dataset()

    done = set()
    print(f"watching {history_dir} for multiples of {args.interval}, "
          f"polling every {args.poll_seconds}s")
    while True:
        for it in find_new_iters(history_dir, args.interval, done):
            try:
                process_iter(ev, history_dir, out_dir, it, args.time_budget, args.anneal_budget)
            except Exception as e:
                print(f"[iter {it}] FAILED: {e}", file=sys.stderr)
            done.add(it)
            if args.stop_after_iter is not None and it >= args.stop_after_iter:
                print(f"reached --stop-after-iter={args.stop_after_iter}, exiting")
                return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

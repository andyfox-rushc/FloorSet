"""Evaluates every training checkpoint at a multiple of --interval as it
appears in checkpoints/history/:
  1. full my_optimizer.py pipeline on the anchor designs (scripts/anchor_results.py:
     images + metrics) -> <out>/iter<N>/
  2. 100-case canvas_scan greedy/sampled score at utilization 0.65 and at the
     checkpoint's trained level -> <out>/iter<N>/scan.json
  3. one summary row appended to <out>/summary.md
Restart-safe: a checkpoint is skipped once <out>/iter<N>/done exists.

  python scripts/checkpoint_eval_watcher.py --start 4000
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
CKPT_RE = re.compile(r"policy_iter(\d+)\.pt$")
SCAN_CFG = "{u}:4:1:0:1:3"   # fine grid 4, aspect mask, no die anchor, shape fit, lookahead 3


def scan_summary(path, cfg):
    rows = json.load(open(path))[cfg]
    g = [r for r in rows if r["greedy_ok"]]
    avg = lambda k: sum(r[k] for r in g) / max(len(g), 1)
    return dict(sampled=sum(r["samp_ok"] for r in rows), sampled_n=4 * len(rows),
                greedy=len(g), cost=avg("cost"), hpwl_gap=avg("hpwl_gap"),
                area_gap=avg("area_gap"), vrel=avg("violations_relative"))


def evaluate(it, ckpt, out_dir, args):
    d = out_dir / f"iter{it}"
    d.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(str(ckpt) + ".meta.json"))
    util = float(meta.get("utilization", 0.65))
    env = dict(os.environ, FLOORSET_CHECKPOINT=str(ckpt))
    print(f"[iter {it}] anchor designs (util {util})", flush=True)
    subprocess.run([sys.executable, "-u", "scripts/anchor_results.py", "--test-ids", *map(str, args.designs),
                    "--out-dir", str(d), "--label", f"iter {it}, trained util {util:.3f}"],
                   cwd=ROOT, env=env, check=True, stdout=open(d / "anchor.log", "w"), stderr=subprocess.STDOUT)
    levels = sorted({0.65, round(util, 4)})
    cfgs = [SCAN_CFG.format(u=u) for u in levels]
    print(f"[iter {it}] 100-case scan {cfgs}", flush=True)
    subprocess.run([sys.executable, "-u", "scripts/canvas_scan.py", "--configs", *cfgs, "--random", "0",
                    "--sampled", "4", "--workers", str(args.workers), "--checkpoint", str(ckpt),
                    "--out", str(d / "scan.json")],
                   cwd=ROOT, check=True, stdout=open(d / "scan.log", "w"), stderr=subprocess.STDOUT)

    anchors = {r["design"]: r for r in json.load(open(d / "results.json"))}
    scans = {cfg: scan_summary(d / "scan.json", cfg) for cfg in cfgs}
    summary = out_dir / "summary.md"
    if not summary.exists():
        summary.write_text(
            "| Iter | Trained util | Design 0 cost | Design 49 cost | Design 98 cost "
            "| 100-case @0.65: sampled finish / greedy cost | 100-case @trained util: sampled finish / greedy cost |\n"
            "|------|--------------|---------------|----------------|----------------"
            "|------------------------------------------------|-----------------------------------------------------|\n")
    s65 = scans[SCAN_CFG.format(u=0.65)]
    su = scans[SCAN_CFG.format(u=round(util, 4))]
    cell = lambda s: f"{100 * s['sampled'] / s['sampled_n']:.0f}% / {s['cost']:.3f}"
    costs = [f"{anchors[t]['cost']:.3f}" if t in anchors else "-" for t in (0, 49, 98)]
    with open(summary, "a") as f:
        f.write(f"| {it} | {util:.3f} | {' | '.join(costs)} | {cell(s65)} | {cell(su)} |\n")
    (d / "done").write_text(json.dumps(dict(anchors={k: v["cost"] for k, v in anchors.items()}, scans=scans)))
    print(f"[iter {it}] done: designs {costs}, @0.65 {cell(s65)}, @{util:.3f} {cell(su)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", default="checkpoints/history")
    ap.add_argument("--out-dir", default="results_checkpoints")
    ap.add_argument("--interval", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--designs", type=int, nargs="+", default=[0, 49, 98])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    args = ap.parse_args()
    history, out_dir = ROOT / args.history_dir, ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"watching {history} for multiples of {args.interval} from {args.start}", flush=True)
    while True:
        todo = []
        for f in history.glob("policy_iter*.pt"):
            m = CKPT_RE.search(f.name)
            it = int(m.group(1)) if m else -1
            if (it >= args.start and it % args.interval == 0 and Path(str(f) + ".meta.json").exists()
                    and not (out_dir / f"iter{it}" / "done").exists()):
                todo.append((it, f))
        for it, f in sorted(todo):
            try:
                evaluate(it, f, out_dir, args)
            except Exception as e:  # keep watching; the next poll retries
                print(f"[iter {it}] FAILED: {e}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

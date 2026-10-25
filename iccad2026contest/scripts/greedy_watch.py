#!/usr/bin/env python3
"""Noise-free convergence curve: for every <history-dir>/policy_iter<N>.pt with
N % interval == 0, a greedy rollout (no fine-tune, no time budget, so the
result depends only on the checkpoint) on the 100 validation designs at the
checkpoint's own canvas utilization, via scripts/frame_scan.py in a
subprocess that inherits this env (FLOORSET_FRAME_UTIL etc.). Writes
<out>/greedy.csv and <out>/greedy_progress.png; prints one line per checkpoint.

  python scripts/greedy_watch.py --history-dir checkpoints/scratch/history --out-dir results_scratch
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
CKPT_RE = re.compile(r"policy_iter(\d+)\.pt$")
FIELDS = ["iter", "util", "finish", "cost", "hpwl_gap", "area_gap", "vrel"]


def plot(out, rows, ref):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharex=True)
    for ax, key, title in zip(axes, ("cost", "hpwl_gap", "area_gap", "finish"),
                              ("Greedy cost (finished designs)", "HPWL gap", "Area gap", "Finish rate")):
        xs = [r["iter"] for r in rows]
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, color="#2a78d6", linewidth=2, marker="o", markersize=4)
        if ref and key in ref:
            ax.axhline(ref[key], color="#eb6834", linewidth=1.2, linestyle=":")
            ax.annotate("installed 12000", (xs[0] if xs else 0, ref[key]), xytext=(0, 4),
                        textcoords="offset points", fontsize=8, color="#333333")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("training iteration")
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("100 validation designs, greedy rollout, no fine-tune (lower is better except finish rate)",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "greedy_progress.png", dpi=130)
    plt.close(fig)


def scan(ckpt, util, tmp, frame=None):
    frame = frame or os.environ.get("FLOORSET_FRAME_UTIL", "none") or "none"
    cmd = [sys.executable, "-u", "scripts/frame_scan.py", "--frames", frame, "--util", str(util),
           "--workers", "2", "--checkpoint", str(ckpt), "--out", str(tmp)]
    env = dict(os.environ)
    env.pop("FLOORSET_FRAME_UTIL", None)   # frame_scan passes the frame explicitly
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        return None
    rs = json.load(open(tmp))
    ok = [x for x in rs if x["finished"]]
    avg = lambda k: sum(x[k] for x in ok) / max(len(ok), 1)
    return dict(finish=len(ok) / len(rs), cost=avg("cost"), hpwl_gap=avg("hpwl_gap"),
                area_gap=avg("area_gap"), vrel=avg("vrel"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--interval", type=int, default=1000)
    p.add_argument("--poll-seconds", type=float, default=120)
    p.add_argument("--stop-after-iter", type=int, default=20000)
    p.add_argument("--reference", default="checkpoints/policy.pt",
                   help="checkpoint scored once (no frame, its own util) as the dotted line")
    a = p.parse_args()
    hist, out = Path(a.history_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if (out / "greedy.csv").exists():
        rows = [{k: (int(v) if k == "iter" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(open(out / "greedy.csv"))]
    done = {r["iter"] for r in rows}
    ref = None
    ref_path = out / "greedy_reference.json"
    if ref_path.exists():
        ref = json.load(open(ref_path))
    elif a.reference and (ROOT / a.reference).exists():
        meta = json.load(open(ROOT / (a.reference + ".meta.json")))
        # Installed policy was trained without the live frame: score it without.
        ref = scan(ROOT / a.reference, meta.get("utilization", 0.7), out / "greedy_ref_raw.json", "none")
        if ref:
            json.dump(ref, open(ref_path, "w"))
            print(f"reference {a.reference}: cost {ref['cost']:.3f} hpwl {ref['hpwl_gap']:.3f} "
                  f"area {ref['area_gap']:.3f} finish {ref['finish']:.0%}", flush=True)
    while True:
        its = sorted(int(m.group(1)) for f in hist.glob("policy_iter*.pt")
                     if (m := CKPT_RE.search(f.name)) and int(m.group(1)) % a.interval == 0)
        for it in its:
            meta_f = hist / f"policy_iter{it:06d}.pt.meta.json"
            if it in done or not meta_f.exists():
                continue
            util = json.load(open(meta_f)).get("utilization", 0.7)
            res = scan(hist / f"policy_iter{it:06d}.pt", util, out / f"greedy_iter{it:06d}.json")
            done.add(it)
            if res is None:
                print(f"greedy iter {it}: FAILED", flush=True)
                continue
            rows.append(dict(iter=it, util=util, **res))
            rows.sort(key=lambda r: r["iter"])
            with open(out / "greedy.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            plot(out, rows, ref)
            print(f"greedy iter {it}: cost {res['cost']:.3f} hpwl {res['hpwl_gap']:.3f} "
                  f"area {res['area_gap']:.3f} finish {res['finish']:.0%} (util {util})", flush=True)
        if a.stop_after_iter is not None and a.stop_after_iter in done:
            print("GREEDY_WATCH_DONE", flush=True)
            return
        time.sleep(a.poll_seconds)


if __name__ == "__main__":
    main()

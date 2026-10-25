#!/usr/bin/env python3
"""Anchor report per training snapshot: for every new <history-dir>/
policy_iter<N>.pt with N % interval == 0, run scripts/anchor_results.py
(full contest pipeline, designs 0/49/98) --repeats times in subprocesses that
inherit this process's env (FLOORSET_* placement settings); per design the
median-cost run is reported (single 2 s-fine-tune runs vary by up to ~0.4
cost under load) and its image kept. Then refresh
  <out>/iter<N>/design<d>_layout.png      per-snapshot layouts (median run;
                                          all runs in <out>/iter<N>/run<k>/)
  <out>/images/design<d>_iter<N>.png      same images, flat for browsing
  <out>/summary.csv, <out>/summary.md     one row per (iter, design)
  <out>/progress.png                      cost / HPWL gap / area gap vs iter
Each new result line is also printed (stdout) so a monitor can relay it.

  python scripts/anchor_watch.py --history-dir checkpoints/scratch/history --out-dir results_scratch
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
CKPT_RE = re.compile(r"policy_iter(\d+)\.pt$")
DESIGNS = [0, 49, 98]
COLORS = {0: "#2a78d6", 49: "#eb6834", 98: "#1baf7a"}   # validated categorical slots 1-3
FIELDS = ["iter", "design", "cost", "hpwl_gap", "area_gap", "vrel", "boundary_viol",
          "grouping_viol", "hpwl", "bbox_area", "runtime_s"]


def load_rows(path):
    if not path.exists():
        return []
    with open(path) as fh:
        return [{k: (int(v) if k in ("iter", "design", "boundary_viol", "grouping_viol") else float(v))
                 for k, v in r.items()} for r in csv.DictReader(fh)]


def write_reports(out, rows, baseline):
    rows.sort(key=lambda r: (r["iter"], r["design"]))
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    lines = ["| Iter  | Design | Cost  | HPWL gap | Area gap | vrel  | Bnd / grp | Time s |",
             "|-------|--------|-------|----------|----------|-------|-----------|--------|"]
    for r in rows:
        lines.append(f"| {r['iter']:>5} | {r['design']:>6} | {r['cost']:.3f} | {r['hpwl_gap']:>8.3f} | "
                     f"{r['area_gap']:>8.3f} | {r['vrel']:.3f} | {r['boundary_viol']:>3} / {r['grouping_viol']:<3} | "
                     f"{r['runtime_s']:>6.1f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, key, title in zip(axes, ("cost", "hpwl_gap", "area_gap"),
                              ("Contest cost", "HPWL gap", "Area gap")):
        for d in DESIGNS:
            pts = [(r["iter"], r[key]) for r in rows if r["design"] == d]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=COLORS[d], linewidth=2, marker="o", markersize=4, label=f"design {d}")
            ax.annotate(f"{d}", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points",
                        color="#333333", fontsize=9, va="center")
            if baseline and key in baseline.get(d, {}):
                ax.axhline(baseline[d][key], color=COLORS[d], linewidth=1, linestyle=":", alpha=0.8)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("training iteration")
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    note = "dotted: installed iter 12000" if baseline else ""
    fig.suptitle(f"Anchor designs over training (lower is better)   {note}", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "progress.png", dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--interval", type=int, default=500)
    p.add_argument("--poll-seconds", type=float, default=60)
    p.add_argument("--stop-after-iter", type=int, default=20000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--baseline", default="results_20261002_base/results.json",
                   help="anchor results.json drawn as dotted reference lines")
    a = p.parse_args()
    hist, out = Path(a.history_dir), Path(a.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = load_rows(out / "summary.csv")
    done = {r["iter"] for r in rows}
    baseline = {}
    bpath = ROOT / a.baseline
    if bpath.exists():
        baseline = {r["design"]: r for r in json.load(open(bpath))}
    while True:
        its = sorted(int(m.group(1)) for f in hist.glob("policy_iter*.pt")
                     if (m := CKPT_RE.search(f.name)) and int(m.group(1)) % a.interval == 0)
        for it in its:
            meta = hist / f"policy_iter{it:06d}.pt.meta.json"
            if it in done or not meta.exists():
                continue
            d = out / f"iter{it:06d}"
            env = dict(__import__("os").environ, FLOORSET_CHECKPOINT=str(hist / f"policy_iter{it:06d}.pt"))
            runs, err = [], None
            for k in range(a.repeats):
                cmd = [sys.executable, "-u", "scripts/anchor_results.py", "--out-dir", str(d / f"run{k}"),
                       "--label", f"iter {it} run {k}", "--test-ids", *map(str, DESIGNS)]
                r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
                res = d / f"run{k}" / "results.json"
                if r.returncode != 0 or not res.exists():
                    err = r.stderr.strip().splitlines()[-1:]
                    continue
                runs.extend(json.load(open(res)))
            if not runs:
                print(f"iter {it}: anchor run FAILED: {err}", flush=True)
                done.add(it)
                continue
            for dz in DESIGNS:
                xs = sorted((x for x in runs if x["design"] == dz), key=lambda x: x["cost"])
                if not xs:
                    continue
                x = xs[len(xs) // 2]
                rows.append({k: x[k] for k in FIELDS if k != "iter"} | {"iter": it})
                shutil.copy(x["image"], d / f"design{dz}_layout.png")
                shutil.copy(x["image"], out / "images" / f"design{dz}_iter{it:06d}.png")
            done.add(it)
            write_reports(out, rows, baseline)
            now = sorted((r for r in rows if r["iter"] == it), key=lambda r: r["design"])
            print(f"iter {it}: " + "  ".join(f"d{r['design']} cost {r['cost']:.3f} hpwl {r['hpwl_gap']:.3f} "
                                             f"area {r['area_gap']:.3f}" for r in now), flush=True)
        if a.stop_after_iter is not None and a.stop_after_iter in done:
            print("ANCHOR_WATCH_DONE", flush=True)
            return
        time.sleep(a.poll_seconds)


if __name__ == "__main__":
    main()

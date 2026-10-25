"""Solve validation designs with the real contest pipeline (my_optimizer.py:
fine-tune + compaction + anneal), then write per-design metrics and an image
of ground truth vs our layout on the same axes (pins and pin box drawn).

  python scripts/anchor_results.py --test-ids 0 49 98 --out-dir results_20260926
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution


def draw(tid, block_count, target_pos, positions, pins, m, label, out_png):
    colors = plt.cm.tab20(np.linspace(0, 1, block_count))
    valid = pins[(pins[:, 0] != -1) | (pins[:, 1] != -1)].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    panels = [(axes[0], target_pos, f"Design {tid} - ground truth\n"
               f"HPWL {m.hpwl_baseline:.1f}   bbox area {m.bbox_area_baseline:.1f}"),
              (axes[1], positions, f"Design {tid} - ours ({label})\n"
               f"HPWL {m.hpwl_total:.1f}   bbox area {m.bbox_area:.1f}   cost {m.cost:.3f}\n"
               f"hpwl_gap {m.hpwl_gap:.3f}  area_gap {m.area_gap:.3f}  "
               f"vrel {m.violations_relative:.3f}  feasible {m.is_feasible}")]
    for ax, rects, title in panels:
        for i in range(block_count):
            x, y, w, h = [float(v) for v in rects[i]]
            ax.add_patch(mpatches.Rectangle((x, y), w, h, facecolor=colors[i],
                                            edgecolor="black", alpha=0.7))
            ax.text(x + w / 2, y + h / 2, str(i), ha="center", va="center", fontsize=6)
        if len(valid):
            ax.scatter(valid[:, 0], valid[:, 1], s=6, c="red", marker="x", label="pins")
            ax.add_patch(mpatches.Rectangle((0, 0), valid[:, 0].max(), valid[:, 1].max(),
                                            fill=False, edgecolor="red", linestyle="--",
                                            label="pin box"))
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.autoscale()
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-ids", type=int, nargs="+", default=[0, 49, 98])
    p.add_argument("--out-dir", default="results")
    p.add_argument("--label", default="")
    a = p.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ev = ContestEvaluator("../", verbose=False)
    ev._load_dataset()
    optimizer = ev._load_optimizer("my_optimizer.py")
    rows = []
    for tid in a.test_ids:
        sample = ev.dataset[tid]
        area_target, b2b, p2b, pins, constraints = sample["input"]
        n = int((area_target != -1).sum().item())
        baseline, target_pos = ev._extract_baseline(tid, sample["label"], b2b, p2b, pins, n)
        result = ev.evaluate("my_optimizer.py", test_ids=[tid])
        tr = result.test_results[0]
        # runtime=1.0 matches the evaluator's neutral local RuntimeFactor.
        m = evaluate_solution({"positions": tr.positions, "runtime": 1.0},
                              baseline, constraints, b2b, p2b, pins, area_target, target_pos,
                              median_runtime=1.0)
        png = out_dir / f"design{tid}_layout.png"
        draw(tid, n, target_pos, tr.positions, pins, m, a.label, png)
        row = dict(design=tid, blocks=n, cost=tr.cost, hpwl=m.hpwl_total,
                   hpwl_gt=m.hpwl_baseline, hpwl_gap=m.hpwl_gap, bbox_area=m.bbox_area,
                   bbox_area_gt=m.bbox_area_baseline, area_gap=m.area_gap,
                   vrel=m.violations_relative, boundary_viol=m.boundary_violations,
                   grouping_viol=m.grouping_violations, mib_viol=m.mib_violations,
                   feasible=tr.is_feasible, runtime_s=tr.runtime_seconds, image=str(png),
                   positions=[list(map(float, r)) for r in tr.positions])
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "positions"}), flush=True)
    json.dump(rows, open(out_dir / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()

"""Draws our optimizer's solution next to the ground-truth floorplan.

Usage: python scripts/visualize_solution.py --test-id 0
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution

p = argparse.ArgumentParser()
p.add_argument("--test-id", type=int, default=0)
p.add_argument("--data-path", default="../")
p.add_argument("--optimizer", default="my_optimizer.py")
args = p.parse_args()

evaluator = ContestEvaluator(args.data_path, verbose=False)
evaluator._load_dataset()
optimizer = evaluator._load_optimizer(args.optimizer)

sample = evaluator.dataset[args.test_id]
inputs, labels = sample["input"], sample["label"]
area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
block_count = int((area_target != -1).sum().item())

baseline, target_pos = evaluator._extract_baseline(
    args.test_id, labels, b2b_conn, p2b_conn, pins_pos, block_count
)

opt_target_pos = torch.full((block_count, 4), -1.0)
if target_pos is not None and constraints is not None:
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(block_count):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_preplaced = nc > 1 and constraints[i, 1] != 0
        if is_preplaced:
            tx, ty, tw, th = target_pos[i]
            opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
        elif is_fixed:
            _, _, tw, th = target_pos[i]
            opt_target_pos[i, 2] = tw
            opt_target_pos[i, 3] = th

positions = optimizer.solve(
    block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints, opt_target_pos
)

metrics = evaluate_solution(
    {"positions": positions, "runtime": 0.0}, baseline, constraints,
    b2b_conn, p2b_conn, pins_pos, area_target, target_pos, median_runtime=1.0,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
colors = plt.cm.tab20(np.linspace(0, 1, block_count))

# Ground truth (left)
ax = axes[0]
ax.set_title(f"Test {args.test_id} - Ground Truth ({block_count} blocks)")
if target_pos is not None:
    for i in range(block_count):
        tx, ty, tw, th = [float(v) for v in target_pos[i]]
        rect = mpatches.Rectangle((tx, ty), tw, th, fill=True,
                                   facecolor=colors[i], edgecolor="black", alpha=0.7)
        ax.add_patch(rect)
        ax.text(tx + tw / 2, ty + th / 2, str(i), ha="center", va="center", fontsize=7)
ax.autoscale()
ax.set_aspect("equal")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Our solution (right)
ax = axes[1]
ax.set_title(
    f"Test {args.test_id} - Our Solution\n"
    f"hpwl_gap={metrics.hpwl_gap:.3f} area_gap={metrics.area_gap:.3f} "
    f"vrel={metrics.violations_relative:.3f} cost={metrics.cost:.3f} "
    f"feasible={metrics.is_feasible}"
)
for i, (x, y, w, h) in enumerate(positions):
    rect = mpatches.Rectangle((x, y), w, h, fill=True,
                               facecolor=colors[i], edgecolor="black", alpha=0.7)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, str(i), ha="center", va="center", fontsize=7)
ax.autoscale()
ax.set_aspect("equal")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
out_path = f"floorplan_test{args.test_id}.png"
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")

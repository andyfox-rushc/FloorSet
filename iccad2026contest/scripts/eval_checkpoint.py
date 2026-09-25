"""
Checkpoint evaluation harness for tracking training progress, per
algorithm.md's reward-fix verification plan: reports the 20-case dev-set
average (hpwl_gap/area_gap/vrel/cost/feasibility), the stranded-component
rate across that same set, and three fixed anchor cases (0, 49, 98 --
understood in detail from this project's own diagnosis sessions, so a
regression there is a sharper trust signal than an aggregate average) with
individual metrics and a regenerated visualization image each.

Usage:
    python scripts/eval_checkpoint.py --checkpoint checkpoints/policy.pt --tag iter5000
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from detect_stranded_components import detect_stranded
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from rl.anneal import anneal_polish
from rl.data import from_solve_args
from rl.finetune import finetune_and_solve
from rl.networks import ActorCritic

DEV_SET = list(range(20))
ANCHOR_CASES = [0, 49, 98]


def _build_instance(evaluator, tid):
    sample = evaluator.dataset[tid]
    inputs, labels = sample["input"], sample["label"]
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    block_count = int((area_target != -1).sum().item())
    baseline, target_pos = evaluator._extract_baseline(tid, labels, b2b_conn, p2b_conn,
                                                         pins_pos, block_count)
    opt_target_pos = torch.full((block_count, 4), -1.0)
    for i in range(block_count):
        if constraints[i, 1] != 0:
            opt_target_pos[i] = torch.tensor(list(target_pos[i]))
        elif constraints[i, 0] != 0:
            opt_target_pos[i, 2] = target_pos[i][2]
            opt_target_pos[i, 3] = target_pos[i][3]
    instance = from_solve_args(block_count, area_target, b2b_conn, p2b_conn, pins_pos,
                                constraints, opt_target_pos)
    return instance, baseline, target_pos, (area_target, b2b_conn, p2b_conn, pins_pos, constraints)


def solve_one(net, evaluator, tid, time_budget, anneal_budget, seed):
    instance, baseline, target_pos, raw = _build_instance(evaluator, tid)
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = raw
    torch.manual_seed(seed)
    positions, _ = finetune_and_solve(instance, net=net, time_budget=time_budget,
                                       max_iterations=150, episodes_per_iter=6, ppo_epochs=3,
                                       seed=seed, verbose=False)
    positions = anneal_polish(instance, positions, time_budget=anneal_budget, seed=seed)
    m = evaluate_solution({"positions": positions, "runtime": 0.0}, baseline, constraints,
                           b2b_conn, p2b_conn, pins_pos, area_target, target_pos, median_runtime=1.0)
    stranded = detect_stranded(positions)
    return positions, m, stranded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/policy.pt")
    p.add_argument("--data-path", default="../")
    p.add_argument("--time-budget", type=float, default=15.0)
    p.add_argument("--anneal-budget", type=float, default=8.0)
    p.add_argument("--tag", default=None, help="label for this eval, e.g. an iteration count")
    p.add_argument("--out", default="logs/checkpoint_eval_history.jsonl")
    p.add_argument("--skip-images", action="store_true")
    args = p.parse_args()

    evaluator = ContestEvaluator(args.data_path, verbose=False)
    evaluator._load_dataset()
    net = ActorCritic()
    if Path(args.checkpoint).exists():
        net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"loaded checkpoint: {args.checkpoint}")
    else:
        print(f"WARNING: checkpoint {args.checkpoint} not found, evaluating an untrained policy")

    dev_rows = []
    for tid in DEV_SET:
        _, m, stranded = solve_one(net, evaluator, tid, args.time_budget, args.anneal_budget, seed=tid)
        dev_rows.append(dict(tid=tid, hpwl_gap=m.hpwl_gap, area_gap=m.area_gap,
                              vrel=m.violations_relative, cost=m.cost, feasible=m.is_feasible,
                              stranded=stranded["stranded"]))
        print(f"  dev tid={tid:>3} hpwl={m.hpwl_gap:.3f} area={m.area_gap:.3f} "
              f"vrel={m.violations_relative:.3f} cost={m.cost:.3f} feas={m.is_feasible} "
              f"stranded={stranded['stranded']}")

    def avg(key):
        return sum(r[key] for r in dev_rows) / len(dev_rows)

    dev_summary = dict(
        avg_hpwl_gap=avg("hpwl_gap"), avg_area_gap=avg("area_gap"), avg_vrel=avg("vrel"),
        avg_cost=avg("cost"), feasible_count=sum(r["feasible"] for r in dev_rows),
        stranded_count=sum(r["stranded"] for r in dev_rows), n=len(dev_rows),
    )
    print(f"\n=== dev-set (n={dev_summary['n']}) === "
          f"avg_hpwl_gap={dev_summary['avg_hpwl_gap']:.4f} "
          f"avg_area_gap={dev_summary['avg_area_gap']:.4f} avg_vrel={dev_summary['avg_vrel']:.4f} "
          f"avg_cost={dev_summary['avg_cost']:.4f} "
          f"feasible={dev_summary['feasible_count']}/{dev_summary['n']} "
          f"stranded={dev_summary['stranded_count']}/{dev_summary['n']}")

    anchor_rows = []
    for tid in ANCHOR_CASES:
        positions, m, stranded = solve_one(net, evaluator, tid, args.time_budget,
                                            args.anneal_budget, seed=tid)
        anchor_rows.append(dict(tid=tid, hpwl_gap=m.hpwl_gap, area_gap=m.area_gap,
                                 vrel=m.violations_relative, cost=m.cost, feasible=m.is_feasible,
                                 stranded=stranded["stranded"],
                                 stranded_detail=stranded["components"]))
        print(f"anchor tid={tid:>3} hpwl={m.hpwl_gap:.3f} area={m.area_gap:.3f} "
              f"vrel={m.violations_relative:.3f} cost={m.cost:.3f} feas={m.is_feasible} "
              f"stranded={stranded['stranded']}")
        if not args.skip_images:
            _regenerate_image(evaluator, tid, positions, m, args.tag)

    record = dict(timestamp=time.time(), tag=args.tag, checkpoint=args.checkpoint,
                  dev_summary=dev_summary, dev_rows=dev_rows, anchor_rows=anchor_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nappended eval record to {out_path}")


def _regenerate_image(evaluator, tid, positions, metrics, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    sample = evaluator.dataset[tid]
    inputs, labels = sample["input"], sample["label"]
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    block_count = int((area_target != -1).sum().item())
    _, target_pos = evaluator._extract_baseline(tid, labels, b2b_conn, p2b_conn, pins_pos, block_count)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, block_count))
    ax = axes[0]
    ax.set_title(f"Test {tid} - Ground Truth ({block_count} blocks)")
    for i in range(block_count):
        tx, ty, tw, th = [float(v) for v in target_pos[i]]
        ax.add_patch(mpatches.Rectangle((tx, ty), tw, th, fill=True, facecolor=colors[i],
                                         edgecolor="black", alpha=0.7))
        ax.text(tx + tw / 2, ty + th / 2, str(i), ha="center", va="center", fontsize=7)
    ax.autoscale(); ax.set_aspect("equal")

    ax = axes[1]
    suffix = f" [{tag}]" if tag else ""
    ax.set_title(f"Test {tid}{suffix}\nhpwl_gap={metrics.hpwl_gap:.3f} "
                 f"area_gap={metrics.area_gap:.3f} vrel={metrics.violations_relative:.3f} "
                 f"cost={metrics.cost:.3f} feasible={metrics.is_feasible}")
    for i, (x, y, w, h) in enumerate(positions):
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=True, facecolor=colors[i],
                                         edgecolor="black", alpha=0.7))
        ax.text(x + w / 2, y + h / 2, str(i), ha="center", va="center", fontsize=7)
    ax.autoscale(); ax.set_aspect("equal")
    plt.tight_layout()

    suffix = f"_{tag}" if tag else ""
    out = f"floorplan_test{tid}{suffix}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()

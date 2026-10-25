"""Check an official evaluation against the ground truth, design by design.

Reads one or more result files written by
  python iccad2026_evaluate.py --evaluate my_optimizer.py --output RESULT.json
scores the ground-truth layout of each of the 100 validation designs with the
same evaluator, and prints per file: the official weighted score (exp(blocks/12)
weights), mean HPWL / area gaps and violations, how many designs beat the ground
truth's own score, and the designs with the largest weighted losses.

  python scripts/compare_with_gt.py official_eval_1011_pack200.json [RESULT.json ...]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def gt_costs(ev):
    """{design: (blocks, contest cost of the ground-truth layout)}."""
    from iccad2026_evaluate import evaluate_solution
    from rl.data import from_validation_sample
    from rl.reward import _target_positions_list
    out = {}
    for t in range(len(ev.dataset)):
        area, b2b, p2b, pins, cons = ev.dataset[t]["input"]
        n = int((area != -1).sum())
        _, tp = ev._extract_baseline(t, ev.dataset[t]["label"], b2b, p2b, pins, n)
        inst = from_validation_sample(ev, t)
        pos = [tuple(map(float, q)) for q in tp[:n]]
        s = evaluate_solution({"positions": pos, "runtime": 1.0}, inst.baseline_metrics, inst.constraints,
                              inst.b2b_connectivity, inst.p2b_connectivity, inst.pins_pos,
                              inst.area_targets, _target_positions_list(inst), median_runtime=1.0)
        out[t] = (n, s.cost)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data-path", default="../")
    ap.add_argument("--worst", type=int, default=8)
    a = ap.parse_args()
    from iccad2026_evaluate import ContestEvaluator
    ev = ContestEvaluator(a.data_path, verbose=False)
    ev._load_dataset()
    gt = gt_costs(ev)
    wt = {t: math.exp(n / 12) for t, (n, _) in gt.items()}
    W = sum(wt.values())
    print(f"ground truth as scored by the evaluator: weighted {sum(wt[t] * c for t, (_, c) in gt.items()) / W:.4f}")
    for f in a.results:
        r = {x["test_id"]: x for x in json.load(open(f))["test_results"]}
        mean = lambda k: sum(x[k] for x in r.values()) / len(r)
        score = sum(wt[t] * x["cost"] for t, x in r.items()) / W
        beat = sum(x["cost"] < gt[t][1] for t, x in r.items())
        print(f"\n{f}: weighted {score:.4f}, feasible {sum(x['is_feasible'] for x in r.values())}/{len(r)}, "
              f"HPWL gap {mean('hpwl_gap'):.3f}, area gap {mean('area_gap'):.3f}, "
              f"vrel {mean('violations_relative'):.3f}, {mean('runtime_seconds'):.0f} s per design")
        print(f"  beats the ground truth's own score on {beat}/{len(r)} designs")
        worst = sorted(r, key=lambda t: -(r[t]["cost"] - gt[t][1]) * wt[t])[:a.worst]
        print("  largest weighted losses: design (blocks) ours vs ground truth")
        for t in worst:
            print(f"    {t:3d} ({gt[t][0]:3d})  {r[t]['cost']:.3f} vs {gt[t][1]:.3f}")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from iccad2026_evaluate import ContestEvaluator, calculate_hpwl_b2b, calculate_hpwl_p2b, calculate_bbox_area
from rl.reward import INFERENCE_AREA_WEIGHT

p = argparse.ArgumentParser()
p.add_argument("--test-id", type=int, default=0)
args = p.parse_args()

evaluator = ContestEvaluator("../", verbose=False)
evaluator._load_dataset()
optimizer = evaluator._load_optimizer("my_optimizer.py")

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

hpwl_total = calculate_hpwl_b2b(positions, b2b_conn) + calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
bbox_area = calculate_bbox_area(positions)
weighted_area = INFERENCE_AREA_WEIGHT * bbox_area
quality = hpwl_total + weighted_area

print(f"test_id={args.test_id} blocks={block_count}")
print(f"hpwl_total       = {hpwl_total:.2f}")
print(f"bbox_area        = {bbox_area:.2f}")
print(f"AREA_WEIGHT      = {INFERENCE_AREA_WEIGHT}")
print(f"weighted_area    = {weighted_area:.2f}  ({100*weighted_area/quality:.1f}% of quality term)")
print(f"hpwl share       = {100*hpwl_total/quality:.1f}% of quality term")
print(f"combined quality = {quality:.2f}")

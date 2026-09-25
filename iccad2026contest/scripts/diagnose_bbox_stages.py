import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from iccad2026_evaluate import ContestEvaluator, calculate_bbox_area
from rl.data import from_solve_args
from rl.finetune import finetune_and_solve
from rl.networks import ActorCritic
from rl.compaction import compact
from rl.anneal import anneal_polish

p = argparse.ArgumentParser()
p.add_argument("--test-id", type=int, default=0)
args = p.parse_args()

evaluator = ContestEvaluator("../", verbose=False)
evaluator._load_dataset()

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

instance = from_solve_args(
    block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints, opt_target_pos
)
net = ActorCritic()
net.load_state_dict(torch.load("checkpoints/policy.pt", map_location="cpu"))

# finetune_and_solve already compacts internally before returning -- so
# "after_finetune" below is really "after finetune + compact". We separately
# monkeypatch-free re-derive the pre-compaction bbox by calling compact()
# again on its own output (idempotence check) and by checking env.py's
# raw rollout via the same path finetune.py uses internally.
positions_after_finetune, _ = finetune_and_solve(
    instance, net=net, time_budget=20.0, max_iterations=150,
    episodes_per_iter=6, ppo_epochs=3, verbose=False,
)
bbox_after_finetune = calculate_bbox_area(positions_after_finetune)

# Re-run compact() again on already-compacted output: if bbox doesn't
# change, compact() has reached its fixed point (expected). If it DOES
# change, compact() is non-idempotent (a bug).
recompacted = compact(positions_after_finetune, instance.constraints)
bbox_recompacted = calculate_bbox_area(recompacted)

annealed = anneal_polish(instance, positions_after_finetune, time_budget=8.0, verbose=False)
bbox_after_anneal = calculate_bbox_area(annealed)

ideal_area = float(area_target.clamp(min=0).sum().item())

print(f"test_id={args.test_id} blocks={block_count}")
print(f"ideal (zero-waste) area      = {ideal_area:.1f}")
print(f"baseline (ground truth) bbox = {baseline['area_baseline']:.1f}")
print(f"env canvas area (2.25x)      = {ideal_area*2.25:.1f}")
print(f"after finetune+compact bbox  = {bbox_after_finetune:.1f}  (ratio to ideal: {bbox_after_finetune/ideal_area:.3f})")
print(f"after re-compact bbox        = {bbox_recompacted:.1f}  (changed: {bbox_recompacted != bbox_after_finetune})")
print(f"after anneal_polish bbox     = {bbox_after_anneal:.1f}  (ratio to ideal: {bbox_after_anneal/ideal_area:.3f})")

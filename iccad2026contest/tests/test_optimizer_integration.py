"""
End-to-end proof that the RL optimizer runs through the REAL contest
evaluator (not a reimplementation) and produces feasible, scored placements.
Uses tiny time/iteration budgets so the suite stays fast -- see
finetune.py's defaults (used by the real `--evaluate my_optimizer.py` CLI
run) for the budget an actual submission would use.
"""

import torch

from iccad2026_evaluate import ContestEvaluator, compute_cost, evaluate_solution, validate_submission
from my_optimizer import MyOptimizer


def test_validate_submission_quick_passes():
    assert validate_submission("my_optimizer.py", quick=True, verbose=False)


def _tiny_optimizer():
    return MyOptimizer(verbose=False, time_budget=4.0, max_iterations=6,
                        episodes_per_iter=3, ppo_epochs=2)


def _solve_and_score(evaluator, test_id, optimizer):
    ev = evaluator
    sample = ev.dataset[test_id]
    inputs, labels = sample['input'], sample['label']
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    block_count = int((area_target != -1).sum().item())
    baseline, target_pos = ev._extract_baseline(test_id, labels, b2b_conn, p2b_conn, pins_pos, block_count)

    opt_target_pos = torch.full((block_count, 4), -1.0)
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(block_count):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_preplaced = nc > 1 and constraints[i, 1] != 0
        if is_preplaced:
            opt_target_pos[i] = torch.tensor(target_pos[i])
        elif is_fixed:
            opt_target_pos[i, 2] = target_pos[i][2]
            opt_target_pos[i, 3] = target_pos[i][3]

    positions = optimizer.solve(block_count, area_target, b2b_conn, p2b_conn, pins_pos,
                                 constraints, opt_target_pos)
    assert isinstance(positions, list) and len(positions) == block_count

    metrics = evaluate_solution(
        {'positions': positions, 'runtime': 1.0}, baseline, constraints,
        b2b_conn, p2b_conn, pins_pos, area_target, target_pos, median_runtime=1.0,
    )
    return metrics


def test_optimizer_produces_feasible_placement_on_real_validation_cases():
    ev = ContestEvaluator(data_path="../", verbose=False)
    ev._load_dataset()
    optimizer = _tiny_optimizer()

    for test_id in (0, 50):
        metrics = _solve_and_score(ev, test_id, optimizer)
        assert metrics.is_feasible, f"test_id={test_id} produced an infeasible placement"
        assert metrics.cost < 10.0

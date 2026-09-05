"""
Two reward modes, both built on the contest's own evaluate_solution (never
reimplemented) so violation accounting always matches what actually gets
scored:

    pretraining_reward -- used when a ground-truth baseline is available
        (training/validation samples): reward = -exact contest cost.

    inference_reward -- used inside solve() at contest time, where no
        baseline exists (the hidden test set doesn't expose ground truth to
        the optimizer): same exact hard/soft violation accounting, but the
        quality term is absolute wirelength+area rather than a gap against
        an unknown baseline.
"""

import math

from iccad2026_evaluate import BETA, M_PENALTY, evaluate_solution


def _target_positions_list(instance):
    return [tuple(row.tolist()) for row in instance.target_positions]


def _evaluate(instance, positions, runtime=1.0, median_runtime=1.0):
    baseline = instance.baseline_metrics
    if baseline is None:
        # Placeholder so evaluate_solution's gap computation runs (division
        # needs *a* baseline); inference_reward below ignores hpwl_gap/
        # area_gap entirely and uses the absolute hpwl_total/bbox_area
        # fields instead, so this placeholder never leaks into the reward.
        baseline = {'hpwl_baseline': 1.0, 'area_baseline': 1.0}
    return evaluate_solution(
        {'positions': positions, 'runtime': runtime},
        baseline,
        instance.constraints,
        instance.b2b_connectivity,
        instance.p2b_connectivity,
        instance.pins_pos,
        instance.area_targets,
        _target_positions_list(instance),
        median_runtime=median_runtime,
    )


def pretraining_reward(instance, positions, runtime: float = 1.0) -> float:
    assert instance.baseline_metrics is not None, (
        "pretraining_reward requires ground-truth baseline metrics; "
        "use inference_reward when none is available"
    )
    metrics = _evaluate(instance, positions, runtime=runtime)
    return -metrics.cost


def inference_reward(instance, positions, runtime: float = 1.0, area_weight: float = 0.01) -> float:
    metrics = _evaluate(instance, positions, runtime=runtime)
    if not metrics.is_feasible:
        return -M_PENALTY
    quality = metrics.hpwl_total + area_weight * metrics.bbox_area
    violation_factor = math.exp(BETA * metrics.violations_relative)
    return -(quality * violation_factor)

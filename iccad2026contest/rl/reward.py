"""
Two reward modes, both built on the contest's own evaluate_solution (never
reimplemented) so violation accounting always matches what actually gets
scored:

    pretraining_reward -- used when a ground-truth baseline is available
        (training/validation samples): reward = -uncapped contest-cost
        formula (see _uncapped_quality below -- deliberately NOT the same as
        evaluate_solution's metrics.cost).

    inference_reward -- used inside solve() at contest time, where no
        baseline exists (the hidden test set doesn't expose ground truth to
        the optimizer): same exact hard/soft violation accounting, but the
        quality term is absolute wirelength+area rather than a gap against
        an unknown baseline.
"""

import math
from typing import Optional

from iccad2026_evaluate import ALPHA, BETA, M_PENALTY, evaluate_solution

from .compaction import compact

INFERENCE_AREA_WEIGHT = 0.01  # must match inference_reward's default below


def _uncapped_quality(metrics) -> float:
    """Same feasible-branch formula as iccad2026_evaluate.compute_cost
    (quality_factor * violation_factor; the runtime term is dropped since at
    training/fine-tuning time runtime is a few hundredths of a second and
    that factor is essentially always floored to a constant 0.7 -- it adds
    no gradient signal), but WITHOUT compute_cost's min(..., M_PENALTY)
    ceiling.

    That ceiling exists so a feasible submission always outscores an
    infeasible one on the leaderboard -- it's a scoring safeguard, not a
    property a training reward should have. Using the capped cost directly
    as the PPO reward (as this module used to) means every rollout bad
    enough to hit the ceiling looks equally bad to the advantage estimator,
    regardless of how much worse one is than another -- exactly the
    regime a 14k-iteration training run got stuck in with zero improvement
    (see the training-log analysis this change was motivated by: whole
    batches averaging reward=-10.0000, i.e. every rollout saturated).
    Infeasible rollouts are similarly given a smooth, violation-count-scaled
    penalty above M_PENALTY instead of one flat value, for the same reason
    -- though rl/env.py's construction makes true infeasibility rare here.
    """
    if not metrics.is_feasible:
        hard_violations = (metrics.overlap_violations + metrics.area_violations
                            + metrics.dimension_violations)
        return M_PENALTY + hard_violations
    quality_factor = 1 + ALPHA * (max(0, metrics.hpwl_gap) + max(0, metrics.area_gap))
    violation_factor = math.exp(BETA * metrics.violations_relative)
    return quality_factor * violation_factor


def step_quality_delta(delta_wl: float, delta_area: float, use_baseline: bool,
                        baseline_metrics: Optional[dict]) -> float:
    """Causal, per-placement-step piece of pretraining_reward/inference_reward's
    quality term, given the exact wirelength/area increase rl/env.py's
    _commit() attributes to that one step (see GridPlacementEnv._step_deltas).

    Both totals only ever grow across a rollout (each b2b/p2b edge cost and
    each bbox-area increment is >= 0), so summing this return value over
    every step of an episode reproduces the un-clipped, un-floored quality
    penalty exactly -- rl/ppo.py's collect_episode adds one residual
    correction at the final step (covering the max(0, gap) floor and any
    untracked auto-placed-cluster-touch cost) to land on the exact terminal
    value, so this function itself never needs to clip or special-case.
    """
    if use_baseline:
        hpwl_baseline = max(baseline_metrics['hpwl_baseline'], 1e-6)
        area_baseline = max(baseline_metrics['area_baseline'], 1e-6)
        return -ALPHA * (delta_wl / hpwl_baseline + delta_area / area_baseline)
    return -(delta_wl + INFERENCE_AREA_WEIGHT * delta_area)


def _target_positions_list(instance):
    return [tuple(row.tolist()) for row in instance.target_positions]


def _evaluate(instance, positions, runtime=1.0, median_runtime=1.0):
    """Scores `compact(positions)`, not `positions` itself: every actual
    submission (rl/finetune.py's finetune_and_solve) always runs the raw RL
    placement through rl/compaction.py's deterministic gravity-compaction
    before returning it, so that's the geometry the contest actually scores
    -- this function's whole purpose (per the module docstring) is to
    reproduce what actually gets scored. Before this fix, training reward
    was computed on the raw, uncompacted rollout, which silently gave the
    policy zero gradient signal toward the one thing (packing tightly)
    compaction can't do for it -- see algorithm.md's area_gap diagnosis."""
    baseline = instance.baseline_metrics
    if baseline is None:
        # Placeholder so evaluate_solution's gap computation runs (division
        # needs *a* baseline); inference_reward below ignores hpwl_gap/
        # area_gap entirely and uses the absolute hpwl_total/bbox_area
        # fields instead, so this placeholder never leaks into the reward.
        baseline = {'hpwl_baseline': 1.0, 'area_baseline': 1.0}
    compacted = compact(positions, instance.constraints)
    return evaluate_solution(
        {'positions': compacted, 'runtime': runtime},
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
    return -_uncapped_quality(metrics)


def inference_reward(instance, positions, runtime: float = 1.0,
                      area_weight: float = INFERENCE_AREA_WEIGHT) -> float:
    metrics = _evaluate(instance, positions, runtime=runtime)
    if not metrics.is_feasible:
        return -M_PENALTY
    quality = metrics.hpwl_total + area_weight * metrics.bbox_area
    violation_factor = math.exp(BETA * metrics.violations_relative)
    return -(quality * violation_factor)

"""
Two reward modes, both built on the contest's own evaluate_solution (never
reimplemented) so violation accounting always matches what actually gets
scored:

    pretraining_reward -- used when a ground-truth baseline is available
        (training/validation samples): reward = -uncapped contest-cost
        formula (see _reward_and_violation_factor below -- deliberately NOT
        the same as evaluate_solution's metrics.cost).

    inference_reward -- used inside solve() at contest time, where no
        baseline exists (the hidden test set doesn't expose ground truth to
        the optimizer): same exact hard/soft violation accounting, but the
        quality term is measured against the instance's own
        `reference_metrics` (a fixed greedy layout, see rl/finetune.py) when
        present, else absolute wirelength+area.

Why the reference matters (2026-09-25): absolute wirelength put feasible
fine-tune rewards at ~-100 to -5000, far below the -M_PENALTY a failed
rollout gets -- so PPO fine-tuning was rewarded for running out of canvas.
Normalizing against a per-instance reference puts feasible rewards on the
same O(1) scale as pretraining_reward, and the infeasible penalty is scaled
by exp(BETA) (the largest possible violation_factor) so it stays below any
plausible feasible reward -- see infeasible_penalty.
"""

import math
import os
from typing import Optional

from iccad2026_evaluate import ALPHA, BETA, M_PENALTY, evaluate_solution

from .compaction import compact

INFERENCE_AREA_WEIGHT = 0.01  # must match inference_reward's default below
# Training-only multiplier on the area term of pretraining_reward (and its
# per-step pieces). Experiment 09-27: layouts spread to fill the canvas;
# does a stronger area pull make the policy pack tighter? 1.0 = contest weights.
TRAIN_AREA_WEIGHT = float(os.environ.get("FLOORSET_AREA_WEIGHT", "1.0"))
# Same for the wirelength term (10-02): big-first ordering packs tighter but
# loses HPWL; does a stronger wirelength pull win it back? 1.0 = contest weights.
TRAIN_HPWL_WEIGHT = float(os.environ.get("FLOORSET_HPWL_WEIGHT", "1.0"))
# Training reward on the slicing-packed layout (rl/slicing.py), 09-29: the
# canvas-sized bbox is fixed after ~11% of placements, so the area term
# gives the policy no lever; scored after packing, the arrangement decides
# how much white space is left. Budget per episode in seconds; 0 = off.
TRAIN_SLICE_BUDGET = float(os.environ.get("FLOORSET_SLICE_REWARD", "0"))


def _normalized_quality(hpwl: float, area: float, reference: dict) -> float:
    """quality_factor for normalized inference_reward: 1 + ALPHA*(ratios).
    Ratios, not gaps, and never clipped -- beating the reference must keep
    paying off, and the >= 1 floor keeps violation_factor meaningful."""
    return 1 + ALPHA * (hpwl / max(reference['hpwl_ref'], 1e-6)
                        + area / max(reference['area_ref'], 1e-6))


# Training-mode penalty for a rollout that ran out of room, graded by how far
# it got: FAIL_FLOOR when nearly done, M_PENALTY when stuck at the start. A
# flat -M_PENALTY on one transition swamped PPO updates at the canvas edge
# (09-26: finish rate slid 89%->59% at utilization 0.675, twice).
FAIL_FLOOR = 5.0


def infeasible_penalty(instance, use_baseline: bool, hard_violations: int = 0,
                       placed_fraction: Optional[float] = None) -> float:
    """Reward for an infeasible or unfinished rollout. Under normalized
    inference_reward a feasible reward can reach -(quality * e^BETA), so the
    flat -M_PENALTY is scaled by e^BETA to stay below it (see module docstring).
    In training (use_baseline) an unfinished rollout's `placed_fraction`
    grades it between -FAIL_FLOOR and -M_PENALTY."""
    penalty = M_PENALTY + hard_violations
    if use_baseline and placed_fraction is not None:
        f = min(max(placed_fraction, 0.0), 1.0)
        penalty = FAIL_FLOOR + (M_PENALTY - FAIL_FLOOR) * (1.0 - f) + hard_violations
    if not use_baseline and getattr(instance, 'reference_metrics', None) is not None:
        penalty *= math.exp(BETA)
    return -penalty


def step_quality_delta(delta_wl: float, delta_area: float, use_baseline: bool,
                        baseline_metrics: Optional[dict],
                        reference_metrics: Optional[dict] = None) -> float:
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
        return -ALPHA * (TRAIN_HPWL_WEIGHT * delta_wl / hpwl_baseline
                       + TRAIN_AREA_WEIGHT * delta_area / area_baseline)
    if reference_metrics is not None:
        return -ALPHA * (delta_wl / max(reference_metrics['hpwl_ref'], 1e-6)
                         + delta_area / max(reference_metrics['area_ref'], 1e-6))
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


def _reward_and_violation_factor(instance, positions, use_baseline: bool, runtime: float = 1.0,
                                  area_weight: float = INFERENCE_AREA_WEIGHT) -> tuple:
    """Shared by pretraining_reward/inference_reward, and by rl/ppo.py's
    collect_episode, which needs `violation_factor` on its own to correctly
    rescale the per-step return_to_go decomposition -- see that module's
    collect_episode docstring/comments for why: the terminal reward is
    `-(quality_factor) * (violation_factor)` (multiplicative), but
    step_quality_delta only ever computes the additive quality piece, so
    collect_episode must rescale each step's tracked delta by this same
    violation_factor before distributing the (now much smaller, bounded)
    residual, instead of dumping the whole multiplicative interaction onto
    one arbitrary transition.

    The feasible-branch formula matches iccad2026_evaluate.compute_cost
    (quality_factor * violation_factor; the runtime term is dropped since at
    training/fine-tuning time runtime is a few hundredths of a second and
    that factor is essentially always floored to a constant 0.7 -- it adds
    no gradient signal), but deliberately WITHOUT compute_cost's
    min(..., M_PENALTY) ceiling: that ceiling exists so a feasible
    submission always outscores an infeasible one on the leaderboard -- a
    scoring safeguard, not a property a training reward should have. Using
    the capped cost directly as the PPO reward (as this module used to)
    means every rollout bad enough to hit the ceiling looks equally bad to
    the advantage estimator, regardless of how much worse one is than
    another -- exactly the regime a 14k-iteration training run got stuck in
    with zero improvement (whole batches averaging reward=-10.0000, i.e.
    every rollout saturated). Infeasible rollouts get a smooth,
    violation-count-scaled penalty above M_PENALTY instead of one flat
    value, for the same reason -- though rl/env.py's construction makes
    true infeasibility rare here, and violation_factor=1.0 on that branch
    since there's no quality_factor to scale: the flat penalty is the whole
    reward."""
    metrics = None
    if use_baseline and TRAIN_SLICE_BUDGET > 0:
        from .slicing import slicing_polish
        packed = slicing_polish(instance, positions, time_budget=TRAIN_SLICE_BUDGET)
        if packed is not positions:          # packing won: score it as returned
            metrics = evaluate_solution({'positions': packed, 'runtime': runtime}, instance.baseline_metrics,
                                        instance.constraints, instance.b2b_connectivity,
                                        instance.p2b_connectivity, instance.pins_pos,
                                        instance.area_targets, _target_positions_list(instance))
    if metrics is None:
        metrics = _evaluate(instance, positions, runtime=runtime)
    if not metrics.is_feasible:
        hard_violations = (metrics.overlap_violations + metrics.area_violations
                            + metrics.dimension_violations)
        return infeasible_penalty(instance, use_baseline, hard_violations), 1.0
    violation_factor = math.exp(BETA * metrics.violations_relative)
    if use_baseline:
        quality_factor = 1 + ALPHA * (TRAIN_HPWL_WEIGHT * max(0, metrics.hpwl_gap)
                                      + TRAIN_AREA_WEIGHT * max(0, metrics.area_gap))
        return -(quality_factor * violation_factor), violation_factor
    reference = getattr(instance, 'reference_metrics', None)
    if reference is not None:
        quality = _normalized_quality(metrics.hpwl_total, metrics.bbox_area, reference)
        return -(quality * violation_factor), violation_factor
    quality = metrics.hpwl_total + area_weight * metrics.bbox_area
    return -(quality * violation_factor), violation_factor


def pretraining_reward(instance, positions, runtime: float = 1.0) -> float:
    assert instance.baseline_metrics is not None, (
        "pretraining_reward requires ground-truth baseline metrics; "
        "use inference_reward when none is available"
    )
    reward, _ = _reward_and_violation_factor(instance, positions, use_baseline=True, runtime=runtime)
    return reward


def inference_reward(instance, positions, runtime: float = 1.0,
                      area_weight: float = INFERENCE_AREA_WEIGHT) -> float:
    reward, _ = _reward_and_violation_factor(instance, positions, use_baseline=False,
                                              runtime=runtime, area_weight=area_weight)
    return reward

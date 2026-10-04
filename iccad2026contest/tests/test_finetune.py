import torch

import rl.env as env_mod
import rl.finetune as finetune_mod
from rl.data import synthetic_instance
from iccad2026_evaluate import check_overlap


def test_greedy_fallback_recovers_from_a_too_tight_starting_canvas():
    # A near-zero padding has essentially no room for any block --
    # position_mask() is expected to raise "canvas too small" on the very
    # first attempt. The fallback must retry with a wider canvas rather
    # than letting that propagate and crash solve(). synthetic_instance
    # has no pins, so the first attempt (canvas_padding=None) falls back
    # to rl.env's own CANVAS_PADDING -- that's the value to shrink here,
    # not finetune.py's own copy (only used for the retry ladder itself).
    original_env_padding = env_mod.CANVAS_PADDING
    original_finetune_padding = finetune_mod.CANVAS_PADDING
    env_mod.CANVAS_PADDING = finetune_mod.CANVAS_PADDING = 0.01
    try:
        instance = synthetic_instance(area_targets=[4.0] * 10)
        positions = finetune_mod.greedy_fallback_positions(instance, grid_dim=32)
    finally:
        env_mod.CANVAS_PADDING = original_env_padding
        finetune_mod.CANVAS_PADDING = original_finetune_padding

    assert len(positions) == 10
    assert check_overlap(positions) == 0
    assert all(w > 0 and h > 0 for (_, _, w, h) in positions)


def test_greedy_fallback_still_succeeds_at_default_padding():
    instance = synthetic_instance(area_targets=[4.0] * 10)
    positions = finetune_mod.greedy_fallback_positions(instance, grid_dim=32)
    assert len(positions) == 10
    assert check_overlap(positions) == 0


def test_reference_metrics_are_positive_and_caller_instance_is_untouched():
    instance = synthetic_instance(area_targets=[4.0] * 6,
                                   b2b_edges=[(i, i + 1, 1.0) for i in range(5)])
    ref = finetune_mod.compute_reference_metrics(instance, grid_dim=16)
    assert ref is not None and ref['hpwl_ref'] > 0 and ref['area_ref'] > 0

    positions, _ = finetune_mod.finetune_and_solve(instance, time_budget=2.0, max_iterations=2,
                                                    episodes_per_iter=2, ppo_epochs=1, seed=0)
    assert instance.reference_metrics is None
    assert check_overlap(positions) == 0

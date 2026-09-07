"""
Fuzz the env against every real validation-set instance (not just synthetic
fixtures) -- real data exercises constraint combinations (e.g. a fixed-shape
block that is also an MIB group member) synthetic tests are unlikely to hit.
"""

import pytest

from iccad2026_evaluate import ContestEvaluator, check_overlap
from rl.data import from_validation_sample
from rl.env import GridPlacementEnv


@pytest.fixture(scope="module")
def evaluator():
    ev = ContestEvaluator(data_path="../", verbose=False)
    ev._load_dataset()
    return ev


@pytest.mark.parametrize("test_id", range(100))
def test_random_rollout_respects_hard_constraints_on_validation_set(evaluator, rollout, test_id):
    inst = from_validation_sample(evaluator, test_id)
    env = GridPlacementEnv(inst, grid_dim=48)
    positions = rollout(env, seed=0, max_seed_tries=30)

    assert check_overlap(positions) == 0

    ncols = inst.constraints.shape[1]
    for i in range(inst.block_count):
        is_fixed = ncols > 0 and inst.constraints[i, 0] != 0
        is_preplaced = ncols > 1 and inst.constraints[i, 1] != 0
        x, y, w, h = positions[i]

        if is_fixed or is_preplaced:
            tx, ty, tw, th = inst.target_positions[i].tolist()
            assert w == pytest.approx(tw, abs=1e-4)
            assert h == pytest.approx(th, abs=1e-4)
            if is_preplaced:
                assert x == pytest.approx(tx, abs=1e-4)
                assert y == pytest.approx(ty, abs=1e-4)
        else:
            area = float(inst.area_targets[i])
            if area > 0:
                assert w * h == pytest.approx(area, rel=1e-4)

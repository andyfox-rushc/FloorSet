"""rl/slicing.py: slicing re-pack keeps hard constraints and never makes things worse."""
import numpy as np
import pytest

from iccad2026_evaluate import ContestEvaluator
from rl.data import from_validation_sample
from rl.env import GridPlacementEnv
from rl.repack import _problem, contest_relative_cost
from rl.slicing import _build, slicing_polish
from conftest import run_random_rollout


@pytest.fixture(scope="module")
def layouts():
    ev = ContestEvaluator(data_path="../", verbose=False)
    ev._load_dataset()
    out = {}
    for tid in (0, 49, 98):
        inst = from_validation_sample(ev, tid)
        out[tid] = (inst, run_random_rollout(GridPlacementEnv(inst, grid_dim=48), seed=0, max_seed_tries=30))
    return out


def test_leaves_tile_the_root_and_obstacles_keep_their_spot(layouts):
    for inst, pos in layouts.values():
        P = _problem(inst, pos)
        p = P["pos"]
        cx, cy = p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2
        W, H = 1.1 * (p[:, 0] + p[:, 2]).max(), 1.1 * (p[:, 1] + p[:, 3]).max()
        rx, ry, rw, rh = _build(P, cx, cy, W, H, "area")
        pre = P["preplaced"]
        assert np.allclose(np.c_[rx, ry, rw, rh][pre], p[pre])
        assert (rw > 0).all() and (rh > 0).all()
        # Movable leaves plus preplaced rectangles never exceed the root.
        assert rx.min() >= -1e-6 and ry.min() >= -1e-6
        assert (rx + rw).max() <= W + 1e-6 and (ry + rh).max() <= H + 1e-6


def test_slicing_polish_is_feasible_and_not_worse(layouts):
    for inst, pos in layouts.values():
        out = slicing_polish(inst, pos, time_budget=1.0)
        cost_in, _ = contest_relative_cost(inst, pos, pos)
        cost_out, feasible = contest_relative_cost(inst, out, pos)
        assert feasible and cost_out <= cost_in + 1e-9

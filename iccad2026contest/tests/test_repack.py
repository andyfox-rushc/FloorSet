"""rl/repack.py: re-packing keeps hard constraints and never makes things worse."""
import numpy as np
import pytest

from iccad2026_evaluate import ContestEvaluator, check_overlap
from rl.data import from_validation_sample
from rl.env import GridPlacementEnv
from rl.repack import _legalize, _problem, contest_relative_cost, repack_polish
from conftest import run_random_rollout


@pytest.fixture(scope="module")
def evaluator():
    ev = ContestEvaluator(data_path="../", verbose=False)
    ev._load_dataset()
    return ev


@pytest.fixture(scope="module")
def layouts(evaluator):
    out = {}
    for tid in (0, 49):
        inst = from_validation_sample(evaluator, tid)
        out[tid] = (inst, run_random_rollout(GridPlacementEnv(inst, grid_dim=48), seed=0, max_seed_tries=30))
    return out


def test_legalize_is_overlap_free_and_keeps_preplaced(layouts):
    rng = np.random.default_rng(0)
    for inst, pos in layouts.values():
        P = _problem(inst, pos)
        p = P["pos"]
        for trial in range(5):   # also from scrambled (overlapping) placements
            x = p[:, 0] + (0 if trial == 0 else rng.normal(0, 5, len(p)) * ~P["preplaced"])
            y = p[:, 1] + (0 if trial == 0 else rng.normal(0, 5, len(p)) * ~P["preplaced"])
            nx, ny, over = _legalize(x, y, p[:, 2], p[:, 3], P)
            if over > 1e-6:
                continue   # would move a preplaced block: the chain rejects these
            new = [(nx[i], ny[i], p[i, 2], p[i, 3]) for i in range(len(p))]
            assert check_overlap(new) == 0
            pre = P["preplaced"]
            assert np.allclose(nx[pre], p[pre, 0]) and np.allclose(ny[pre], p[pre, 1])


def test_legalize_of_legal_layout_never_grows_bbox(layouts):
    for inst, pos in layouts.values():
        P = _problem(inst, pos)
        p = P["pos"]
        nx, ny, over = _legalize(p[:, 0], p[:, 1], p[:, 2], p[:, 3], P)
        assert over <= 1e-6
        assert (nx + p[:, 2]).max() <= (p[:, 0] + p[:, 2]).max() + 1e-6
        assert (ny + p[:, 3]).max() <= (p[:, 1] + p[:, 3]).max() + 1e-6


def test_repack_polish_is_feasible_and_not_worse(layouts):
    for inst, pos in layouts.values():
        out = repack_polish(inst, pos, time_budget=2.0, seed=0)
        cost_in, _ = contest_relative_cost(inst, pos, pos)
        cost_out, feasible = contest_relative_cost(inst, out, pos)
        assert feasible and cost_out <= cost_in + 1e-9

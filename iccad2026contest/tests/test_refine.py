import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

import prime_evaluate as E
from rl import partition as PT
from rl.refine import Refiner, layout_ok, strip_collinear


@pytest.fixture(scope="module")
def designs():
    return E.load_designs(ids=[1, 3])


def _problem(d):
    return PT.Problem(d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"], d["constraints"],
                      d["target_polygons"])


def _gt_seeds(d):
    return np.array([Polygon(v).centroid.coords[0] for v in d["gt"]])


def test_strip_collinear_drops_points_on_straight_runs():
    ring = [(0, 0), (2, 0), (4, 0), (4, 3), (4, 3), (0, 3)]
    assert strip_collinear(ring) == [[0, 0], [4, 0], [4, 3], [0, 3]]


def test_strip_collinear_keeps_real_corners():
    ring = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 3), (0, 3)]
    assert len(strip_collinear(ring)) == 6


def test_refined_layout_stays_valid_and_tiles(designs):
    for d in designs:
        P = _problem(d)
        lay = PT.solve(P, _gt_seeds(d), time_budget=3.0, seed=0)
        assert lay is not None
        before = E.evaluate(d, lay)
        R = Refiner(P, lay)
        R.run(time_budget=3.0, seed=0)
        out = R.layout()
        assert layout_ok(P, out)
        r = E.evaluate(d, out)
        assert r["feasible"], r["hard"]
        polys = [Polygon(v) for v in out]
        assert abs(unary_union(polys).area - P.W * P.H) < 1e-4 * P.W * P.H      # still no white space
        assert r["quality_cost"] <= before["quality_cost"] + 0.02                # proxy, not the exact cost


def test_layout_ok_rejects_a_broken_fixed_shape(designs):
    d = designs[0]
    P = _problem(d)
    lay = [list(map(list, v)) for v in d["gt"]]
    fixed = [i for i in range(P.n) if P.fixed[i] and not P.pre[i]]
    if not fixed:
        pytest.skip("no fixed block")
    i = fixed[0]
    x0, y0, x1, y1 = Polygon(lay[i]).bounds
    lay[i] = [[x0, y0], [x1, y0], [x1, y0 + (y1 - y0) / 2], [x0, y0 + (y1 - y0) / 2]]   # half height
    assert not layout_ok(P, lay)


def test_corridor_puts_a_far_block_on_its_edge():
    # design 77: soft block 42 must touch the left edge, 43 units away behind a band of blocks
    import json
    from pathlib import Path
    f = Path(__file__).resolve().parent.parent / "prime_results" / "final_1023.json"
    if not f.exists():
        pytest.skip("needs prime_results/final_1023.json")
    lay = next(r for r in json.loads(f.read_text()) if r["id"] == 77)["layout"]
    d = E.load_designs(ids=[77])[0]
    P = _problem(d)
    R = Refiner(P, lay)
    R.hp = R.hp_ref = P.hpwl(R.c[:, 0], R.c[:, 1])
    R.viol = R.soft_viol(range(P.n))
    assert R._edge_gap(42) > 0
    import time
    assert R.corridor(42, time.time() + 60)
    assert R._edge_gap(42) == 0 and R.viol == 0
    assert layout_ok(P, R.layout())

import numpy as np
import pytest
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

import prime_evaluate as E
from rl import partition as PT
from rl import polygon as G


@pytest.fixture(scope="module")
def designs():
    return E.load_designs(ids=[1, 3])


def _gt_seeds(d):
    return np.array([Polygon(v).centroid.coords[0] for v in d["gt"]])


def test_split_is_exact():
    R = PT.decompose_geom(box(0, 0, 10, 6).difference(box(3, 2, 5, 4)))
    for axis in (0, 1):
        for a in (5.0, 17.3, 40.0):
            lo, hi = PT.split(R, axis, a)
            assert abs(PT.area(lo) - a) < 1e-9 and abs(PT.area(lo) + PT.area(hi) - PT.area(R)) < 1e-9


def test_partition_tiles_frame_exactly(designs):
    for d in designs:
        args = (d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"], d["constraints"],
                d["target_polygons"])
        P = PT.Problem(*args)
        layout = PT.solve(P, _gt_seeds(d), time_budget=3.0, seed=0)
        assert layout is not None
        polys = [Polygon(v) for v in layout]
        assert abs(unary_union(polys).area - P.W * P.H) < 1e-6 * P.W * P.H      # no white space
        assert abs(sum(p.area for p in polys) - P.W * P.H) < 1e-6 * P.W * P.H    # no overlap
        r = E.evaluate(d, layout)
        assert r["feasible"], r["hard"]
        assert abs(r["area_gap"]) < 1e-6
        for i, p in enumerate(polys):
            if not P.rigid[i]:
                assert len(p.simplify(0).exterior.coords) - 1 <= PT.MAX_CORNERS
            elif P.fixed[i]:
                assert G.same_shape(p, Polygon(d["target_polygons"][i]))


def test_rectangular_pocket_takes_a_matching_mib_block():
    # design 75: a 5 x 7 pocket walled in by preplaced blocks; three MIB blocks of area 35
    import prime_partition as PP
    d = E.load_designs(ids=[75])[0]
    args = (d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"], d["constraints"],
            d["target_polygons"])
    P = PT.Problem(*args)
    part = PT.Partitioner(P, _gt_seeds(d))
    assert PT.components(part.region) == 1
    b = next(b for b in part.pocket_rigid if P.mib_soft[b])
    group = [int(m) for m in np.where(P.mib == P.mib[b])[0] if int(m) in part.options]
    assert all(part.options[m] == part.options[group[0]] for m in group)

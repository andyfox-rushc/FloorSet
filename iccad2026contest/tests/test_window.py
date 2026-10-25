import json
import time
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

import prime_evaluate as E
from rl import partition as PT
from rl import window as WN
from rl.refine import Refiner, layout_ok

RES = Path(__file__).resolve().parent.parent / "prime_results"


def _refiner(d, lay):
    P = PT.Problem(d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"], d["constraints"],
                   d["target_polygons"])
    R = Refiner(P, lay)
    R.hp = R.hp_ref = P.hpwl(R.c[:, 0], R.c[:, 1])
    R.viol = R.soft_viol(range(P.n))
    return P, R


def test_lns_keeps_the_layout_valid_and_never_worse():
    d = E.load_designs(ids=[1])[0]
    P0 = PT.Problem(d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"], d["constraints"],
                    d["target_polygons"])
    lay = PT.solve(P0, np.array([Polygon(v).centroid.coords[0] for v in d["gt"]]), time_budget=3.0, seed=0)
    P, R = _refiner(d, lay)
    s0 = R.score(R.hp, R.viol)
    st = WN.lns(R, time.time() + 10, anneal_s=1.0, seed=0)
    assert st["score"] <= s0 + 1e-12
    out = R.layout()
    assert layout_ok(P, out)
    assert E.evaluate(d, out)["feasible"]
    assert abs(unary_union([Polygon(v) for v in out]).area - P.W * P.H) < 1e-4 * P.W * P.H


def test_cluster_shift_joins_a_rigid_pair_to_its_preplaced_mate():
    # design 8: MIB pair 4+5 (5 pinned right) sits 22 below its preplaced cluster mate 12
    f = RES / "partition_full_refine_tree_merged.json"
    if not f.exists():
        pytest.skip("needs prime_results/partition_full_refine_tree_merged.json")
    lay = next(r for r in json.loads(f.read_text()) if r["id"] == 8)["layout"]
    d = E.load_designs(ids=[8])[0]
    P, R = _refiner(d, lay)
    g = next(g for g in P.groups if 12 in g)
    assert len(R._components([int(x) for x in g])) == 2
    best = WN.cluster_shift(R, g, time.time() + 60, seeds=(0,))
    assert best is not None and best[4] == 0
    _, R.rects, R.c, R.hp, R.viol = best
    out = R.layout()
    assert layout_ok(P, out)
    r = E.evaluate(d, out)
    assert r["feasible"] and r["soft"]["cluster"] == 0 and r["soft"]["boundary"] == 0

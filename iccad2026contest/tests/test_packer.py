"""rl/packer.py: the sequence pair reproduces a legal layout, packing never
overlaps or moves preplaced blocks, and compact_pack never makes things worse."""
import numpy as np
import pytest

from iccad2026_evaluate import ContestEvaluator, check_overlap
from rl.data import from_validation_sample
from rl.env import GridPlacementEnv
from rl.packer import _Batch, compact_pack, reattach_clusters, relax_hpwl, sequence_pair
from rl.repack import _problem, contest_relative_cost
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


def _pack(P, gp, gm, w, h, floors):
    B = _Batch(P, 1.0, 1.0, floors)
    return B.pack(gm[None], np.argsort(gp)[None], np.argsort(gm)[None], w[None], h[None])


def test_sequence_pair_packs_no_bigger_and_keeps_preplaced(layouts):
    for inst, pos in layouts.values():
        P = _problem(inst, pos)
        p = P["pos"]
        x, y, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        gp, gm = sequence_pair(x, y, w, h)
        assert sorted(gp) == sorted(gm) == list(range(P["n"]))
        floors = (min(0.0, x.min()), min(0.0, y.min()))
        px, py, over = _pack(P, gp, gm, w, h, floors)
        px, py = px[0], py[0]
        assert over[0] <= 1e-6
        area = lambda X, Y: ((X + w).max() - X.min()) * ((Y + h).max() - Y.min())
        assert area(px, py) <= area(x, y) * (1 + 1e-9)
        pre = P["preplaced"]
        assert np.allclose(px[pre], x[pre]) and np.allclose(py[pre], y[pre])
        assert check_overlap([(px[i], py[i], w[i], h[i]) for i in range(P["n"])]) == 0


def test_random_sequence_pairs_never_overlap(layouts):
    rng = np.random.default_rng(0)
    for inst, pos in layouts.values():
        P = _problem(inst, pos)
        P = dict(P, preplaced=np.zeros_like(P["preplaced"]))   # no fixed spots: always feasible
        w, h = P["pos"][:, 2], P["pos"][:, 3]
        for _ in range(5):
            gp, gm = rng.permutation(P["n"]), rng.permutation(P["n"])
            px, py, _ = _pack(P, gp, gm, w, h, (0.0, 0.0))
            assert check_overlap([(px[0, i], py[0, i], w[i], h[i]) for i in range(P["n"])]) == 0


def test_compact_pack_is_feasible_and_not_worse(layouts):
    for inst, pos in layouts.values():
        out = compact_pack(inst, pos, time_budget=0.3, seed=0)
        old, _ = contest_relative_cost(inst, pos, pos)
        new, ok = contest_relative_cost(inst, out, pos)
        assert ok and new <= old + 1e-12


def test_relax_hpwl_keeps_bbox_and_is_not_worse(layouts):
    for inst, pos in layouts.values():
        start = compact_pack(inst, pos, time_budget=0.3, seed=0)
        out = relax_hpwl(inst, start)
        old, _ = contest_relative_cost(inst, start, start)
        new, ok = contest_relative_cost(inst, out, start)
        assert ok and new <= old + 1e-12
        assert check_overlap(out) == 0
        box = lambda q: (min(a[0] for a in q), min(a[1] for a in q),
                         max(a[0] + a[2] for a in q), max(a[1] + a[3] for a in q))
        assert np.allclose(box(out), box(start))


def test_squeeze_is_feasible_and_not_worse(layouts):
    from rl.packer import squeeze
    for inst, pos in layouts.values():
        start = compact_pack(inst, pos, time_budget=0.3, seed=0)
        out = squeeze(inst, start)
        old, _ = contest_relative_cost(inst, start, start)
        new, ok = contest_relative_cost(inst, out, start)
        assert ok and new <= old + 1e-12
        assert check_overlap(out) == 0


def test_origin_repair_packs_inside_the_box(layouts):
    from rl.packer import _origin_repair
    rng = np.random.default_rng(0)
    fixed = 0
    for inst, pos in layouts.values():
        P = _problem(inst, pos)
        w, h = P["pos"][:, 2].copy(), P["pos"][:, 3].copy()
        Bt = _Batch(P, 1.0, 1.0, (0.0, 0.0))
        for _ in range(3):
            gp, gm = rng.permutation(P["n"]), rng.permutation(P["n"])
            out = _origin_repair(Bt, gp, gm, w, h)
            if out is None:
                continue
            fixed += 1
            px, py, over = _pack(P, out[0], out[1], w, h, (0.0, 0.0))
            assert over[0] <= 1e-6 and px.min() >= -1e-6 and py.min() >= -1e-6
            assert check_overlap([(px[0, i], py[0, i], w[i], h[i]) for i in range(P["n"])]) == 0
    assert fixed >= 1      # random pairs are far harder than real layouts (81/81 official 10-07 fixed)


def test_reattach_clusters_is_feasible_and_not_worse(layouts):
    for inst, pos in layouts.values():
        out = reattach_clusters(inst, pos)
        assert check_overlap(out) == 0
        new, ok = contest_relative_cost(inst, out, pos)
        assert ok and new <= contest_relative_cost(inst, pos, pos)[0] + 1e-12


def test_reattach_clusters_joins_a_split_cluster():
    """Design 5 of the 10/12 official run: a corner-pinned member sits 0.27 short of
    its preplaced clustermate; stretching it in place closes the gap."""
    import json
    from pathlib import Path
    from iccad2026_evaluate import evaluate_solution
    from rl.reward import _target_positions_list
    f = Path(__file__).parent.parent / "official_eval_1012_cap16.json"
    if not f.exists():
        pytest.skip("reference run not present")
    ev = ContestEvaluator(data_path="../", verbose=False)
    ev._load_dataset()
    inst = from_validation_sample(ev, 5)
    pos = [tuple(p) for p in next(r for r in json.load(open(f))["test_results"] if r["test_id"] == 5)["positions"]]

    def score(q):
        return evaluate_solution({"positions": q, "runtime": 1.0}, inst.baseline_metrics, inst.constraints,
                                 inst.b2b_connectivity, inst.p2b_connectivity, inst.pins_pos, inst.area_targets,
                                 _target_positions_list(inst), median_runtime=1.0)
    before, after = score(pos), score(reattach_clusters(inst, pos))
    assert before.grouping_violations == 1 and after.grouping_violations == 0
    assert after.is_feasible and after.cost < before.cost

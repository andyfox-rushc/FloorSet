#!/usr/bin/env python3
"""
FloorSet-Prime evaluator: the ICCAD 2026 contest cost applied to rectilinear
polygon layouts.

FloorSet-Prime ships constraint checks (../utils.py) but no overall score and
no solve-time interface, so this file defines both, mirroring the Lite
contest (iccad2026_evaluate.py) wherever Prime's own rules allow:

  Interface   solve(block_count, area_targets, b2b_connectivity,
                    p2b_connectivity, pins_pos, constraints, target_polygons)
              -> list of block_count polygons, each a list of (x, y) vertices.
              target_polygons[i] is None for soft blocks, the shape at the
              origin for fixed blocks, and the placed polygon for preplaced
              ones (Lite passes the same information as target_positions).

  Cost        (1 + 0.5 (HPWL_gap + Area_gap)) x exp(2 V_rel) x runtime term,
              10 if infeasible -- iccad2026_evaluate.compute_cost, unchanged.
              HPWL is Prime's: weighted Manhattan distance between polygon
              centroids (and pins). Area is Prime's: W x H with W, H the
              largest x, y, i.e. the frame [0, W] x [0, H].

  Hard        right polygon count; valid, rectilinear, no coordinate < 0;
              no overlap; soft area within 1% of target and at most 16
              corners (FLOORSET_MAX_CORNERS); fixed shape equal up
              to translation and rotation; preplaced polygon exactly in place.

  Soft        boundary: touches its frame edge(s) (Prime's check); cluster:
              union pieces - 1 per group; MIB: distinct shapes - 1 per group
              (rotation allowed). Normalized by N_soft as in the contest.

Differences from Prime's utils.py checks: shape equality ignores the starting
vertex (Prime's flags 343 ground-truth MIB pairs that are identical), and
overlap and the area tolerance are checked at all (Prime checks neither).

USAGE:
  python prime_evaluate.py --gt                         # score the ground truth
  python prime_evaluate.py --evaluate prime_baseline.py [--ids 0 5] [--output r.json]
"""

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from iccad2026_evaluate import AREA_TOLERANCE, M_PENALTY, compute_cost  # noqa: E402
from rl import polygon as G  # noqa: E402

Vertices = List[Tuple[float, float]]
EPS = 1e-6
# Corner cap for soft blocks (user choice 10-20: 16 allows jigsaw-like shapes)
MAX_CORNERS = int(os.environ.get("FLOORSET_MAX_CORNERS", "16"))


class PrimeOptimizer:
    """Base class for FloorSet-Prime optimizers."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def solve(self, block_count: int, area_targets: torch.Tensor, b2b_connectivity: torch.Tensor,
              p2b_connectivity: torch.Tensor, pins_pos: torch.Tensor, constraints: torch.Tensor,
              target_polygons: Sequence[Optional[Vertices]]) -> List[Vertices]:
        raise NotImplementedError


# =============================================================================
# DATA
# =============================================================================
def load_designs(root: Optional[str] = None, ids: Optional[Sequence[int]] = None) -> List[Dict]:
    """The 100 Prime test designs (downloaded on first use into the repo root),
    in test-loader order. Each: inputs, target_polygons for solve(), and the
    ground-truth polygons and metrics for scoring."""
    from prime_dataset_test import FloorplanDatasetPrimeTest
    root = root or str(HERE.parent)
    cwd = os.getcwd()
    os.chdir(root)        # the dataset class resolves its paths from the cwd
    try:
        ds = FloorplanDatasetPrimeTest("./")
        out = []
        for i in (ids if ids is not None else range(len(ds))):
            s = ds[i]
            area, b2b, p2b, pins, cons = s["input"]
            polys, metrics = s["label"]
            gt = [G.clean_vertices(p.tolist()) for p in polys]
            targets = []
            for j, v in enumerate(gt):
                if cons[j, 1]:
                    targets.append(v)
                elif cons[j, 0]:
                    x0, y0 = min(p[0] for p in v), min(p[1] for p in v)
                    targets.append([(x - x0, y - y0) for x, y in v])
                else:
                    targets.append(None)
            out.append(dict(id=i, block_count=len(gt), area_targets=area.float(), b2b=b2b.float(),
                            p2b=p2b.float(), pins_pos=pins.float(), constraints=cons.float(),
                            target_polygons=targets, gt=gt, metrics=metrics.float()))
        return out
    finally:
        os.chdir(cwd)


# =============================================================================
# SCORING
# =============================================================================
def _rectilinear(poly: Polygon) -> bool:
    c = poly.exterior.coords
    return all(abs(a[0] - b[0]) < EPS or abs(a[1] - b[1]) < EPS for a, b in zip(c, c[1:]))


def hpwl(centroids: torch.Tensor, b2b: torch.Tensor, p2b: torch.Tensor, pins: torch.Tensor) -> Tuple[float, float]:
    """Prime's cost.py wirelength: weighted Manhattan distance between centroids."""
    b = 0.0
    if len(b2b):
        i, j, w = b2b[:, 0].long(), b2b[:, 1].long(), b2b[:, 2]
        b = float(((centroids[i] - centroids[j]).abs().sum(1) * w).sum())
    p = 0.0
    if len(p2b):
        k, j, w = p2b[:, 0].long(), p2b[:, 1].long(), p2b[:, 2]
        p = float(((centroids[j] - pins[k]).abs().sum(1) * w).sum())
    return b, p


def evaluate(design: Dict, layout: Sequence[Vertices], runtime: float = 0.0,
             median_runtime: Optional[float] = None) -> Dict:
    """Score one layout. median_runtime=None leaves the runtime term at 1."""
    n = design["block_count"]
    cons = design["constraints"]
    res = dict(id=design["id"], block_count=n, feasible=False, cost=M_PENALTY, runtime=runtime, hard={})
    hard = res["hard"]
    if layout is None or len(layout) != n:
        hard["count"] = 1
        return res
    polys = []
    bad = 0
    for v in layout:
        try:
            g = Polygon(G.clean_vertices(v))
            ok = g.is_valid and g.area > EPS and _rectilinear(g) and min(g.bounds[:2]) >= -EPS
        except Exception:
            g, ok = Polygon(), False
        bad += not ok
        polys.append(g)
    hard["invalid_polygon"] = bad
    if bad:
        return res

    tree = STRtree(polys)
    over = 0
    for i, g in enumerate(polys):
        for j in tree.query(g):
            if j > i and g.intersection(polys[j]).area > 1e-6:
                over += 1
    hard["overlap"] = over

    area = design["area_targets"]
    targets = design["target_polygons"]
    hard["area"] = sum(1 for i in range(n) if not cons[i, 0] and not cons[i, 1]
                       and abs(polys[i].area - float(area[i])) > AREA_TOLERANCE * float(area[i]))
    hard["fixed_shape"] = sum(1 for i in range(n) if cons[i, 0] and not cons[i, 1]
                              and not G.same_shape(polys[i], Polygon(targets[i])))
    hard["corners"] = sum(1 for i in range(n) if not cons[i, 0] and not cons[i, 1]
                          and len(polys[i].simplify(0).exterior.coords) - 1 > MAX_CORNERS)
    hard["preplaced"] = sum(1 for i in range(n) if cons[i, 1]
                            and polys[i].symmetric_difference(Polygon(targets[i])).area > 1e-3)
    feasible = not any(hard.values())

    W = max(g.bounds[2] for g in polys)
    H = max(g.bounds[3] for g in polys)
    cent = torch.tensor([g.centroid.coords[0] for g in polys], dtype=torch.float32)
    b, p = hpwl(cent, design["b2b"], design["p2b"], design["pins_pos"])
    m = design["metrics"]
    base_hpwl = float(m[-2] + m[-1])
    base_area = float(m[0])

    # soft constraints and N_soft (contest normalization)
    bound = cons[:, 4].long()
    frame = {1: LineString([(0, 0), (0, H)]), 2: LineString([(W, 0), (W, H)]),
             4: LineString([(0, H), (W, H)]), 8: LineString([(0, 0), (W, 0)])}
    v_bound = sum(1 for i in range(n) if bound[i]
                  and not all(polys[i].intersects(frame[bit]) for bit in (1, 2, 4, 8) if bound[i] & bit))
    n_soft = int((bound != 0).sum())
    v_clust = v_mib = 0
    for col in (3, 2):
        ids_ = cons[:, col].long()
        for gid in range(1, int(ids_.max()) + 1 if len(ids_) else 1):
            members = [i for i in range(n) if ids_[i] == gid]
            if len(members) < 2:
                continue
            n_soft += len(members) - 1
            if col == 3:
                u = unary_union([polys[i] for i in members])
                v_clust += (len(u.geoms) - 1) if u.geom_type == "MultiPolygon" else 0
            else:
                v_mib += len({G.canonical_key(polys[i]) for i in members}) - 1
    v_rel = (v_bound + v_clust + v_mib) / max(n_soft, 1)

    hpwl_gap = (b + p - base_hpwl) / max(base_hpwl, 1e-6)
    area_gap = (W * H - base_area) / max(base_area, 1e-6)
    rf = runtime / max(median_runtime, 0.01) if median_runtime else 1.0
    res.update(feasible=feasible, hpwl=b + p, hpwl_baseline=base_hpwl, area=W * H, area_baseline=base_area,
               hpwl_gap=hpwl_gap, area_gap=area_gap,
               soft=dict(boundary=v_bound, cluster=v_clust, mib=v_mib, n_soft=n_soft), v_rel=v_rel,
               cost=compute_cost(hpwl_gap, area_gap, v_rel, rf, feasible),
               quality_cost=compute_cost(hpwl_gap, area_gap, v_rel, 1.0, feasible))
    return res


def relative_cost(solve_args, layout, reference) -> Tuple[float, bool]:
    """(cost, feasible) of `layout` on the contest formula with `reference`'s
    HPWL and frame area as the baseline: comparable across layouts at solve
    time, when there is no ground truth."""
    n, area, b2b, p2b, pins, cons, targets = solve_args
    ref_polys = [Polygon(G.clean_vertices(v)) for v in reference]
    cent = torch.tensor([g.centroid.coords[0] for g in ref_polys], dtype=torch.float32)
    rb, rp = hpwl(cent, b2b, p2b, pins)
    W = max(g.bounds[2] for g in ref_polys)
    H = max(g.bounds[3] for g in ref_polys)
    design = dict(id=-1, block_count=n, area_targets=area, b2b=b2b, p2b=p2b, pins_pos=pins,
                  constraints=cons, target_polygons=targets,
                  metrics=torch.tensor([W * H, 0, 0, 0, 0, 0, rb, rp], dtype=torch.float32))
    r = evaluate(design, layout)
    if not r["feasible"]:
        return M_PENALTY, False
    # unclamped gaps (compute_cost floors them at 0, which would always favour the reference);
    # 10-23: was 1 + 0.5 * (ratio + ratio), a base near 2 that doubled the weight of soft
    # violations against wirelength and area (design 77 picked a layout 0.074 worse)
    q = 1 + 0.5 * ((r["hpwl"] / max(rb + rp, 1e-9) - 1) + (r["area"] / max(W * H, 1e-9) - 1))
    return q * math.exp(2 * r["v_rel"]), True


def total_score(results: Sequence[Dict]) -> float:
    """Contest weighting: exp((n - n_max) / 12), so big designs dominate."""
    n_max = max(r["block_count"] for r in results)
    w = [math.exp((r["block_count"] - n_max) / 12) for r in results]
    return sum(r["cost"] * x for r, x in zip(results, w)) / sum(w)


# =============================================================================
# CLI
# =============================================================================
def _load_optimizer(path: str) -> PrimeOptimizer:
    spec = importlib.util.spec_from_file_location("prime_user_optimizer", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prime_user_optimizer"] = mod
    spec.loader.exec_module(mod)
    for v in vars(mod).values():
        if isinstance(v, type) and v.__name__ != "PrimeOptimizer" and any(
                b.__name__ == "PrimeOptimizer" for b in v.__mro__[1:]):
            return v()
    raise SystemExit(f"{path}: no PrimeOptimizer subclass found")


def _summary(results: Sequence[Dict], label: str) -> None:
    feas = sum(r["feasible"] for r in results)
    ok = [r for r in results if "hpwl_gap" in r]
    mean = lambda k: sum(r[k] for r in ok) / max(len(ok), 1)
    q = total_score([dict(r, cost=r.get("quality_cost", r["cost"])) for r in results])
    print(f"{label}: weighted score {total_score(results):.4f} (without runtime term {q:.4f}) | "
          f"feasible {feas}/{len(results)} | "
          f"mean HPWL gap {mean('hpwl_gap'):.3f}, area gap {mean('area_gap'):.3f}, V_rel {mean('v_rel'):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gt", action="store_true", help="score the ground-truth layouts")
    ap.add_argument("--evaluate", help="optimizer file with a PrimeOptimizer subclass")
    ap.add_argument("--ids", type=int, nargs="*", help="design indices (default all 100)")
    ap.add_argument("--output", help="write per-design results (and layouts) as JSON")
    a = ap.parse_args()
    designs = load_designs(ids=a.ids)
    if a.gt:
        _summary([evaluate(d, d["gt"]) for d in designs], "ground truth")
    if not a.evaluate:
        return
    opt = _load_optimizer(a.evaluate)
    raw = []
    for d in designs:
        t0 = time.time()
        try:
            layout = opt.solve(d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"],
                               d["constraints"], d["target_polygons"])
        except Exception as e:          # a crash scores as infeasible, like the contest
            print(f"  design {d['id']}: solve() raised {e!r}")
            layout = None
        raw.append((d, layout, time.time() - t0))
        if a.output:      # keep finished layouts if the run is stopped early
            Path(a.output + ".partial").write_text(json.dumps(
                [dict(id=x["id"], runtime=t, layout=[list(map(list, G.clean_vertices(v))) for v in l] if l else None)
                 for x, l, t in raw]))
        r = evaluate(d, layout)          # progress line; the runtime term needs the median, so scored below
        print(f"  solved design {d['id']:3d} n={d['block_count']:3d} in {raw[-1][2]:.1f}s: "
              f"quality cost {r.get('quality_cost', r['cost']):.4f} feasible={r['feasible']}", flush=True)
    med = statistics.median(t for _, _, t in raw)
    results = []
    for d, layout, t in raw:
        r = evaluate(d, layout, t, med)
        print(f"  design {d['id']:3d} n={d['block_count']:3d}: cost {r['cost']:.4f} "
              f"(quality {r.get('quality_cost', r['cost']):.4f}) "
              f"feasible={r['feasible']} {t:.1f}s" + ("" if r["feasible"] else f" {r['hard']}"))
        r["layout"] = [list(map(list, G.clean_vertices(v))) for v in layout] if layout else None
        results.append(r)
    _summary(results, Path(a.evaluate).name)
    if a.output:
        Path(a.output).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

"""
FloorSet-Prime <-> Lite adapter: turns a Prime design (rectilinear polygons)
into the rectangle problem the Lite pipeline solves, and its rectangle answer
back into polygons.

  soft       unchanged (the pipeline gives it a rectangle at exact area)
  fixed      its target shape's bounding box, as a fixed w x h block; the
             polygon is drawn back inside that box, so no new overlap is
             possible (its notch stays empty -- the polygon-aware packer is
             what fills it)
  preplaced  split into its rectangle pieces, each its own preplaced block
             (pieces are disjoint, so the notch of a preplaced L is free for
             other blocks). Piece 0 keeps the original index, the others are
             appended after the last block. Nets to the polygon become pin
             nets to a new pin at its centroid (exact: it never moves), and
             nets between preplaced polygons are dropped (constant).

Boundary bits go to the pieces that lie on the polygon's own edge in that
direction; cluster ids go to every piece; MIB ids stay on piece 0 only.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from shapely.geometry import Polygon

from . import polygon as G

Vertices = List[Tuple[float, float]]


@dataclass
class LiteProblem:
    block_count: int
    area_targets: torch.Tensor
    b2b: torch.Tensor
    p2b: torch.Tensor
    pins_pos: torch.Tensor
    constraints: torch.Tensor
    target_positions: torch.Tensor
    n_prime: int                       # blocks in the Prime design (the first n_prime here)
    shapes: List[Optional[G.Shape]]    # Prime fixed blocks' target shapes (orientation 0)
    preplaced: List[Optional[Vertices]]


def to_lite(block_count: int, area_targets, b2b, p2b, pins_pos, constraints,
            target_polygons: Sequence[Optional[Vertices]]) -> LiteProblem:
    n = block_count
    cons = constraints[:n].clone().float()
    area = area_targets[:n].clone().float()
    tp = torch.full((n, 4), -1.0)
    shapes: List[Optional[G.Shape]] = [None] * n
    pre: List[Optional[Vertices]] = [None] * n
    extra_rows, extra_area, extra_tp = [], [], []
    pins = [tuple(map(float, p)) for p in pins_pos.tolist()]
    centroid_pin = {}
    for i in range(n):
        if cons[i, 1]:
            v = G.clean_vertices(target_polygons[i])
            pre[i] = v
            poly = Polygon(v)
            x0, y0, x1, y1 = poly.bounds
            pieces = G.decompose(poly)
            code = int(cons[i, 4])
            for k, (px, py, pw, ph) in enumerate(pieces):
                bits = 0
                bits |= 1 if code & 1 and abs(px - x0) < 1e-6 else 0
                bits |= 2 if code & 2 and abs(px + pw - x1) < 1e-6 else 0
                bits |= 4 if code & 4 and abs(py + ph - y1) < 1e-6 else 0
                bits |= 8 if code & 8 and abs(py - y0) < 1e-6 else 0
                row = cons[i].clone()
                row[4] = bits
                if k == 0:
                    cons[i] = row
                    area[i] = pw * ph
                    tp[i] = torch.tensor([px, py, pw, ph])
                else:
                    row[2] = 0
                    extra_rows.append(row)
                    extra_area.append(pw * ph)
                    extra_tp.append([px, py, pw, ph])
            centroid_pin[i] = len(pins)
            pins.append(tuple(poly.centroid.coords[0]))
        elif cons[i, 0]:
            s = G.shape_from_vertices(target_polygons[i])
            shapes[i] = s
            tp[i, 2], tp[i, 3] = s.w, s.h
    new_b2b, new_p2b = [], []
    for a, b, w in b2b.tolist():
        a, b = int(a), int(b)
        if a in centroid_pin and b in centroid_pin:
            continue
        if a in centroid_pin:
            new_p2b.append([centroid_pin[a], b, w])
        elif b in centroid_pin:
            new_p2b.append([centroid_pin[b], a, w])
        else:
            new_b2b.append([a, b, w])
    for k, b, w in p2b.tolist():
        if int(b) not in centroid_pin:
            new_p2b.append([k, b, w])
    if extra_rows:
        cons = torch.cat([cons, torch.stack(extra_rows)])
        area = torch.cat([area, torch.tensor(extra_area)])
        tp = torch.cat([tp, torch.tensor(extra_tp)])
    t = lambda rows, k: torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(0, k)
    return LiteProblem(len(area), area, t(new_b2b, 3), t(new_p2b, 3), t(pins, 2), cons, tp,
                       n, shapes, pre)


def instance_shapes(prob: LiteProblem) -> Optional[list]:
    """FloorplanInstance.shapes for the Lite problem: the Shape of each fixed
    block that isn't a rectangle, else None; None overall if there are none."""
    out = [None] * prob.block_count
    for i, s in enumerate(prob.shapes):
        if s is not None and not s.is_rect:
            out[i] = s
    return out if any(s is not None for s in out) else None


# -----------------------------------------------------------------------------
# Exact evaluation of a Lite layout whose fixed blocks are polygons
# -----------------------------------------------------------------------------
class _Expanded:
    """Constraint tensors with every polygon block split into its pieces
    (piece 0 keeps the index, the rest appended): overlap, area, dimensions,
    boundary and cluster contact are then exact under the Lite evaluator."""

    def __init__(self, instance):
        n = instance.block_count
        cons = instance.constraints[:n].clone().float()
        tp = instance.target_positions[:n].clone().float()
        area = instance.area_targets[:n].clone().float()
        self.owner, self.offset = [], []          # appended piece -> (block, (dx, dy, w, h))
        rows, extra_tp, extra_area = [], [], []
        for i, s in enumerate(instance.shapes):
            if s is None:
                continue
            code = int(cons[i, 4])
            for k, (px, py, pw, ph) in enumerate(s.pieces):
                bits = ((1 if code & 1 and px < 1e-6 else 0) | (2 if code & 2 and abs(px + pw - s.w) < 1e-6 else 0)
                        | (4 if code & 4 and abs(py + ph - s.h) < 1e-6 else 0) | (8 if code & 8 and py < 1e-6 else 0))
                row = cons[i].clone()
                row[4] = bits
                if k == 0:
                    cons[i] = row
                    tp[i, 2], tp[i, 3] = pw, ph
                    area[i] = pw * ph
                else:
                    row[2] = 0
                    rows.append(row)
                    extra_tp.append([-1, -1, pw, ph])
                    extra_area.append(pw * ph)
                    self.owner.append(i)
                    self.offset.append((px, py, pw, ph))
        if rows:
            cons = torch.cat([cons, torch.stack(rows)])
            tp = torch.cat([tp, torch.tensor(extra_tp)])
            area = torch.cat([area, torch.tensor(extra_area)])
        self.constraints, self.area = cons, area
        self.target_positions = [tuple(r) for r in tp.tolist()]
        self.shapes = instance.shapes

    def positions(self, positions):
        """Expanded rectangles, or None if a polygon block's box isn't its shape's."""
        out = list(positions)
        for i, s in enumerate(self.shapes):
            if s is None:
                continue
            x, y, w, h = positions[i]
            if abs(w - s.w) > 1e-4 or abs(h - s.h) > 1e-4:
                return None
            px, py, pw, ph = s.pieces[0]
            out[i] = (x + px, y + py, pw, ph)
        for i, (px, py, pw, ph) in zip(self.owner, self.offset):
            x, y = positions[i][:2]
            out.append((x + px, y + py, pw, ph))
        return out


_EXPANDED = {}


def _expanded(instance) -> _Expanded:
    key = id(instance.shapes)
    e = _EXPANDED.get(key)
    if e is None or e.shapes is not instance.shapes:
        e = _EXPANDED[key] = _Expanded(instance)
    return e


def centroids(instance, positions):
    """Block centroids: the box centre, or the polygon's centroid for shaped blocks."""
    shapes = instance.shapes or [None] * len(positions)
    return torch.tensor([(x + s.cx, y + s.cy) if s is not None else (x + w / 2, y + h / 2)
                         for (x, y, w, h), s in zip(positions, shapes)], dtype=torch.float64)


def polygon_metrics(instance, positions):
    """(hpwl_total, bbox_area, violations_relative, total_soft_violations, feasible)
    of a Lite layout with polygon blocks; HPWL from true centroids."""
    from iccad2026_evaluate import evaluate_solution
    e = _expanded(instance)
    ex = e.positions(positions)
    c = centroids(instance, positions)
    hp = 0.0
    b2b, p2b, pins = instance.b2b_connectivity, instance.p2b_connectivity, instance.pins_pos
    if b2b is not None and len(b2b):
        b2b = b2b[b2b[:, 0] >= 0]
        i, j, w = b2b[:, 0].long(), b2b[:, 1].long(), b2b[:, 2].double()
        hp += float(((c[i] - c[j]).abs().sum(1) * w).sum())
    if p2b is not None and len(p2b):
        p2b = p2b[p2b[:, 0] >= 0]
        k, j, w = p2b[:, 0].long(), p2b[:, 1].long(), p2b[:, 2].double()
        hp += float(((c[j] - pins[k].double()).abs().sum(1) * w).sum())
    if ex is None:
        return hp, 0.0, 0.0, 0, False
    m = evaluate_solution({'positions': ex, 'runtime': 1.0}, {'hpwl_baseline': 1.0, 'area_baseline': 1.0},
                          e.constraints, torch.zeros(0, 3), torch.zeros(0, 3), pins, e.area,
                          e.target_positions)
    return hp, m.bbox_area, m.violations_relative, int(m.total_soft_violations), bool(m.is_feasible)


def from_lite(prob: LiteProblem, positions: Sequence[Tuple[float, float, float, float]]) -> List[Vertices]:
    """Lite rectangles -> one polygon per Prime block."""
    out = []
    for i in range(prob.n_prime):
        x, y, w, h = positions[i]
        if prob.preplaced[i] is not None:
            out.append(list(prob.preplaced[i]))
        elif prob.shapes[i] is not None:
            out.append(list(prob.shapes[i].polygon(x, y).exterior.coords[:-1]))
        else:
            out.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    return out

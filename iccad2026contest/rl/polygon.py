"""
Rectilinear polygon geometry for FloorSet-Prime blocks.

Every Prime block is a rectilinear polygon (4, 6 or 8 corners in the test
set: rectangles, L, T/U/Z/step shapes). The solver treats each one as a rigid
group of axis-aligned rectangles ("pieces") at fixed offsets from the
polygon's lower-left bounding-box corner, so the rectangle machinery in rl/
(packing, overlap, touch tests) carries over piece by piece.

A Shape is position-free: its pieces live in [0, w] x [0, h]. A placed block
is (shape, x, y) with (x, y) the bounding box's lower-left corner.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

Rect = Tuple[float, float, float, float]          # (x, y, w, h)
EPS = 1e-6


@dataclass(frozen=True)
class Shape:
    w: float
    h: float
    pieces: Tuple[Rect, ...]          # offsets from the bbox lower-left corner
    cx: float                          # centroid offset from the bbox lower-left corner
    cy: float
    area: float

    @property
    def is_rect(self) -> bool:
        return len(self.pieces) == 1

    def polygon(self, x: float = 0.0, y: float = 0.0) -> Polygon:
        return unary_union([box(x + px, y + py, x + px + pw, y + py + ph)
                            for px, py, pw, ph in self.pieces]).simplify(0)

    def placed(self, x: float, y: float) -> List[Rect]:
        return [(x + px, y + py, pw, ph) for px, py, pw, ph in self.pieces]


def clean_vertices(vertices) -> List[Tuple[float, float]]:
    """Vertex list from a Prime tensor/list: drops -1 padding and the closing repeat."""
    pts = [(float(p[0]), float(p[1])) for p in vertices if not (p[0] == -1 and p[1] == -1)]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def decompose(poly: Polygon) -> List[Rect]:
    """Split a rectilinear polygon into rectangles: horizontal slabs between
    consecutive vertex y's, then vertically adjacent slabs with the same x
    interval merged. At most 3 pieces for the shapes in the Prime test set."""
    ys = []
    for y in sorted(y for _, y in poly.exterior.coords):
        if not ys or y - ys[-1] > EPS:
            ys.append(y)
    x0, _, x1, _ = poly.bounds
    slabs = []
    for ya, yb in zip(ys, ys[1:]):
        # x intervals from a line through the slab's middle: slab boxes cut at
        # rounded y's can overlap the next slab by a sliver and join its pieces
        cut = poly.intersection(LineString([(x0 - 1, (ya + yb) / 2), (x1 + 1, (ya + yb) / 2)]))
        geoms = getattr(cut, "geoms", [cut])
        ivs = sorted((g.bounds[0], g.bounds[2]) for g in geoms if g.length > EPS)
        slabs.append((ya, yb, ivs))
    rects, open_ = [], {}          # open_: (xa, xb) -> [y_start, y_end]
    for ya, yb, ivs in slabs:
        nxt = {}
        for iv in ivs:
            if iv in open_ and abs(open_[iv][1] - ya) < EPS:
                nxt[iv] = [open_.pop(iv)[0], yb]
            else:
                nxt[iv] = [ya, yb]
        for (xa, xb), (s, e) in open_.items():
            rects.append((xa, s, xb - xa, e - s))
        open_ = nxt
    for (xa, xb), (s, e) in open_.items():
        rects.append((xa, s, xb - xa, e - s))
    return rects


def shape_from_polygon(poly: Polygon) -> Shape:
    """Position-free Shape of a placed polygon (offsets from its bbox corner)."""
    x0, y0, x1, y1 = poly.bounds
    pieces = tuple(sorted((px - x0, py - y0, pw, ph) for px, py, pw, ph in decompose(poly)))
    a = sum(pw * ph for _, _, pw, ph in pieces)
    cx = sum((px + pw / 2) * pw * ph for px, py, pw, ph in pieces) / a
    cy = sum((py + ph / 2) * pw * ph for px, py, pw, ph in pieces) / a
    return Shape(x1 - x0, y1 - y0, pieces, cx, cy, a)


def shape_from_vertices(vertices) -> Shape:
    return shape_from_polygon(Polygon(clean_vertices(vertices)))


def rect_shape(w: float, h: float) -> Shape:
    return Shape(w, h, ((0.0, 0.0, w, h),), w / 2, h / 2, w * h)


def transform(shape: Shape, k: int) -> Shape:
    """One of the 8 rotations/flips (k = 0..7: k & 3 quarter turns, k & 4 mirror
    in x first), re-anchored so the bbox starts at (0, 0)."""
    def pt(x, y):
        if k & 4:
            x = shape.w - x
        for _ in range(k & 3):          # quarter turn: (x, y) -> (-y, x)
            x, y = -y, x
        return x, y
    pieces = []
    for px, py, pw, ph in shape.pieces:
        (ax, ay), (bx, by) = pt(px, py), pt(px + pw, py + ph)
        pieces.append((min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay)))
    mx, my = min(p[0] for p in pieces), min(p[1] for p in pieces)
    pieces = tuple(sorted((px - mx, py - my, pw, ph) for px, py, pw, ph in pieces))
    cx, cy = pt(shape.cx, shape.cy)
    w, h = (shape.h, shape.w) if k & 1 else (shape.w, shape.h)
    return Shape(w, h, pieces, cx - mx, cy - my, shape.area)


def orientations(shape: Shape, mirror: bool = False) -> List[Shape]:
    """The distinct rotations (and, with mirror, flips) of a shape: 1 for a
    square, 2 for a rectangle, up to 4 (8 mirrored) for an asymmetric L.
    Prime allows fixed shapes to be translated or rotated, not mirrored."""
    out, seen = [], set()
    for k in range(8 if mirror else 4):
        t = transform(shape, k)
        key = (round(t.w, 6), round(t.h, 6), tuple(tuple(round(v, 6) for v in p) for p in t.pieces))
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


@lru_cache(maxsize=None)
def _canon_cached(coords: Tuple[Tuple[float, float], ...], mirror: bool) -> Tuple:
    poly = Polygon(coords)
    keys = []
    for t in orientations(shape_from_polygon(poly), mirror):
        g = shapely.normalize(t.polygon())
        keys.append(_dedupe(tuple((round(x, 3), round(y, 3)) for x, y in g.exterior.coords[:-1])))
    return min(keys)


def _dedupe(pts: Tuple) -> Tuple:
    """Drop repeated and straight-run vertices left after rounding: float noise from
    a fractional position made an exact fixed shape read as different (10-22)."""
    pts = list(pts)
    changed = True
    while changed and len(pts) > 4:
        changed = False
        for k in range(len(pts)):
            a, b, c = pts[k - 1], pts[k], pts[(k + 1) % len(pts)]
            if a == b or a[0] == b[0] == c[0] or a[1] == b[1] == c[1]:
                del pts[k]
                changed = True
                break
    return tuple(pts)


def canonical_key(poly: Polygon, mirror: bool = False) -> Tuple:
    """Shape identity up to translation, rotation (and mirroring) and vertex
    order. FloorSet's polygons_have_same_shape depends on the starting vertex,
    so it calls identical rectangles listed from different corners different."""
    return _canon_cached(tuple(poly.simplify(0).exterior.coords[:-1]), mirror)


def same_shape(a: Polygon, b: Polygon, mirror: bool = False) -> bool:
    return abs(a.area - b.area) <= 1e-3 and canonical_key(a, mirror) == canonical_key(b, mirror)


def rects_overlap_area(a: Sequence[Rect], b: Sequence[Rect]) -> float:
    tot = 0.0
    for ax, ay, aw, ah in a:
        for bx, by, bw, bh in b:
            dx = min(ax + aw, bx + bw) - max(ax, bx)
            dy = min(ay + ah, by + bh) - max(ay, by)
            if dx > EPS and dy > EPS:
                tot += dx * dy
    return tot

"""
GridPlacementEnv: a sequential, grid-based placement environment that
guarantees the contest's hard constraints by construction (see
rl/ordering.py for role assignment and the module docstring below for how
each constraint is handled). No learning happens here -- this is pure
placement mechanics; rl/networks.py + rl/ppo.py decide *which* action to
take, this module guarantees that *any* action it allows is safe.

Guarantees:
    - Overlap:  position actions are masked to grid cells whose full
                (conservatively rounded-up) footprint is currently free.
    - Area:     free-block (w, h) are derived from the block's own area
                target and a chosen aspect ratio, so area == target exactly.
    - Fixed/preplaced dimensions: taken verbatim from target_positions,
                never altered by an action.
    - MIB:      only the first-placed ("leader") member of a group chooses
                a shape; every other member copies it exactly -- UNLESS its
                own area target isn't within the 1% hard-constraint
                tolerance of the leader's shape (real data does have MIB
                groups whose members carry different target areas; the two
                requirements are then mathematically incompatible, since
                exact-copy would break the follower's own hard area
                constraint). In that case the follower keeps the leader's
                aspect ratio (best-effort MIB similarity) but sizes itself
                to its own area target, so the hard constraint always wins
                over the soft one.
    - Boundary: the coordinate(s) implied by the required edge bit(s) are
                pinned to the working canvas edge, which nothing can ever
                extend past -- guaranteed *unless* the pinned cell is
                already occupied by an earlier boundary-pinned block, in
                which case we fall back to a free placement (recorded via
                `boundary_fallback_count`) rather than risk an overlap.
    - Grouping: not fully guaranteed (order-dependent across blocks already
                placed for *other* clusters), but any cluster member after
                the first searches every grid-aligned exact geometric touch
                against an already-placed clustermate (checked against true
                placed rectangles, not the conservative grid, so this can
                never introduce an overlap -- see _try_cluster_touch); if no
                safe touch exists anywhere along any clustermate's edges it
                falls back to an adjacency-preferring masked position, then
                to a plain free mask -- both of which are grid-dilation
                based and can select a merely-nearby (even corner-only)
                cell that doesn't actually count as touching.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from iccad2026_evaluate import AREA_TOLERANCE

from .data import FloorplanInstance
from .ordering import PlacementStep, compute_order

DEFAULT_GRID_DIM = 48
# Slack beyond total block area for placement to always fit. Was 2.2
# (~4.84x area) -- an undertrained policy's bbox matched that ceiling almost
# exactly instead of packing tight; 1.5 (~2.25x) halves the free spreading room.
CANVAS_PADDING = 1.5
# Log-spaced aspect ratio buckets (w/h); index 4 == square.
ASPECT_RATIOS = [0.2, 0.3, 0.45, 0.67, 1.0, 1.5, 2.22, 3.33, 5.0]


def _rect_overlap(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ox = min(ax + aw, bx + bw) - max(ax, bx)
    oy = min(ay + ah, by + bh) - max(ay, by)
    return ox > 1e-9 and oy > 1e-9


class GridPlacementEnv:
    def __init__(self, instance: FloorplanInstance, grid_dim: int = DEFAULT_GRID_DIM,
                 aspect_ratios: List[float] = ASPECT_RATIOS):
        self.instance = instance
        self.grid_dim = grid_dim
        self.aspect_ratios = aspect_ratios
        self.plan = compute_order(instance.constraints, instance.area_targets,
                                   instance.b2b_connectivity)
        self.reset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def reset(self):
        inst = self.instance
        n = inst.block_count
        ncols = inst.constraints.shape[1] if inst.constraints.dim() > 1 else 0

        def col(j):
            return inst.constraints[:, j] if ncols > j else torch.zeros(n)

        preplaced_mask = col(1) != 0
        cluster_col = col(3).long()

        xs0, ys0, xs1, ys1 = [0.0], [0.0], [], []
        for i in range(n):
            if preplaced_mask[i]:
                x, y, w, h = [float(v) for v in inst.target_positions[i].tolist()]
                xs0.append(x)
                ys0.append(y)
                xs1.append(x + w)
                ys1.append(y + h)

        total_area = float(inst.area_targets.clamp(min=0).sum().item())
        est_side = math.sqrt(max(total_area, 1.0)) * CANVAS_PADDING

        self.x_min = min(xs0)
        self.y_min = min(ys0)
        self.x_max = max([self.x_min + est_side] + xs1)
        self.y_max = max([self.y_min + est_side] + ys1)

        self.cell_w = (self.x_max - self.x_min) / self.grid_dim
        self.cell_h = (self.y_max - self.y_min) / self.grid_dim

        self.occupancy = torch.zeros(self.grid_dim, self.grid_dim)
        self.positions: List[Optional[Tuple[float, float, float, float]]] = [None] * n
        self.mib_shape: Dict[int, Tuple[float, float]] = {}
        self.cluster_cells: Dict[int, set] = defaultdict(set)
        self.boundary_fallback_count = 0
        self.grouping_fallback_count = 0

        # Adjacency for wiremask() -- built once so each step's wiremask
        # computation is just a lookup, not a rescan of every edge.
        self._b2b_adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        b2b = inst.b2b_connectivity
        if b2b is not None and b2b.numel() > 0:
            valid = b2b[b2b[:, 0] >= 0]
            for edge in valid:
                i, j, wt = int(edge[0]), int(edge[1]), float(edge[2])
                self._b2b_adj[i].append((j, wt))
                self._b2b_adj[j].append((i, wt))
        self._p2b_adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        p2b = inst.p2b_connectivity
        if p2b is not None and p2b.numel() > 0:
            valid = p2b[p2b[:, 0] >= 0]
            for edge in valid:
                pin_idx, block_idx, wt = int(edge[0]), int(edge[1]), float(edge[2])
                self._p2b_adj[block_idx].append((pin_idx, wt))

        # Running bbox over placed content only (not the canvas) -- seeded
        # from preplaced blocks so the first pending block's step_deltas()
        # doesn't get blamed for area that was already fixed before the
        # rollout started. None means "nothing placed yet".
        self._bbox: Optional[List[float]] = None
        self._resolved_wl = 0.0
        self.last_step_deltas: Tuple[float, float] = (0.0, 0.0)

        for i in range(n):
            if preplaced_mask[i]:
                x, y, w, h = [float(v) for v in inst.target_positions[i].tolist()]
                self.positions[i] = (x, y, w, h)
                self._mark_occupied(x, y, w, h)
                cid = int(cluster_col[i])
                if cid:
                    self._mark_cluster(cid, x, y, w, h)
                self._grow_bbox(x, y, w, h)

        self._pending = [s for s in self.plan.order if s.role != 'preplaced']
        self._cursor = 0
        self._pending_shape: Optional[Tuple[float, float]] = None
        return None

    # ------------------------------------------------------------------
    # Grid <-> continuous helpers
    # ------------------------------------------------------------------
    def _rect_to_cells(self, x, y, w, h):
        # Epsilon nudges guard against floating-point noise landing a value
        # that is mathematically exactly on a cell boundary just off of it
        # (e.g. 6.999999999996 for an intended 7.0) -- floor needs "+eps" so
        # such a value still rounds up to 7, and ceil needs "-eps" so a value
        # like 9.000000001 still rounds down to 9. Getting the sign backwards
        # here silently over-reserves an extra row/column on the *start* side
        # of every grid-aligned placement (which every one of our own
        # placements is, since they originate from x_min + gx*cell_w).
        gx0 = int(math.floor((x - self.x_min) / self.cell_w + 1e-9))
        gy0 = int(math.floor((y - self.y_min) / self.cell_h + 1e-9))
        gx1 = int(math.ceil((x + w - self.x_min) / self.cell_w - 1e-9))
        gy1 = int(math.ceil((y + h - self.y_min) / self.cell_h - 1e-9))
        gx0 = max(0, min(gx0, self.grid_dim))
        gy0 = max(0, min(gy0, self.grid_dim))
        gx1 = max(gx0, min(gx1, self.grid_dim))
        gy1 = max(gy0, min(gy1, self.grid_dim))
        return gx0, gy0, gx1, gy1

    def _mark_occupied(self, x, y, w, h):
        gx0, gy0, gx1, gy1 = self._rect_to_cells(x, y, w, h)
        if gx1 > gx0 and gy1 > gy0:
            self.occupancy[gy0:gy1, gx0:gx1] = 1.0

    def _mark_cluster(self, cid, x, y, w, h):
        gx0, gy0, gx1, gy1 = self._rect_to_cells(x, y, w, h)
        for gy in range(gy0, gy1):
            for gx in range(gx0, gx1):
                self.cluster_cells[cid].add((gy, gx))

    def _grow_bbox(self, x, y, w, h) -> float:
        """Extends the running placed-content bbox to include (x,y,w,h);
        returns the resulting area *increase* (bbox area only ever grows as
        points are added, so this is always >= 0). Used by _commit() to
        attribute the exact, causal share of final bbox_area to the step
        that caused it -- see rl/reward.py's _step_quality_delta."""
        x1, y1 = x + w, y + h
        if self._bbox is None:
            self._bbox = [x, y, x1, y1]
            return (x1 - x) * (y1 - y)
        old_area = (self._bbox[2] - self._bbox[0]) * (self._bbox[3] - self._bbox[1])
        self._bbox[0] = min(self._bbox[0], x)
        self._bbox[1] = min(self._bbox[1], y)
        self._bbox[2] = max(self._bbox[2], x1)
        self._bbox[3] = max(self._bbox[3], y1)
        new_area = (self._bbox[2] - self._bbox[0]) * (self._bbox[3] - self._bbox[1])
        return new_area - old_area

    def _footprint_cells(self, w, h):
        cells_w = min(self.grid_dim, max(1, math.ceil(w / self.cell_w - 1e-9)))
        cells_h = min(self.grid_dim, max(1, math.ceil(h / self.cell_h - 1e-9)))
        return cells_w, cells_h

    # ------------------------------------------------------------------
    # Step introspection
    # ------------------------------------------------------------------
    def done(self) -> bool:
        return self._cursor >= len(self._pending)

    def current_step(self) -> PlacementStep:
        return self._pending[self._cursor]

    def needs_aspect(self) -> bool:
        return self.current_step().role == 'free'

    def current_area(self) -> float:
        return float(self.instance.area_targets[self.current_step().block_idx])

    def current_shape(self) -> Tuple[float, float]:
        """(w, h) for the current step when it does not need an aspect
        choice (fixed / mib_follower)."""
        step = self.current_step()
        i = step.block_idx
        if step.role == 'fixed':
            tp = self.instance.target_positions[i]
            return float(tp[2]), float(tp[3])
        if step.role == 'mib_follower':
            leader_w, leader_h = self.mib_shape[step.mib_leader]
            own_area = float(self.instance.area_targets[i])
            leader_area = leader_w * leader_h
            if own_area > 0 and abs(leader_area - own_area) / own_area <= AREA_TOLERANCE:
                return leader_w, leader_h
            # Own area target is incompatible with an exact copy (see module
            # docstring): keep the leader's aspect ratio, but size to this
            # block's own area so its hard area constraint still holds.
            aspect = leader_w / leader_h
            return math.sqrt(own_area * aspect), math.sqrt(own_area / aspect)
        raise ValueError(f"current_shape() called for role={step.role}; call choose_aspect() first")

    def choose_aspect(self, aspect_idx: int) -> Tuple[float, float]:
        step = self.current_step()
        assert step.role == 'free', f"choose_aspect() invalid for role={step.role}"
        area = self.current_area()
        r = self.aspect_ratios[aspect_idx]
        w = math.sqrt(area * r)
        h = math.sqrt(area / r)
        self._pending_shape = (w, h)
        return w, h

    # ------------------------------------------------------------------
    # Position action
    # ------------------------------------------------------------------
    def position_mask(self, w: float, h: float):
        """Returns (mask, cells_w, cells_h) where mask is a bool tensor of
        shape [out_h, out_w] over valid top-left grid positions -- or None
        if the block was auto-placed deterministically (cluster touch) and
        no position action is needed."""
        step = self.current_step()
        cid = step.cluster_id
        self._boundary_pin_ok = False

        if step.boundary_code == 0 and cid and self.cluster_cells.get(cid):
            touch = self._try_cluster_touch(cid, w, h)
            if touch is not None:
                x, y = touch
                self._commit(x, y, w, h)
                return None

        cells_w, cells_h = self._footprint_cells(w, h)
        occ = self.occupancy.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones(1, 1, cells_h, cells_w)
        window_sum = F.conv2d(occ, kernel)[0, 0]
        free = window_sum == 0

        if free.numel() == 0 or not bool(free.any()):
            raise RuntimeError(
                f"No free position for block {step.block_idx} (w={w}, h={h}); "
                f"canvas too small -- increase CANVAS_PADDING or grid_dim."
            )

        code = step.boundary_code
        if code:
            mask = self._boundary_mask(free, cells_w, cells_h, code)
            if mask is not None:
                self._boundary_pin_ok = True
                return mask, cells_w, cells_h
            self.boundary_fallback_count += 1
            return free, cells_w, cells_h

        if cid:
            adj = self._adjacency_mask(free, cells_w, cells_h, cid)
            if adj is not None and bool(adj.any()):
                return adj, cells_w, cells_h
            if self.cluster_cells.get(cid):
                self.grouping_fallback_count += 1

        # Soft compactness preference: cells touching any already-placed
        # block, before falling back to the fully unrestricted free mask --
        # a strict subset of `free`, so this can only narrow the candidate
        # set, never introduce an illegal position.
        compact = self._compactness_mask(free, cells_w, cells_h)
        if bool(compact.any()):
            return compact, cells_w, cells_h
        return free, cells_w, cells_h

    def _compactness_mask(self, free, cells_w, cells_h):
        dilated = F.max_pool2d(self.occupancy.unsqueeze(0).unsqueeze(0),
                                kernel_size=3, stride=1, padding=1)[0, 0]
        kernel = torch.ones(1, 1, cells_h, cells_w)
        window_sum = F.conv2d(dilated.unsqueeze(0).unsqueeze(0), kernel)[0, 0]
        return (window_sum > 0) & free

    def _boundary_mask(self, free, cells_w, cells_h, code):
        out_h, out_w = free.shape
        mask = free.clone()
        if code & 1:  # left
            col = torch.zeros_like(mask)
            col[:, 0] = True
            mask &= col
        if code & 2:  # right
            col = torch.zeros_like(mask)
            col[:, out_w - 1] = True
            mask &= col
        if code & 4:  # top
            row = torch.zeros_like(mask)
            row[out_h - 1, :] = True
            mask &= row
        if code & 8:  # bottom
            row = torch.zeros_like(mask)
            row[0, :] = True
            mask &= row
        if bool(mask.any()):
            return mask
        return None

    def _adjacency_mask(self, free, cells_w, cells_h, cid):
        cluster_grid = torch.zeros(self.grid_dim, self.grid_dim)
        for (gy, gx) in self.cluster_cells[cid]:
            cluster_grid[gy, gx] = 1.0
        dilated = F.max_pool2d(cluster_grid.unsqueeze(0).unsqueeze(0),
                                kernel_size=3, stride=1, padding=1)[0, 0]
        kernel = torch.ones(1, 1, cells_h, cells_w)
        window_sum = F.conv2d(dilated.unsqueeze(0).unsqueeze(0), kernel)[0, 0]
        return (window_sum > 0) & free

    def wiremask(self, w: float, h: float) -> torch.Tensor:
        """[grid_dim, grid_dim] channel giving the current block's placement
        head a direct spatial HPWL signal, which occupancy/cluster_grid
        alone don't provide (see rl/networks.py's PositionCNN docstring):
        at each candidate top-left cell, the weighted-Manhattan wirelength
        cost of centering the block there against every already-placed
        connected block/pin -- the exact same b2b/p2b formula
        iccad2026_evaluate.py scores with (calculate_hpwl_b2b/_p2b), for
        just the portion of each net that's resolved so far.

        Manhattan distance separates additively into independent x and y
        terms, so this is two O(grid_dim) sweeps against an outer sum
        rather than an O(grid_dim^2) loop over cells.
        """
        i = self.current_step().block_idx
        targets_x: List[Tuple[float, float]] = []  # (target x, weight)
        targets_y: List[Tuple[float, float]] = []
        for j, wt in self._b2b_adj.get(i, []):
            pos = self.positions[j]
            if pos is not None:
                targets_x.append((pos[0] + pos[2] / 2, wt))
                targets_y.append((pos[1] + pos[3] / 2, wt))
        for pin_idx, wt in self._p2b_adj.get(i, []):
            px, py = self.instance.pins_pos[pin_idx].tolist()
            targets_x.append((float(px), wt))
            targets_y.append((float(py), wt))

        if not targets_x:
            return torch.zeros(self.grid_dim, self.grid_dim)

        cells = torch.arange(self.grid_dim, dtype=torch.float32)
        cell_x = self.x_min + cells * self.cell_w + w / 2
        cell_y = self.y_min + cells * self.cell_h + h / 2

        fx = torch.zeros(self.grid_dim)
        for tx, wt in targets_x:
            fx += wt * (cell_x - tx).abs()
        fy = torch.zeros(self.grid_dim)
        for ty, wt in targets_y:
            fy += wt * (cell_y - ty).abs()

        cost = fy.unsqueeze(1) + fx.unsqueeze(0)  # [gy, gx]
        # Per-step normalized attraction into [0, 1] (higher = better): raw
        # weighted distances have no fixed scale across instances/edge
        # weights, but relative-to-this-step's-own-spread does. Normalizing
        # by the canvas-dimension sum instead (as this used to) crushed
        # values down to ~1e-3, two orders of magnitude below the
        # occupancy/cluster_grid channels' 0-1 range and too faint to
        # compete with them in PositionCNN's conv1 -- see algorithm.md.
        spread = max((cost.max() - cost.min()).item(), 1e-6)
        return (cost.max() - cost) / spread

    def _grid_values(self, lo: float, hi: float, step: float, anchor: float):
        """Grid-aligned coordinate values (anchor + k*step) in [lo, hi],
        closest-to-anchor first -- used by _try_cluster_touch to search
        every legal touching offset along a shared edge, not just the one
        implied by the neighbor's own position (see its docstring)."""
        if hi < lo - 1e-9 or step <= 0:
            return
        k_lo = math.ceil((lo - anchor) / step - 1e-9)
        k_hi = math.floor((hi - anchor) / step + 1e-9)
        for k in range(k_lo, k_hi + 1):
            yield anchor + k * step

    def _try_cluster_touch(self, cid, w, h):
        """Exact geometric touch against an already-placed clustermate --
        see the module docstring's Grouping guarantee.

        For each neighbor and each of the 4 sides, the set of positions
        that would produce a genuine (positive-length, not just corner)
        touch is a whole *range* along the shared edge, not the single
        offset implied by copying the neighbor's own coordinate: any y
        where [y, y+h] overlaps [ny, ny+nh] gives a real right/left touch,
        and symmetrically for x on the top/bottom sides. Only trying the
        one clamped offset (the previous version) meant a single collision
        there gave up on that neighbor entirely and fell through to
        rl/env.py's imprecise, grid-dilation-based adjacency mask -- which
        can select a merely-nearby (even corner-only) cell that position_mask()
        allows but evaluate_solution's real connected-components check
        does not count as touching, showing up as a `grouping_violations`
        soft-constraint hit with no corresponding `grouping_fallback_count`
        (diagnosed 2026-09-13: violations were consistently several times
        higher than the fallback counter, which this exhaustive search
        directly targets). Trying every grid-aligned offset along that
        range before moving to the next neighbor/side finds a real touch
        far more often, without ever weakening the overlap check itself."""
        n = self.instance.block_count
        ncols = self.instance.constraints.shape[1]
        cluster_col = self.instance.constraints[:, 3] if ncols > 3 else torch.zeros(n)
        neighbors = [self.positions[j] for j in range(n)
                     if self.positions[j] is not None and int(cluster_col[j]) == cid]
        for (nx, ny, nw, nh) in neighbors:
            y_lo, y_hi = max(self.y_min, ny - h), min(self.y_max - h, ny + nh)
            for cx in (nx + nw, nx - w):
                if cx < self.x_min - 1e-9 or cx + w > self.x_max + 1e-9:
                    continue
                for cy in self._grid_values(y_lo, y_hi, self.cell_h, ny):
                    candidate = (cx, cy, w, h)
                    if not any(p is not None and _rect_overlap(candidate, p) for p in self.positions):
                        return cx, cy
            x_lo, x_hi = max(self.x_min, nx - w), min(self.x_max - w, nx + nw)
            for cy in (ny + nh, ny - h):
                if cy < self.y_min - 1e-9 or cy + h > self.y_max + 1e-9:
                    continue
                for cx in self._grid_values(x_lo, x_hi, self.cell_w, nx):
                    candidate = (cx, cy, w, h)
                    if not any(p is not None and _rect_overlap(candidate, p) for p in self.positions):
                        return cx, cy
        return None

    def _commit(self, x, y, w, h):
        step = self.current_step()
        i = step.block_idx
        self.positions[i] = (x, y, w, h)
        self._mark_occupied(x, y, w, h)
        if step.role == 'free':
            pass  # shape already recorded via choose_aspect
        if step.mib_leader == -1 and self._is_mib_leader(i):
            self.mib_shape[i] = (w, h)
        if step.cluster_id:
            self._mark_cluster(step.cluster_id, x, y, w, h)
        self.last_step_deltas = self._step_deltas(i, x, y, w, h)
        self._pending_shape = None
        self._cursor += 1

    def _step_deltas(self, i, x, y, w, h) -> Tuple[float, float]:
        """Exact, causal (x,y,w,h)-caused increase in total b2b+p2b
        wirelength and bbox area, at the moment block i is committed --
        every b2b/p2b edge is resolved exactly once, at whichever endpoint
        is placed second (the other's position is already known), so
        summing this over every _commit() call plus one residual
        correction reproduces evaluate_solution's totals exactly. See
        rl/reward.py's _step_quality_delta, which consumes this."""
        cx, cy = x + w / 2, y + h / 2
        delta_wl = 0.0
        for j, wt in self._b2b_adj.get(i, []):
            pj = self.positions[j]
            if pj is not None and j != i:
                pcx, pcy = pj[0] + pj[2] / 2, pj[1] + pj[3] / 2
                delta_wl += wt * (abs(cx - pcx) + abs(cy - pcy))
        for pin_idx, wt in self._p2b_adj.get(i, []):
            px, py = self.instance.pins_pos[pin_idx].tolist()
            delta_wl += wt * (abs(cx - px) + abs(cy - py))
        delta_area = self._grow_bbox(x, y, w, h)
        return delta_wl, delta_area

    def _is_mib_leader(self, i):
        ncols = self.instance.constraints.shape[1]
        return ncols > 2 and self.instance.constraints[i, 2] != 0

    def place(self, gy: int, gx: int):
        step = self.current_step()
        if step.role == 'free':
            w, h = self._pending_shape
        else:
            w, h = self.current_shape()

        x = self.x_min + gx * self.cell_w
        y = self.y_min + gy * self.cell_h

        code = step.boundary_code
        if code and self._boundary_pin_ok:
            # Safe because position_mask() already restricted the candidate
            # set to gx/gy=0 or the rightmost/topmost valid cell for the
            # required bit(s); shifting flush to the canvas edge here only
            # ever moves the footprint further into that already-verified
            # free window (see env.py module docstring), never into
            # unverified space.
            if code & 1:
                x = self.x_min
            if code & 2:
                x = self.x_max - w
            if code & 4:
                y = self.y_max - h
            if code & 8:
                y = self.y_min

        self._commit(x, y, w, h)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def finalize(self) -> List[Tuple[float, float, float, float]]:
        assert self.done(), "finalize() called before all blocks were placed"
        assert all(p is not None for p in self.positions), "unplaced block remains"
        return list(self.positions)

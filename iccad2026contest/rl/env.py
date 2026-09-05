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
                a shape; every other member copies it exactly.
    - Boundary: the coordinate(s) implied by the required edge bit(s) are
                pinned to the working canvas edge, which nothing can ever
                extend past -- guaranteed *unless* the pinned cell is
                already occupied by an earlier boundary-pinned block, in
                which case we fall back to a free placement (recorded via
                `boundary_fallback_count`) rather than risk an overlap.
    - Grouping: not fully guaranteed (order-dependent across blocks already
                placed for *other* clusters), but any cluster member after
                the first attempts an exact geometric touch against an
                already-placed clustermate (checked against true placed
                rectangles, not the conservative grid, so this can never
                introduce an overlap); if no safe touch exists it falls back
                to an adjacency-preferring masked position, then to a plain
                free mask.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .data import FloorplanInstance
from .ordering import PlacementStep, compute_order

DEFAULT_GRID_DIM = 48
# Generous slack: the working canvas is never what gets scored (the scored
# bbox is the tight box around actually-placed blocks), so extra room here
# only helps packing succeed -- it doesn't inflate the final area cost.
CANVAS_PADDING = 2.2
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

        for i in range(n):
            if preplaced_mask[i]:
                x, y, w, h = [float(v) for v in inst.target_positions[i].tolist()]
                self.positions[i] = (x, y, w, h)
                self._mark_occupied(x, y, w, h)
                cid = int(cluster_col[i])
                if cid:
                    self._mark_cluster(cid, x, y, w, h)

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
            return self.mib_shape[step.mib_leader]
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

        return free, cells_w, cells_h

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

    def _try_cluster_touch(self, cid, w, h):
        n = self.instance.block_count
        ncols = self.instance.constraints.shape[1]
        cluster_col = self.instance.constraints[:, 3] if ncols > 3 else torch.zeros(n)
        neighbors = [self.positions[j] for j in range(n)
                     if self.positions[j] is not None and int(cluster_col[j]) == cid]
        for (nx, ny, nw, nh) in neighbors:
            # For a horizontal touch (left/right), the touching x-coordinate
            # is exact; the free y-coordinate is clamped into canvas bounds
            # rather than copied verbatim from the neighbor, since [ny, ny+nh]
            # and [y_min, y_max] both contain ny, this clamp is guaranteed to
            # still overlap [ny, ny+nh] (i.e. still a real touch) -- see
            # env.py module docstring for the general argument. Symmetric for
            # vertical touches (top/bottom).
            cy_h = min(max(ny, self.y_min), self.y_max - h)
            cx_v = min(max(nx, self.x_min), self.x_max - w)
            for (cx, cy) in ((nx + nw, cy_h), (nx - w, cy_h), (cx_v, ny + nh), (cx_v, ny - h)):
                if cx < self.x_min - 1e-9 or cx + w > self.x_max + 1e-9:
                    continue
                if cy < self.y_min - 1e-9 or cy + h > self.y_max + 1e-9:
                    continue
                candidate = (cx, cy, w, h)
                if any(p is not None and _rect_overlap(candidate, p) for p in self.positions):
                    continue
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
        self._pending_shape = None
        self._cursor += 1

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

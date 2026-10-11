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

                A block that's BOTH boundary-pinned AND clustered runs the
                same touch search first, filtered to candidates that also
                satisfy its own boundary bit(s) -- fixed 2026-09-21 after
                validation-set data showed clusters with 2+ boundary-pinned
                members failing to stay connected 97.8% of the time
                (45/46), vs. 46.5% for ordinary clusters. The prior version
                only ever ran the touch search when `boundary_code == 0`,
                so a boundary-pinned cluster member never even attempted to
                satisfy grouping -- each one independently picked any free
                cell on its required edge, with zero regard for where its
                clustermates were. See [[floorset-grouping]] memory (or
                this commit's message) for the full diagnosis.
"""

import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from iccad2026_evaluate import AREA_TOLERANCE

from .data import FloorplanInstance
from .ordering import PlacementStep, compute_order

DEFAULT_GRID_DIM = 48
# Slack beyond total block area for placement to always fit. Was 2.2
# (~4.84x area), then 1.5 (~2.25x); lowered to 1.4 (~1.96x) 2026-09-20 to
# test whether a tighter working canvas -- combined with the fixed per-step
# credit assignment (violation_factor now correctly propagated, see
# rl/ppo.py) -- reduces the block-50-style pattern where a boundary-pinned
# block anchors to this canvas's edge (rl/env.py's _boundary_mask, set once
# here before any placement happens) far from where the interior mass ends
# up settling. 1.2 was tried first and rejected outright: it reproduced the
# exact catastrophic whole-batch avg_reward=-10.0000 flatline this project
# already fixed once, but for a different reason (canvas too small to
# complete most rollouts at all, not reward-capping) -- confirmed by
# test_ppo_overfit.py and most of test_env_real_data.py's validation-set
# stress tests failing outright even with 20 random-seed retries per
# instance. 1.35 still failed 9/192 tests (specific instances with an
# extreme-aspect-ratio block). 1.4 passes the full suite cleanly, so this
# is the tightest value tried that doesn't break basic feasibility -- see
# training-history notes for whether it actually helps. Now only a
# fallback -- see GridPlacementEnv._estimate_canvas_size -- for the rare
# instance with too few usable pins to size from directly.
CANVAS_PADDING = 1.4
# Log-spaced aspect ratio buckets (w/h); index 4 == square.
ASPECT_RATIOS = [0.2, 0.3, 0.45, 0.67, 1.0, 1.5, 2.22, 3.33, 5.0]

# Target utilization (total block area / working canvas area) used to size
# the canvas -- see GridPlacementEnv._estimate_canvas_size. SHAPE (aspect
# ratio) and AREA are deliberately estimated from two different, exact
# sources rather than one multiplicative fudge factor on the pin bbox:
#   - shape: pins_pos's own bounding-box ratio (validated across the
#     validation set: median 5.4%, mean 6.0%, max 17.5% relative error vs
#     the true ground-truth ratio -- a real per-instance shape signal).
#   - area: area_targets.sum() (the EXACT total block area already in the
#     problem input) divided by this utilization target, not the pin
#     bbox's own area (which itself carries ~22% inflation over true area
#     on average -- compounding two approximations was needless).
# A prior version multiplied the pin bbox by one linear-per-dimension
# margin (PIN_SAFETY_MARGIN / PIN_AREA_MARGIN) -- this conflated shape and
# area uncertainty into one knob and, worse, the very first form of it
# applied the margin per-dimension, which compounds into AREA^2: 1.4 per
# dimension produced a median canvas area of 2.4x (mean 2.395x, max
# 2.756x) the true ground-truth bbox area, caught 2026-09-21 when a
# stranded-block pattern (test-98's block 50) persisted even after the
# canvas SHAPE already closely matched ground truth, pointing at excess
# SIZE, not shape, as the remaining problem.
# UTILIZATION_TARGET itself: real ground-truth utilization in this
# dataset is ~0.97 (FloorSet-Lite's own <5% whitespace target), but that's
# the tightly-PACKED final answer, not a safe construction-time budget --
# same lesson as CANVAS_PADDING's own history (a size just enough for the
# tight optimum is not enough for a foresight-free construction process).
# AlphaChip itself reportedly targets 60-80% utilization, but that number
# doesn't transfer here: AlphaChip's RL agent places only fixed hard
# macros, with standard cells filled into the remaining whitespace
# afterward by a separate, non-RL, force-directed method -- our RL policy
# places EVERY block itself, sequentially, with no such downstream filler
# to lean on, so it needs more room during construction than AlphaChip's
# macro-only placement phase does.
# Ground-truth-undershoot alone is a weak bar (0/100 undershoot for
# U=0.5-0.8 in a validation-set scan) -- U=0.7 (area/gt_area ~1.39x) still
# collapsed the REAL test suite (99/193 failed, almost all uniform-random
# rollout stress tests) despite passing that weaker check cleanly. U=0.5
# (area/gt_area ~1.94x) passes the full suite (193/193) -- consistent with
# CANVAS_PADDING=1.4's own independently-tuned safe area (~1.90x
# ground-truth area, backing out its sqrt(total_area) formula against this
# dataset's real ~0.97 utilization), a useful cross-check that this number
# is real and not an artifact of one particular estimation method.
# 2026-09-25: raised to 0.6 together with DEFAULT_FINE_FACTOR=4 + the policy
# aspect mask (scripts/canvas_scan.py, 100 val cases, untrained checkpoint):
# U=0.6 still finishes 96% of sampled rollouts (0.65: 70%, 0.7: 24%) and cuts
# compacted greedy cost 2.16 -> 1.98 vs the old U=0.5 coarse-grid setup.
UTILIZATION_TARGET = 0.6
# Below this many valid (non-sentinel) pins, the bounding box they'd give
# is too small a sample to trust -- fall back to the old formula instead.
MIN_VALID_PINS = 2
# Anchor right/top-pinned blocks to the pin box ("die") instead of the
# canvas edge, and prefer positions inside it; the canvas beyond the die is
# overflow space only. Fine-grid path only (see _position_mask_fine).
DEFAULT_DIE_ANCHOR = False
# shape_fit: after the policy picks a cell, a free block's exact w/h (same
# area) is adjusted within FIT_BAND of the chosen shape so it ends flush with
# neighbours/gaps. lookahead: reject a cell if the next k largest unplaced
# blocks would no longer fit anywhere (see lookahead_ok). Fine grid only.
# edge_last: place edge-pinned blocks after the interior, against the interior's
# bounding box instead of the canvas edges (see _set_edge_frame). Prototype.
DEFAULT_EDGE_LAST = os.environ.get("FLOORSET_EDGE_LAST", "0") != "0"   # 09-29 retrain arm
# known_dims: the ground-truth bbox starts at (0, 0), so a right-pinned
# preplaced block's right edge is the true width (top-pinned: height). Use it
# as the canvas width (height then from area / utilization); if only the
# height is known, likewise. See _known_dims.
DEFAULT_KNOWN_DIMS = os.environ.get("FLOORSET_KNOWN_DIMS", "0") != "0"   # off until validated
# frame_util: live frame (prototype, 09-30). Placement starts inside an
# origin-anchored frame sized for this tighter utilization, nested in the
# normal canvas; the frame grows by FRAME_GROW (area) only when the current
# block fits nowhere inside it. None = off. Fine grid only; see _set_frame.
_FRAME_ENV = os.environ.get("FLOORSET_FRAME_UTIL", "")
DEFAULT_FRAME_UTIL = float(_FRAME_ENV) if _FRAME_ENV else None
FRAME_GROW = float(os.environ.get("FLOORSET_FRAME_GROW", "1.05"))
FRAME_SLIDE = os.environ.get("FLOORSET_FRAME_SLIDE", "1") != "0"   # see _slide_pinned_out
DEFAULT_SHAPE_FIT = True  # unpinned free blocks only; 09-26 scan: more finishes, same cost
DEFAULT_LOOKAHEAD = 3  # 09-26 scan: more rollouts finish at every canvas size, same cost
FIT_BAND = 1.4
# Occupancy resolution multiplier over the action grid (see __init__). 4
# removes most whole-cell rounding waste; 8 was ~4x slower for no cost gain.
DEFAULT_FINE_FACTOR = 4


def _rect_overlap(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ox = min(ax + aw, bx + bw) - max(ax, bx)
    oy = min(ay + ah, by + bh) - max(ay, by)
    return ox > 1e-9 and oy > 1e-9


def _rect_touch(a, b, eps=1e-6) -> bool:
    """True if a and b share a boundary segment of positive length."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ox = min(ax + aw, bx + bw) - max(ax, bx)
    oy = min(ay + ah, by + bh) - max(ay, by)
    return (abs(ox) <= eps and oy > eps) or (abs(oy) <= eps and ox > eps)


class GridPlacementEnv:
    def __init__(self, instance: FloorplanInstance, grid_dim: int = DEFAULT_GRID_DIM,
                 aspect_ratios: List[float] = ASPECT_RATIOS,
                 canvas_padding: Optional[Union[float, Tuple[float, float]]] = None,
                 pin_canvas: bool = False, fine_factor: Optional[int] = None,
                 utilization: Optional[float] = None, die_anchor: Optional[bool] = None,
                 shape_fit: Optional[bool] = None, lookahead: Optional[int] = None,
                 edge_last: Optional[bool] = None, known_dims: Optional[bool] = None,
                 frame_util: Optional[float] = None):
        self.instance = instance
        self.grid_dim = grid_dim
        self.aspect_ratios = aspect_ratios
        # Prototype options (defaults keep the trained behavior): pin_canvas
        # makes the canvas exactly the pin box; fine_factor > 1 tracks
        # occupancy on a (grid_dim*fine_factor)^2 grid so blocks aren't
        # rounded up to whole coarse cells -- see _position_mask_fine.
        self.pin_canvas = pin_canvas
        self.fine_factor = max(1, int(DEFAULT_FINE_FACTOR if fine_factor is None else fine_factor))
        self.utilization = UTILIZATION_TARGET if utilization is None else utilization
        self.die_anchor = DEFAULT_DIE_ANCHOR if die_anchor is None else die_anchor
        self.shape_fit = DEFAULT_SHAPE_FIT if shape_fit is None else shape_fit
        self.edge_last = DEFAULT_EDGE_LAST if edge_last is None else edge_last
        self.known_dims = DEFAULT_KNOWN_DIMS if known_dims is None else known_dims
        self.lookahead = DEFAULT_LOOKAHEAD if lookahead is None else int(lookahead)
        self.frame_util = DEFAULT_FRAME_UTIL if frame_util is None else frame_util
        if self.frame_util is not None and (self.frame_util <= self.utilization
                                            or self.die_anchor or self.edge_last):
            self.frame_util = None   # no tighter than the canvas, or anchors owned elsewhere
        if self.fine_factor == 1:
            self.shape_fit, self.lookahead, self.edge_last = False, 0, False
            self.frame_util = None
        # An explicit override skips the pins-based sizing below entirely
        # (see reset()) and forces the old sqrt(total_area)*padding formula
        # -- used by rl/finetune.py's greedy_fallback_positions to widen the
        # working canvas on retry when even the default turns out too
        # tight. Accepts a single float (both dimensions the same) or a
        # (width_padding, height_padding) pair.
        self._explicit_padding = canvas_padding is not None
        if canvas_padding is None:
            self.width_padding = self.height_padding = CANVAS_PADDING
        elif isinstance(canvas_padding, tuple):
            self.width_padding, self.height_padding = canvas_padding
        else:
            self.width_padding = self.height_padding = canvas_padding
        self.plan = compute_order(instance.constraints, instance.area_targets,
                                   instance.b2b_connectivity)
        self.reset()

    @staticmethod
    def _known_dims(inst: FloorplanInstance) -> Tuple[Optional[float], Optional[float]]:
        """(width, height) revealed by right/top-pinned preplaced blocks, or None."""
        c = inst.constraints
        if c.dim() < 2 or c.shape[1] < 5:
            return None, None
        pre = c[:, 1] != 0
        code = c[:, 4].long()
        tp = inst.target_positions
        right = [float(tp[i, 0] + tp[i, 2]) for i in range(inst.block_count) if pre[i] and code[i] & 2]
        top = [float(tp[i, 1] + tp[i, 3]) for i in range(inst.block_count) if pre[i] and code[i] & 4]
        return (max(right) if right else None), (max(top) if top else None)

    def _estimate_canvas_size(self, inst: FloorplanInstance) -> Tuple[float, float]:
        if not self._explicit_padding and self.known_dims:
            kw, kh = self._known_dims(inst)
            area = max(float(inst.area_targets.clamp(min=0).sum().item()), 1.0) / self.utilization
            # Width wins when both are known: both exact would be ~97% full.
            if kw and kw > 0:
                return kw, area / kw
            if kh and kh > 0:
                return area / kh, kh
        return self._estimate_canvas_size_pins(inst)

    def _estimate_canvas_size_pins(self, inst: FloorplanInstance) -> Tuple[float, float]:
        """The working canvas's width/height, before any block is placed.

        Primary method: SHAPE from the instance's own pins (`pins_pos`,
        given in the problem input -- real I/O pins sit at or near the
        physical die edge, so their bounding-box ratio closely tracks the
        true floorplan's aspect ratio), AREA from `area_targets.sum()` (the
        exact total block-area budget already in the problem input)
        divided by UTILIZATION_TARGET. This needs no estimation, training,
        or scouting for either quantity -- unlike every other approach
        tried this session (fixed padding, an online-learned EMA, a
        policy-driven scout rollout, a simulated-annealing scout), both are
        exact data already in hand before placement starts. See
        UTILIZATION_TARGET for the validation-set numbers behind this
        design and its target value.

        Skipped (falls through to the old sqrt(total_area)*padding
        formula) when an explicit `canvas_padding` override was given
        (see __init__ -- used by greedy_fallback_positions's widen-on-
        failure retry, which needs to force a specific size regardless of
        pins) or when the instance has too few valid pins to trust for
        shape."""
        if not self._explicit_padding:
            pins = inst.pins_pos
            if pins is not None and pins.numel() > 0:
                valid = pins[(pins[:, 0] != -1) | (pins[:, 1] != -1)]
                if self.pin_canvas and valid.shape[0] >= MIN_VALID_PINS:
                    # Die = origin-anchored box covering the pins' max x/y.
                    die_w = float(valid[:, 0].max().item())
                    die_h = float(valid[:, 1].max().item())
                    if die_w > 0 and die_h > 0:
                        return die_w, die_h
                if valid.shape[0] >= MIN_VALID_PINS:
                    pin_w = float((valid[:, 0].max() - valid[:, 0].min()).item())
                    pin_h = float((valid[:, 1].max() - valid[:, 1].min()).item())
                    if pin_w > 0 and pin_h > 0:
                        total_block_area = float(inst.area_targets.clamp(min=0).sum().item())
                        canvas_area = max(total_block_area, 1.0) / self.utilization
                        ratio = pin_w / pin_h
                        est_width = math.sqrt(canvas_area * ratio)
                        est_height = canvas_area / est_width
                        return est_width, est_height

        total_area = float(inst.area_targets.clamp(min=0).sum().item())
        scale = math.sqrt(max(total_area, 1.0))
        return scale * self.width_padding, scale * self.height_padding

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

        est_width, est_height = self._estimate_canvas_size(inst)

        self.x_min = min(xs0)
        self.y_min = min(ys0)
        self.x_max = max([self.x_min + est_width] + xs1)
        self.y_max = max([self.y_min + est_height] + ys1)

        self.cell_w = (self.x_max - self.x_min) / self.grid_dim
        self.cell_h = (self.y_max - self.y_min) / self.grid_dim

        # Edges right/top-pinned blocks anchor to: the canvas edge, or with
        # die_anchor the pin box's max x/y (clamped inside the canvas, and
        # never left of a preplaced block's far edge).
        self.anchor_x_max, self.anchor_y_max = self.x_max, self.y_max
        self.anchor_x_min, self.anchor_y_min = self.x_min, self.y_min
        self._frame_set = False
        if self.die_anchor and self.fine_factor > 1:
            pins = inst.pins_pos
            if pins is not None and pins.numel() > 0:
                valid = pins[(pins[:, 0] != -1) | (pins[:, 1] != -1)]
                if valid.shape[0] >= MIN_VALID_PINS:
                    die_w = float(valid[:, 0].max().item())
                    die_h = float(valid[:, 1].max().item())
                    if die_w > self.x_min and die_h > self.y_min:
                        self.anchor_x_max = min(self.x_max, max([die_w] + xs1))
                        self.anchor_y_max = min(self.y_max, max([die_h] + ys1))

        self.occupancy = torch.zeros(self.grid_dim, self.grid_dim)
        if self.fine_factor > 1:
            self.fine_dim = self.grid_dim * self.fine_factor
            self.fine_cell_w = self.cell_w / self.fine_factor
            self.fine_cell_h = self.cell_h / self.fine_factor
            self.fine_occ = torch.zeros(self.fine_dim, self.fine_dim)
            self._sat_cache = None
            self.fine_cluster: Dict[int, torch.Tensor] = {}
            self._fine_choice: Optional[torch.Tensor] = None
        self.frame_growths = 0
        if self.frame_util is not None:
            # Real block occupancy; fine_occ = this OR the cells outside the frame.
            self._real_occ = torch.zeros(self.fine_dim, self.fine_dim)
            s = math.sqrt(self.utilization / self.frame_util)
            self._set_frame(max([self.x_min + (self.x_max - self.x_min) * s] + xs1),
                            max([self.y_min + (self.y_max - self.y_min) * s] + ys1))
        self.positions: List[Optional[Tuple[float, float, float, float]]] = [None] * n
        self.mib_shape: Dict[int, Tuple[float, float]] = {}
        self.cluster_cells: Dict[int, set] = defaultdict(set)
        self.boundary_fallback_count = 0
        # Analytic placer targets for the policy's target-map channel
        # (networks.TARGET_MAP experiment, 10-04): one global placement per
        # episode; block centres in canvas coordinates.
        self._gp_centres = None
        from .networks import TARGET_MAP
        if TARGET_MAP:
            from .analytic import global_place
            gp = global_place(self.instance, util=0.85, iters=150)
            self._gp_centres = [(x + w / 2, y + h / 2) for x, y, w, h in gp]
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
        if self.edge_last:
            self._pending = self._edge_last_order(self._pending)
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

    def _fine_rect_to_cells(self, x, y, w, h):
        """_rect_to_cells on the fine grid (same epsilon reasoning)."""
        d = self.fine_dim
        fx0 = int(math.floor((x - self.x_min) / self.fine_cell_w + 1e-9))
        fy0 = int(math.floor((y - self.y_min) / self.fine_cell_h + 1e-9))
        fx1 = int(math.ceil((x + w - self.x_min) / self.fine_cell_w - 1e-9))
        fy1 = int(math.ceil((y + h - self.y_min) / self.fine_cell_h - 1e-9))
        fx0, fy0 = max(0, min(fx0, d)), max(0, min(fy0, d))
        return fx0, fy0, max(fx0, min(fx1, d)), max(fy0, min(fy1, d))

    def _mark_occupied(self, x, y, w, h):
        if self.fine_factor > 1:
            # Observation = fraction of each coarse cell that's occupied.
            fx0, fy0, fx1, fy1 = self._fine_rect_to_cells(x, y, w, h)
            self.fine_occ[fy0:fy1, fx0:fx1] = 1.0
            if self.frame_util is not None:
                self._real_occ[fy0:fy1, fx0:fx1] = 1.0
            self._sat_cache = None
            self.occupancy = F.avg_pool2d(self.fine_occ[None, None], self.fine_factor)[0, 0]
            return
        gx0, gy0, gx1, gy1 = self._rect_to_cells(x, y, w, h)
        if gx1 > gx0 and gy1 > gy0:
            self.occupancy[gy0:gy1, gx0:gx1] = 1.0

    def _mark_cluster(self, cid, x, y, w, h):
        if self.fine_factor > 1:
            fx0, fy0, fx1, fy1 = self._fine_rect_to_cells(x, y, w, h)
            if cid not in self.fine_cluster:
                self.fine_cluster[cid] = torch.zeros(self.fine_dim, self.fine_dim)
            self.fine_cluster[cid][fy0:fy1, fx0:fx1] = 1.0
        gx0, gy0, gx1, gy1 = self._rect_to_cells(x, y, w, h)
        for gy in range(gy0, gy1):
            for gx in range(gx0, gx1):
                self.cluster_cells[cid].add((gy, gx))

    # -- live frame (frame_util) --------------------------------------------
    def _set_frame(self, x1, y1):
        """Frame = [x_min, x1] x [y_min, y1], snapped outward to whole fine
        cells so right/top anchors sit on a fine grid line."""
        cols = min(self.fine_dim, max(1, math.ceil((x1 - self.x_min) / self.fine_cell_w - 1e-9)))
        rows = min(self.fine_dim, max(1, math.ceil((y1 - self.y_min) / self.fine_cell_h - 1e-9)))
        self._set_frame_cells(rows, cols)

    def _set_frame_cells(self, rows, cols):
        d = self.fine_dim
        self._frame_cells = (rows, cols)
        # Right/top-pinned blocks anchor to the frame edge.
        self.anchor_x_max = self.x_max if cols == d else self.x_min + cols * self.fine_cell_w
        self.anchor_y_max = self.y_max if rows == d else self.y_min + rows * self.fine_cell_h
        outside = torch.ones(d, d)
        outside[:rows, :cols] = 0.0
        self.fine_occ = torch.maximum(self._real_occ, outside)
        self._sat_cache = None
        self.occupancy = F.avg_pool2d(self.fine_occ[None, None], self.fine_factor)[0, 0]

    def _grow_frame(self) -> bool:
        """Grow the frame's area by FRAME_GROW (at least one fine cell per
        side), capped at the canvas. False when it already is the canvas."""
        d = self.fine_dim
        rows, cols = self._frame_cells
        if rows == d and cols == d:
            return False
        g = math.sqrt(FRAME_GROW)
        old_x1, old_y1 = self.anchor_x_max, self.anchor_y_max
        self._set_frame_cells(min(d, max(rows + 1, math.ceil(rows * g))),
                              min(d, max(cols + 1, math.ceil(cols * g))))
        self.frame_growths += 1
        if FRAME_SLIDE:
            self._slide_pinned_out(old_x1, old_y1)
        return True

    def _slide_pinned_out(self, old_x1, old_y1):
        """Right/top-pinned blocks flush with the old frame edge follow it to
        the new edge when the swept path is clear, so growth doesn't strand
        them. Occupancy and cluster maps are rebuilt if anything moved."""
        c = self.instance.constraints
        if c.dim() < 2 or c.shape[1] < 5:
            return
        pre = c[:, 1] != 0
        codes = c[:, 4].long()
        eps = 1e-6
        moved = False
        for i, p in enumerate(self.positions):
            if p is None or bool(pre[i]):
                continue
            code = int(codes[i])
            x, y, w, h = p
            nx, ny = x, y
            if code & 2 and abs(x + w - old_x1) < eps:
                nx = self.anchor_x_max - w
            if code & 4 and abs(y + h - old_y1) < eps:
                ny = self.anchor_y_max - h
            if nx == x and ny == y:
                continue
            # Swept rectangle from the old to the new position.
            swept = (x, y, nx + w - x, ny + h - y)
            if any(j != i and q is not None and _rect_overlap(swept, q)
                   for j, q in enumerate(self.positions)):
                continue
            # A clustered block must still touch a placed clustermate.
            cid = int(c[i, 3]) if c.shape[1] > 3 else 0
            mates = [q for j, q in enumerate(self.positions)
                     if cid and j != i and q is not None and int(c[j, 3]) == cid]
            if mates and not any(_rect_touch((nx, ny, w, h), q) for q in mates):
                continue
            self.positions[i] = (nx, ny, w, h)
            moved = True
        if moved:
            self._rebuild_maps()

    def _rebuild_maps(self):
        """Recompute real occupancy and cluster maps from self.positions."""
        c = self.instance.constraints
        cluster_col = c[:, 3].long() if c.dim() > 1 and c.shape[1] > 3 else None
        self._real_occ.zero_()
        self.fine_cluster = {}
        self.cluster_cells = defaultdict(set)
        for i, p in enumerate(self.positions):
            if p is None:
                continue
            fx0, fy0, fx1, fy1 = self._fine_rect_to_cells(*p)
            self._real_occ[fy0:fy1, fx0:fx1] = 1.0
            cid = int(cluster_col[i]) if cluster_col is not None else 0
            if cid:
                self._mark_cluster(cid, *p)
        self._set_frame_cells(*self._frame_cells)
        if self._bbox is not None:
            for p in self.positions:
                if p is not None:
                    self._grow_bbox(*p)

    def _grow_frame_until(self, fits) -> None:
        """Grow the frame until `fits()` or it reaches the canvas."""
        if self.frame_util is None:
            return
        while not fits() and self._grow_frame():
            pass

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

    def aspect_mask(self) -> torch.Tensor:
        """Bool [len(aspect_ratios)]: True where the current free block, at
        that aspect ratio, has at least one legal position right now. Lets
        a caller avoid picking a shape that can't fit anywhere (all False
        means the block can't be placed in any shape). With a live frame,
        grows it first until at least one shape fits."""
        if self.frame_util is not None:
            out = None

            def fits():
                nonlocal out
                out = self._aspect_mask_now()
                return bool(out.any())
            self._grow_frame_until(fits)
            return out
        return self._aspect_mask_now()

    def _aspect_mask_now(self) -> torch.Tensor:
        area = self.current_area()
        out = torch.zeros(len(self.aspect_ratios), dtype=torch.bool)
        for a, r in enumerate(self.aspect_ratios):
            w, h = math.sqrt(area * r), math.sqrt(area / r)
            if self.fine_factor > 1:
                fw = max(1, math.ceil(w / self.fine_cell_w - 1e-9))
                fh = max(1, math.ceil(h / self.fine_cell_h - 1e-9))
                if fw <= self.fine_dim and fh <= self.fine_dim:
                    out[a] = bool(self._free_map(fw, fh).any())
            else:
                cells_w, cells_h = self._footprint_cells(w, h)
                window = F.conv2d(self.occupancy[None, None], torch.ones(1, 1, cells_h, cells_w))[0, 0]
                out[a] = bool((window == 0).any())
        return out

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

        if cid and self.cluster_cells.get(cid):
            touch = self._try_cluster_touch(cid, w, h, code=step.boundary_code)
            if touch is not None:
                x, y = touch
                self._commit(x, y, w, h)
                return None

        if self.fine_factor > 1:
            return self._position_mask_fine(w, h, step)

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

    # -- fine-grid legality (fine_factor > 1) ------------------------------
    @staticmethod
    def _sat(grid):
        """Summed-area table of a 0/1 grid, zero-padded on the top/left."""
        return F.pad(grid.to(torch.int32).cumsum(0).cumsum(1), (1, 0, 1, 0))

    def _window_sums(self, grid, fw, fh, sat=None):
        """Sum of `grid` over every fw x fh window, indexed by top-left
        fine cell -- shape [dim-fh+1, dim-fw+1], via a summed-area table.
        Pass a precomputed `sat` to skip rebuilding it."""
        if sat is None:
            sat = self._sat(grid)
        d = sat.shape[0] - 1
        return (sat[fh:, fw:] - sat[:d - fh + 1, fw:]
                - sat[fh:, :d - fw + 1] + sat[:d - fh + 1, :d - fw + 1])

    def _occ_sats(self):
        """(SAT of fine_occ, SAT of its 3x3 dilation), cached until the
        next _mark_occupied -- aspect_mask and position_mask reuse them."""
        if self._sat_cache is None:
            dil = F.max_pool2d(self.fine_occ[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
            self._sat_cache = (self._sat(self.fine_occ), self._sat(dil))
        return self._sat_cache

    def _free_map(self, fw, fh):
        return self._window_sums(self.fine_occ, fw, fh, sat=self._occ_sats()[0]) == 0

    def _touch_map(self, grid, fw, fh, sat=None):
        """True where a fw x fh footprint would touch or overlap `grid`."""
        if sat is None:
            dil = F.max_pool2d(grid[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
            sat = self._sat(dil)
        return self._window_sums(grid, fw, fh, sat=sat) > 0

    def _to_coarse(self, fine_map):
        """Pads a fine top-left map to [fine_dim]^2 and marks each coarse
        cell containing at least one legal fine position."""
        d, k, g = self.fine_dim, self.fine_factor, self.grid_dim
        full = torch.zeros(d, d, dtype=torch.bool)
        full[:fine_map.shape[0], :fine_map.shape[1]] = fine_map
        self._fine_choice = full
        return full.view(g, k, g, k).any(dim=3).any(dim=1)

    def _position_mask_fine(self, w, h, step):
        """position_mask() for fine_factor > 1. Legality is decided on the
        fine grid; the returned mask is over coarse cells (the action
        space), and place() picks the legal fine position nearest the
        chosen cell's corner. Same preference order as the coarse path:
        boundary edge, then cluster adjacency, then compactness."""
        d = self.fine_dim
        fw = max(1, math.ceil(w / self.fine_cell_w - 1e-9))
        fh = max(1, math.ceil(h / self.fine_cell_h - 1e-9))
        cells_w, cells_h = self._footprint_cells(w, h)
        if fw > d or fh > d:
            raise RuntimeError(
                f"No free position for block {step.block_idx} (w={w}, h={h}); "
                f"block larger than canvas.")
        self._grow_frame_until(lambda: bool(self._free_map(fw, fh).any()))
        free = self._free_map(fw, fh)
        if not bool(free.any()):
            raise RuntimeError(
                f"No free position for block {step.block_idx} (w={w}, h={h}); "
                f"canvas too small -- increase CANVAS_PADDING or grid_dim.")

        code = step.boundary_code
        if code:
            if self.edge_last and not self._frame_set:
                self._set_edge_frame()
            pinned = self._pinned_window(code, w, h, fw, fh)
            if not bool(pinned.any()) and self.edge_last and self._frame_set:
                # No room flush inside the frame: widen the frame by this block
                # on its pinned side(s) and sit just outside the old edge.
                saved = (self.anchor_x_min, self.anchor_x_max, self.anchor_y_min, self.anchor_y_max)
                if code & 1:
                    self.anchor_x_min = max(self.x_min, self.anchor_x_min - w)
                if code & 2:
                    self.anchor_x_max = min(self.x_max, self.anchor_x_max + w)
                if code & 8:
                    self.anchor_y_min = max(self.y_min, self.anchor_y_min - h)
                if code & 4:
                    self.anchor_y_max = min(self.y_max, self.anchor_y_max + h)
                pinned = self._pinned_window(code, w, h, fw, fh)
                if not bool(pinned.any()):
                    self.anchor_x_min, self.anchor_x_max, self.anchor_y_min, self.anchor_y_max = saved
            if bool(pinned.any()):
                self._boundary_pin_ok = True
                return self._to_coarse(self._prefer_inside(pinned, fw, fh)), cells_w, cells_h
            self.boundary_fallback_count += 1
            return self._to_coarse(self._prefer_inside(free, fw, fh)), cells_w, cells_h

        cid = step.cluster_id
        if cid:
            cgrid = self.fine_cluster.get(cid)
            if cgrid is not None:
                adj = self._touch_map(cgrid, fw, fh) & free
                if bool(adj.any()):
                    return self._to_coarse(self._prefer_inside(adj, fw, fh)), cells_w, cells_h
                self.grouping_fallback_count += 1

        compact = self._touch_map(self.fine_occ, fw, fh, sat=self._occ_sats()[1]) & free
        if self.die_anchor:
            # Inside the die beats touching: the slack is overflow only.
            for cand in (compact, free):
                inner = cand & self._inside_die(cand.shape, fw, fh)
                if bool(inner.any()):
                    return self._to_coarse(inner), cells_w, cells_h
        if bool(compact.any()):
            return self._to_coarse(compact), cells_w, cells_h
        return self._to_coarse(free), cells_w, cells_h

    def _pinned_window(self, code, w, h, fw, fh):
        """Top-left fine cells where the block can sit flush with the anchor
        edge(s) its code requires. The checked window runs from the fine
        cell holding the snapped coordinate to the cell holding the anchor,
        so snapping flush in place() stays inside verified free space."""
        d = self.fine_dim
        bw, bh, fx_pin, fy_pin = fw, fh, None, None
        if code & 1:
            fx_pin = int(math.floor((self.anchor_x_min - self.x_min) / self.fine_cell_w + 1e-9))
            bw = max(fw, min(d, math.ceil((self.anchor_x_min + w - self.x_min) / self.fine_cell_w - 1e-9)) - fx_pin)
        if code & 2:
            fx_pin = max(0, int(math.floor((self.anchor_x_max - w - self.x_min) / self.fine_cell_w + 1e-9)))
            bw = max(fw, self._die_fine_cols() - fx_pin)
        if code & 8:
            fy_pin = int(math.floor((self.anchor_y_min - self.y_min) / self.fine_cell_h + 1e-9))
            bh = max(fh, min(d, math.ceil((self.anchor_y_min + h - self.y_min) / self.fine_cell_h - 1e-9)) - fy_pin)
        if code & 4:
            fy_pin = max(0, int(math.floor((self.anchor_y_max - h - self.y_min) / self.fine_cell_h + 1e-9)))
            bh = max(fh, self._die_fine_rows() - fy_pin)
        if bw > d or bh > d:
            return torch.zeros(1, 1, dtype=torch.bool)
        pinned = self._free_map(bw, bh)
        keep = torch.zeros_like(pinned)
        # A block wider/taller than its anchor span can't sit flush.
        too_big = ((code & 2 and self.anchor_x_max - w < self.x_min - 1e-9)
                   or (code & 4 and self.anchor_y_max - h < self.y_min - 1e-9)
                   or (code & 1 and self.anchor_x_min + w > self.x_max + 1e-9)
                   or (code & 8 and self.anchor_y_min + h > self.y_max + 1e-9))
        if too_big:
            return keep
        if fx_pin is not None and fy_pin is not None:
            if fy_pin < keep.shape[0] and fx_pin < keep.shape[1]:
                keep[fy_pin, fx_pin] = True
        elif fx_pin is not None:
            if fx_pin < keep.shape[1]:
                keep[:, fx_pin] = True
        elif fy_pin is not None:
            if fy_pin < keep.shape[0]:
                keep[fy_pin, :] = True
        return pinned & keep

    def _set_edge_frame(self):
        """edge_last: freeze the anchor edges at the bounding box of what is
        placed so far (the interior), instead of the canvas edges."""
        placed = [p for p in self.positions if p is not None]
        if placed:
            self.anchor_x_min = min(p[0] for p in placed)
            self.anchor_y_min = min(p[1] for p in placed)
            self.anchor_x_max = max(p[0] + p[2] for p in placed)
            self.anchor_y_max = max(p[1] + p[3] for p in placed)
        self._frame_set = True

    @staticmethod
    def _edge_last_order(pending):
        """Stable reorder: units with no boundary pin first, then the rest.
        A unit is an MIB leader plus its followers (kept contiguous so a
        follower always comes after the shape it copies); it counts as
        pinned if any member is."""
        units = []
        for st in pending:
            if st.role == 'mib_follower' and units:
                units[-1].append(st)
            else:
                units.append([st])
        inner = [u for u in units if not any(st.boundary_code for st in u)]
        edge = [u for u in units if any(st.boundary_code for st in u)]
        return [st for u in inner + edge for st in u]

    def _die_fine_cols(self):
        return min(self.fine_dim, math.ceil((self.anchor_x_max - self.x_min) / self.fine_cell_w - 1e-9))

    def _die_fine_rows(self):
        return min(self.fine_dim, math.ceil((self.anchor_y_max - self.y_min) / self.fine_cell_h - 1e-9))

    def _inside_die(self, shape, fw, fh):
        """Bool map (top-left fine cells, `shape`) of footprints lying
        entirely within the die."""
        inside = torch.zeros(shape, dtype=torch.bool)
        # floor: only fine cells lying wholly inside the die count.
        die_r = int(math.floor((self.anchor_y_max - self.y_min) / self.fine_cell_h + 1e-9))
        die_c = int(math.floor((self.anchor_x_max - self.x_min) / self.fine_cell_w + 1e-9))
        rows, cols = die_r - fh + 1, die_c - fw + 1
        if rows > 0 and cols > 0:
            inside[:rows, :cols] = True
        return inside

    def _prefer_inside(self, cand, fw, fh):
        """`cand` restricted to inside-die footprints when die_anchor is on
        and any exist; otherwise `cand` unchanged."""
        if not self.die_anchor:
            return cand
        inner = cand & self._inside_die(cand.shape, fw, fh)
        return inner if bool(inner.any()) else cand

    def _fine_position_in_cell(self, gy, gx):
        """Legal fine top-left (fy, fx) inside coarse cell (gy, gx) nearest
        its lower-left corner, from the last _position_mask_fine call."""
        k = self.fine_factor
        window = self._fine_choice[gy * k:(gy + 1) * k, gx * k:(gx + 1) * k]
        idx = window.nonzero(as_tuple=False)
        assert idx.numel() > 0, f"coarse cell ({gy}, {gx}) not in the position mask"
        best = min(idx.tolist(), key=lambda t: (t[0] + t[1], t[0]))
        return gy * k + best[0], gx * k + best[1]

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

    def target_map(self, block_idx: int, w: float, h: float) -> torch.Tensor:
        """[grid_dim, grid_dim]: a Gaussian bump (sigma = half the block size)
        at the top-left cell that would centre this w x h block where the
        episode's analytic global placement put it. Zeros without one."""
        g = self.grid_dim
        if self._gp_centres is None:
            return torch.zeros(g, g)
        cx, cy = self._gp_centres[block_idx]
        col = (cx - w / 2 - self.x_min) / self.cell_w
        row = (cy - h / 2 - self.y_min) / self.cell_h
        sx = max(1.0, w / self.cell_w / 2)
        sy = max(1.0, h / self.cell_h / 2)
        ys = torch.arange(g, dtype=torch.float32).view(-1, 1)
        xs = torch.arange(g, dtype=torch.float32).view(1, -1)
        return torch.exp(-((xs - col) ** 2 / (2 * sx * sx) + (ys - row) ** 2 / (2 * sy * sy)))

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

    def _x_bit_satisfied(self, x, w, code) -> bool:
        """True if x-coordinate `x` satisfies code's left(1)/right(2) bit,
        if any (assumes at most one is set -- see _try_cluster_touch)."""
        eps = 1e-6
        if code & 1 and abs(x - self.anchor_x_min) > eps:
            return False
        if code & 2 and abs(x + w - self.anchor_x_max) > eps:
            return False
        return True

    def _y_bit_satisfied(self, y, h, code) -> bool:
        """Mirror of _x_bit_satisfied for top(4)/bottom(8)."""
        eps = 1e-6
        if code & 4 and abs(y + h - self.anchor_y_max) > eps:
            return False
        if code & 8 and abs(y - self.anchor_y_min) > eps:
            return False
        return True

    def _forced_x(self, w, code) -> Optional[float]:
        """The x-coordinate code's left/right bit forces, or None if
        neither bit is set (caller should search x freely instead)."""
        if code & 1:
            return self.anchor_x_min
        if code & 2:
            return self.anchor_x_max - w
        return None

    def _forced_y(self, h, code) -> Optional[float]:
        """Mirror of _forced_x for top/bottom."""
        if code & 4:
            return self.anchor_y_max - h
        if code & 8:
            return self.anchor_y_min
        return None

    def _try_cluster_touch(self, cid, w, h, code=0):
        """Touch search, trying positions inside the die first when
        die_anchor is on (see _try_cluster_touch_within)."""
        if self.die_anchor:
            hit = self._try_cluster_touch_within(cid, w, h, code, self.anchor_x_max, self.anchor_y_max)
            if hit is not None:
                return hit
        if self.frame_util is not None:
            return self._try_cluster_touch_within(cid, w, h, code, self.anchor_x_max, self.anchor_y_max)
        return self._try_cluster_touch_within(cid, w, h, code, self.x_max, self.y_max)

    def _try_cluster_touch_within(self, cid, w, h, code, lim_x, lim_y):
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
        far more often, without ever weakening the overlap check itself.

        `code`, if nonzero (the block is ALSO boundary-pinned -- see
        position_mask), additionally requires the touch to land on the
        block's required edge(s). The axis the touch search already fixes
        (e.g. cx = nx+nw when touching a neighbor's right side) is just
        checked against that axis's bit; the FREE axis (cy in that same
        example) is not searched at all when code constrains it -- it's
        computed directly from the exact required coordinate (e.g.
        self.y_max - h), since that's almost never a value _grid_values'
        neighbor-relative stepping would happen to land on by chance (a
        bug in an earlier version of this fix: filtering the generic
        enumeration instead of solving for the exact required value meant
        it essentially never found a real candidate, even for same-edge
        pairs). boundary_code encodes at most one bit per axis (corners
        are one x-bit + one y-bit, e.g. 5 = top-left = 4+1), so each axis
        has a single well-defined forced value when constrained."""
        n = self.instance.block_count
        ncols = self.instance.constraints.shape[1]
        cluster_col = self.instance.constraints[:, 3] if ncols > 3 else torch.zeros(n)
        neighbors = [self.positions[j] for j in range(n)
                     if self.positions[j] is not None and int(cluster_col[j]) == cid]
        for (nx, ny, nw, nh) in neighbors:
            # Touch on the neighbor's left/right side: cx is fixed by the
            # touch itself, cy is otherwise free.
            y_lo, y_hi = max(self.y_min, ny - h), min(lim_y - h, ny + nh)
            for cx in (nx + nw, nx - w):
                if cx < self.x_min - 1e-9 or cx + w > lim_x + 1e-9:
                    continue
                if code and not self._x_bit_satisfied(cx, w, code):
                    continue
                forced_cy = self._forced_y(h, code) if code else None
                if forced_cy is not None:
                    cy_candidates = [forced_cy] if y_lo - 1e-9 <= forced_cy <= y_hi + 1e-9 else []
                else:
                    cy_candidates = self._grid_values(y_lo, y_hi, self.cell_h, ny)
                for cy in cy_candidates:
                    candidate = (cx, cy, w, h)
                    if not any(p is not None and _rect_overlap(candidate, p) for p in self.positions):
                        return cx, cy
            # Touch on the neighbor's top/bottom side: cy is fixed, cx free.
            x_lo, x_hi = max(self.x_min, nx - w), min(lim_x - w, nx + nw)
            for cy in (ny + nh, ny - h):
                if cy < self.y_min - 1e-9 or cy + h > lim_y + 1e-9:
                    continue
                if code and not self._y_bit_satisfied(cy, h, code):
                    continue
                forced_cx = self._forced_x(w, code) if code else None
                if forced_cx is not None:
                    cx_candidates = [forced_cx] if x_lo - 1e-9 <= forced_cx <= x_hi + 1e-9 else []
                else:
                    cx_candidates = self._grid_values(x_lo, x_hi, self.cell_w, nx)
                for cx in cx_candidates:
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
        self._commit(*self._resolve_placement(gy, gx))

    def _resolve_placement(self, gy: int, gx: int):
        """The exact (x, y, w, h) place(gy, gx) would commit, without
        committing it."""
        step = self.current_step()
        if step.role == 'free':
            w, h = self._pending_shape
        else:
            w, h = self.current_shape()

        if self.fine_factor > 1:
            fy, fx = self._fine_position_in_cell(gy, gx)
            x = self.x_min + fx * self.fine_cell_w
            y = self.y_min + fy * self.fine_cell_h
        else:
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
                x = self.anchor_x_min
            if code & 2:
                x = self.anchor_x_max - w
            if code & 4:
                y = self.anchor_y_max - h
            if code & 8:
                y = self.anchor_y_min

        if self.shape_fit and step.role == 'free' and not step.boundary_code:
            x, y, w, h = self._fit_shape(x, y, w, h, code if self._boundary_pin_ok else 0)
        return x, y, w, h

    # -- shape fitting and lookahead (fine grid only) ----------------------
    def _fit_shape(self, x, y, w, h, code):
        """Same-area reshape of a free block, within FIT_BAND of (w, h). Ranked
        by: least growth of the placed layout's bbox, then most edge contact
        with placed blocks and the canvas's left/bottom edges (closes gaps,
        aligns edges), then closeness to the chosen shape. Ranking contact
        first stretched blocks outward and grew the bbox (09-26 scan). The
        corner on its pinned edges stays put (right/top pins keep their
        right/top edge). Candidates are checked exactly against every placed
        rect."""
        area = w * h
        right_fixed, top_fixed = bool(code & 2), bool(code & 4)
        x_ref = x + w if right_fixed else x
        y_ref = y + h if top_fixed else y
        placed = [p for p in self.positions if p is not None]
        xs = {self.x_min, self.x_max, self.anchor_x_max}
        ys = {self.y_min, self.y_max, self.anchor_y_max}
        for (px, py, pw, ph) in placed:
            xs.update((px, px + pw))
            ys.update((py, py + ph))
        widths = {abs(e - x_ref) for e in xs}
        heights = {abs(e - y_ref) for e in ys}
        cands = {(w, h)}
        cands.update((cw, area / cw) for cw in widths if w / FIT_BAND <= cw <= w * FIT_BAND)
        cands.update((area / ch, ch) for ch in heights if h / FIT_BAND <= ch <= h * FIT_BAND)

        if placed:
            bx0 = min(p[0] for p in placed)
            by0 = min(p[1] for p in placed)
            bx1 = max(p[0] + p[2] for p in placed)
            by1 = max(p[1] + p[3] for p in placed)
        else:
            bx0 = by0 = bx1 = by1 = None

        def bbox_growth(r):
            rx, ry, rw, rh = r
            if bx0 is None:
                return rw * rh
            return ((max(bx1, rx + rw) - min(bx0, rx)) * (max(by1, ry + rh) - min(by0, ry))
                    - (bx1 - bx0) * (by1 - by0))

        best, best_key = (x, y, w, h), None
        # A live frame bounds the reshape too (its outside is not real space).
        lim_x, lim_y = ((self.anchor_x_max, self.anchor_y_max) if self.frame_util is not None
                        else (self.x_max, self.y_max))
        for cw, ch in cands:
            cx = x_ref - cw if right_fixed else x_ref
            cy = y_ref - ch if top_fixed else y_ref
            if (cx < self.x_min - 1e-9 or cy < self.y_min - 1e-9
                    or cx + cw > lim_x + 1e-9 or cy + ch > lim_y + 1e-9):
                continue
            rect = (cx, cy, cw, ch)
            if any(_rect_overlap(rect, p) for p in placed):
                continue
            key = (-round(bbox_growth(rect), 6), round(self._contact(rect, placed), 9),
                   -abs(math.log(cw / w)))
            if best_key is None or key > best_key:
                best, best_key = rect, key
        return best

    def _contact(self, rect, placed):
        """Length of rect's boundary shared with placed rects or the
        canvas's left/bottom edges."""
        x, y, w, h = rect
        eps = 1e-6
        total = (h if abs(x - self.x_min) < eps else 0.0) + (w if abs(y - self.y_min) < eps else 0.0)
        for (px, py, pw, ph) in placed:
            if abs(px + pw - x) < eps or abs(x + w - px) < eps:
                total += max(0.0, min(y + h, py + ph) - max(y, py))
            if abs(py + ph - y) < eps or abs(y + h - py) < eps:
                total += max(0.0, min(x + w, px + pw) - max(x, px))
        return total

    def _upcoming_shapes(self):
        """Shape options for the `lookahead` largest unplaced blocks after
        the current one: a single (w, h) for fixed blocks and followers
        whose leader is placed, every aspect bucket otherwise."""
        key = self._cursor
        if getattr(self, '_upcoming_key', None) == key:
            return self._upcoming_cache
        rest = []
        for st in self._pending[self._cursor + 1:]:
            i = st.block_idx
            area = float(self.instance.area_targets[i])
            if st.role == 'fixed':
                tp = self.instance.target_positions[i]
                opts = [(float(tp[2]), float(tp[3]))]
                area = opts[0][0] * opts[0][1]
            elif st.role == 'mib_follower' and st.mib_leader in self.mib_shape:
                lw, lh = self.mib_shape[st.mib_leader]
                opts = [(lw, lh)]
            else:
                opts = [(math.sqrt(area * r), math.sqrt(area / r)) for r in self.aspect_ratios]
            if area > 0:
                rest.append((area, opts))
        rest.sort(key=lambda t: -t[0])
        self._upcoming_key, self._upcoming_cache = key, [o for _, o in rest[:self.lookahead]]
        return self._upcoming_cache

    def step_features(self) -> torch.Tensor:
        """[NUM_STEP_FEATS] summary of what is still to come, for the policy
        and value heads (rl/networks.py): (current + future block area) /
        free canvas area; largest future block's area / free area; its
        size / canvas width and / canvas height; share of its shape options
        that still fit somewhere now (fine grid; 1.0 on the coarse path);
        fraction of steps left. Cached per step."""
        if getattr(self, '_feat_key', None) == self._cursor:
            return self._feat_cache
        cw, ch = self.x_max - self.x_min, self.y_max - self.y_min
        if self.frame_util is not None:   # the frame is the space actually available
            cw, ch = self.anchor_x_max - self.x_min, self.anchor_y_max - self.y_min
        placed = sum(p[2] * p[3] for p in self.positions if p is not None)
        free = max(cw * ch - placed, 1e-9)
        cur_area = float(self.instance.area_targets[self.current_step().block_idx]) if not self.done() else 0.0
        future, largest, opts = 0.0, 0.0, []
        for st in self._pending[self._cursor + 1:]:
            i = st.block_idx
            if st.role == 'fixed':
                tp = self.instance.target_positions[i]
                a, o = float(tp[2]) * float(tp[3]), [(float(tp[2]), float(tp[3]))]
            elif st.role == 'mib_follower' and st.mib_leader in self.mib_shape:
                lw, lh = self.mib_shape[st.mib_leader]
                a, o = lw * lh, [(lw, lh)]
            else:
                a = max(float(self.instance.area_targets[i]), 0.0)
                o = [(math.sqrt(a * r), math.sqrt(a / r)) for r in self.aspect_ratios]
            future += a
            if a > largest:
                largest, opts = a, o
        fit = 1.0
        if opts and self.fine_factor > 1:
            sat = self._occ_sats()[0]
            ok = 0
            for (bw, bh) in opts:
                fw = max(1, math.ceil(bw / self.fine_cell_w - 1e-9))
                fh = max(1, math.ceil(bh / self.fine_cell_h - 1e-9))
                ok += fw <= self.fine_dim and fh <= self.fine_dim and bool(
                    (self._window_sums(self.fine_occ, fw, fh, sat=sat) == 0).any())
            fit = ok / len(opts)
        # Size of the largest future block: its real w/h if its shape is set,
        # else its square size.
        side_w, side_h = opts[0] if len(opts) == 1 else (math.sqrt(largest), math.sqrt(largest))
        n_left = len(self._pending) - self._cursor - 1
        f = torch.tensor([(cur_area + future) / free, largest / free,
                          min(side_w, cw) / cw if opts else 0.0, min(side_h, ch) / ch if opts else 0.0,
                          fit, n_left / max(len(self._pending), 1)], dtype=torch.float32)
        self._feat_key, self._feat_cache = self._cursor, f
        return f

    def lookahead_ok(self, gy: int, gx: int) -> bool:
        """True if, after place(gy, gx), each of the `lookahead` largest
        unplaced blocks still fits somewhere in at least one of its shapes.
        Always True when lookahead is off."""
        if not self.lookahead:
            return True
        upcoming = self._upcoming_shapes()
        if not upcoming:
            return True
        x, y, w, h = self._resolve_placement(gy, gx)
        grid = self.fine_occ.clone()
        fx0, fy0, fx1, fy1 = self._fine_rect_to_cells(x, y, w, h)
        grid[fy0:fy1, fx0:fx1] = 1.0
        sat = self._sat(grid)
        for opts in upcoming:
            fits = False
            for (bw, bh) in opts:
                fw = max(1, math.ceil(bw / self.fine_cell_w - 1e-9))
                fh = max(1, math.ceil(bh / self.fine_cell_h - 1e-9))
                if fw <= self.fine_dim and fh <= self.fine_dim and \
                        bool((self._window_sums(grid, fw, fh, sat=sat) == 0).any()):
                    fits = True
                    break
            if not fits:
                return False
        return True

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def finalize(self) -> List[Tuple[float, float, float, float]]:
        assert self.done(), "finalize() called before all blocks were placed"
        assert all(p is not None for p in self.positions), "unplaced block remains"
        return list(self.positions)

"""
Learned placer (10-07, roadmap "B"): a graph network trained on the
ground-truth layouts of the 1M-design training set to predict every block's
centre (in pin-box coordinates) and aspect ratio in one shot.

Why: started from the ground truth's arrangement, our shape fitting
(rl/sizing.py) plus the packer beat the ground truth's own score on the
anchors (0: 1.000, 49: 1.087, 98: 1.021 vs 1.091 / 1.133 / 1.080); our own
arrangements end ~0.1 higher. This model supplies the arrangement.

Prediction -> layout: the predicted centres give a sequence pair directly
(G+ by x - y, G- by x + y: a left of b puts a first in both, a below b
puts a later in G+ and earlier in G-), soft shapes start from the predicted
aspect ratios; `to_layout` packs it from the origin (collisions with
preplaced blocks repaired), then rl/sizing.py fit_shapes and the packer
take over as for any candidate.
"""
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .encoder import NetlistEncoder, build_block_features

FEAT_DIM = 15
LAYOUT_DIM = 4          # repair net (10-16): current centre and size of each block, pin-box units
GEOM_DIM = 8            # per pair: current relation one-hot, centre offset, signed gaps (pin-box units)
REL_W = 0.5             # weight of the pairwise relation loss
Position = Tuple[float, float, float, float]


def pin_box_of(instance) -> Tuple[float, float]:
    pins = instance.pins_pos
    v = pins[(pins[:, 0] >= 0) & (pins[:, 1] >= 0)] if pins is not None and pins.numel() else None
    if v is None or not len(v):
        s = math.sqrt(float(instance.area_targets.clamp(min=0).sum()) / 0.8)
        return s, s
    return max(float(v[:, 0].max()), 1e-3), max(float(v[:, 1].max()), 1e-3)


def features(instance, layout=None):
    """(block feats [n, 15], pin feats [p, 2], b2b, p2b with mean-normalized weights, (PX, PY)).
    `layout` (positions) appends each block's current centre and size: the repair net's input."""
    n = instance.block_count
    PX, PY = pin_box_of(instance)
    base = build_block_features(instance, 1.0)
    area = instance.area_targets[:n].clamp(min=0).float()
    c = instance.constraints[:n]
    tp = instance.target_positions[:n].float()
    pre = (c[:, 1] != 0).float() if c.shape[1] > 1 else torch.zeros(n)
    known = ((c[:, 0] != 0) | (c[:, 1] != 0)).float() if c.shape[1] > 1 else torch.zeros(n)
    extra = torch.stack([
        area.sqrt() / PX, area.sqrt() / PY,
        pre * (tp[:, 0] + tp[:, 2] / 2) / PX, pre * (tp[:, 1] + tp[:, 3] / 2) / PY,
        known * tp[:, 2].clamp(min=0) / PX, known * tp[:, 3].clamp(min=0) / PY,
    ], dim=1)
    feats = torch.cat([base.float(), extra], dim=1)
    if layout is not None:
        q = torch.tensor([list(map(float, r[:4])) for r in layout[:n]])
        feats = torch.cat([feats, torch.stack([(q[:, 0] + q[:, 2] / 2) / PX, (q[:, 1] + q[:, 3] / 2) / PY,
                                               q[:, 2] / PX, q[:, 3] / PY], dim=1)], dim=1)
    pins = instance.pins_pos.float() if instance.pins_pos is not None else torch.zeros(0, 2)
    pin_feats = pins / torch.tensor([PX, PY])

    def norm(e):
        if e is None or not e.numel():
            return torch.zeros(0, 3)
        e = e[e[:, 0] >= 0].float().clone()
        if len(e):
            e[:, 2] = e[:, 2] / e[:, 2].mean().clamp(min=1e-9)
        return e
    return feats, pin_feats, norm(instance.b2b_connectivity), norm(instance.p2b_connectivity), (PX, PY)


class PlacerNet(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 6, norm: bool = False, rel: bool = False,
                 layout: bool = False, rel_geom: bool = False):
        super().__init__()
        # layout (10-16): repair net, reads a current layout too (features(..., layout))
        self.layout = layout
        self.encoder = NetlistEncoder(hidden_dim=hidden, num_layers=layers,
                                      block_feat_dim=FEAT_DIM + (LAYOUT_DIM if layout else 0), norm=norm)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        # rel (10-09): pairwise arrangement head -- for a pair (i, j): i left of j,
        # j left of i, i below j, j below i (the four sequence-pair relations)
        self.rel = rel
        # rel_geom (10-16): the pair's geometry in the input layout also feeds the relation
        # head, so keeping a correct input relation is easy to learn
        self.rel_geom = rel_geom
        self.rel_head = (nn.Sequential(nn.Linear(3 * hidden + (GEOM_DIM if rel_geom else 0), hidden), nn.ReLU(),
                                       nn.Linear(hidden, 4)) if rel else None)
        self.geom_skip = nn.Linear(GEOM_DIM, 4) if rel_geom else None
        self._h = None

    def rel_logits(self, h, i, j, geom=None):
        hi, hj = h[i], h[j]
        if not self.rel_geom:
            return self.rel_head(torch.cat([hi + hj, (hi - hj), hi * hj], dim=1))
        return self.rel_head(torch.cat([hi + hj, (hi - hj), hi * hj, geom], dim=1)) + self.geom_skip(geom)

    def forward(self, feats, pin_feats, b2b, p2b):
        h, _ = self.encoder(feats, pin_feats, b2b, p2b)
        self._h = h
        g = h.mean(dim=0, keepdim=True).expand_as(h)
        out = self.head(torch.cat([h, g], dim=1))
        # centre in [0, 1]^2 of the pin box; log aspect bounded to the solver's 1/8 .. 8
        return torch.sigmoid(out[:, :2]), math.log(8.0) * torch.tanh(out[:, 2] / math.log(8.0))


def targets(instance, fp_sol):
    """Ground truth as (centre [n, 2] in pin-box units, log aspect [n])."""
    n = instance.block_count
    PX, PY = pin_box_of(instance)
    s = fp_sol[:n].float()
    w, h, x, y = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    centre = torch.stack([(x + w / 2) / PX, (y + h / 2) / PY], dim=1)
    return centre, torch.log(w.clamp(min=1e-6) / h.clamp(min=1e-6))


def gt_relations(instance, fp_sol):
    """Ground-truth sequence pair ranks (rp, rm) from the ground-truth layout."""
    from .packer import sequence_pair
    n = instance.block_count
    s = fp_sol[:n].double().numpy()
    w, h, x, y = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    gp, gm = sequence_pair(x, y, w, h)
    rp, rm = np.empty(n, int), np.empty(n, int)
    rp[gp], rm[gm] = np.arange(n), np.arange(n)
    return rp, rm


def pair_geom(instance, layout, i, j):
    """[pairs, GEOM_DIM] geometry of pairs (i, j) in `layout` (see GEOM_DIM)."""
    from .packer import sequence_pair
    n = instance.block_count
    PX, PY = pin_box_of(instance)
    q = np.array([list(map(float, r[:4])) for r in layout[:n]])
    x, y, w, h = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    gp, gm = sequence_pair(x, y, w, h)
    rp, rm = np.empty(n, int), np.empty(n, int)
    rp[gp], rm[gm] = np.arange(n), np.arange(n)
    one = np.eye(4)[rel_labels(rp, rm, i, j)]
    cx, cy = x + w / 2, y + h / 2
    gx = np.maximum(x[j] - (x[i] + w[i]), x[i] - (x[j] + w[j]))     # > 0: apart horizontally
    gy = np.maximum(y[j] - (y[i] + h[i]), y[i] - (y[j] + h[j]))
    g = np.concatenate([one, np.stack([(cx[j] - cx[i]) / PX, (cy[j] - cy[i]) / PY, gx / PX, gy / PY], 1)], 1)
    return torch.tensor(g, dtype=torch.float32)


def rel_labels(rp, rm, i, j):
    """Class per pair: 0 i left of j, 1 j left of i, 2 i below j, 3 j below i."""
    left_ij = (rp[i] < rp[j]) & (rm[i] < rm[j])
    left_ji = (rp[i] > rp[j]) & (rm[i] > rm[j])
    below_ij = (rp[i] > rp[j]) & (rm[i] < rm[j])
    return np.where(left_ij, 0, np.where(left_ji, 1, np.where(below_ij, 2, 3)))


REL_PAIRS = 2048        # pairs sampled per design for the relation loss
# 10-18: share of those pairs drawn from blocks that touch or nearly touch in the ground
# truth (85% accurate vs 96% for far pairs, ~40% of all errors on 4% of the pairs)
NEAR_FRAC = float(os.environ.get("FLOORSET_REL_NEAR", "0"))


def near_mask(instance, fp_sol, iu, ju):
    """Pairs (iu, ju) whose ground-truth rectangles touch or are closer than the mean block side."""
    n = instance.block_count
    s = fp_sol[:n].double().numpy()
    w, h, x, y = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    gx = np.maximum(x[ju] - (x[iu] + w[iu]), x[iu] - (x[ju] + w[ju]))
    gy = np.maximum(y[ju] - (y[iu] + h[iu]), y[iu] - (y[ju] + h[ju]))
    return np.maximum(np.maximum(gx, 0), np.maximum(gy, 0)) < np.sqrt(w * h).mean()


def rel_accuracy(net, instance, fp_sol):
    """(all pairs, near pairs) relation accuracy after a forward pass (uses net._h)."""
    n = instance.block_count
    rp, rm = gt_relations(instance, fp_sol)
    iu, ju = np.triu_indices(n, 1)
    lab = rel_labels(rp, rm, iu, ju)
    with torch.no_grad():
        ok = net.rel_logits(net._h, torch.tensor(iu), torch.tensor(ju)).argmax(1).numpy() == lab
    m = near_mask(instance, fp_sol, iu, ju)
    return float(ok.mean()), float(ok[m].mean()) if m.any() else 1.0


def loss_fn(net, instance, fp_sol, layout=None):
    feats, pin_feats, b2b, p2b, _ = features(instance, layout)
    c_pred, a_pred = net(feats, pin_feats, b2b, p2b)
    c_true, a_true = targets(instance, fp_sol)
    con = instance.constraints[:instance.block_count]
    movable = (con[:, 1] == 0) if con.shape[1] > 1 else torch.ones(instance.block_count, dtype=torch.bool)
    soft = movable & ((con[:, 0] == 0) if con.shape[1] > 0 else True)
    lc = (c_pred - c_true).abs().sum(dim=1)[movable].mean() if movable.any() else c_pred.sum() * 0
    la = (a_pred - a_true).abs()[soft].mean() if soft.any() else a_pred.sum() * 0
    loss = lc + 0.1 * la
    if getattr(net, "rel", False):
        n = instance.block_count
        rp, rm = gt_relations(instance, fp_sol)
        iu, ju = np.triu_indices(n, 1)
        if len(iu) > REL_PAIRS:
            if NEAR_FRAC > 0:
                nm = np.where(near_mask(instance, fp_sol, iu, ju))[0]
                kn = np.random.choice(nm, min(len(nm), int(NEAR_FRAC * REL_PAIRS)), replace=False)
                k = np.concatenate([kn, np.random.choice(len(iu), REL_PAIRS - len(kn), replace=False)])
            else:
                k = np.random.choice(len(iu), REL_PAIRS, replace=False)
            iu, ju = iu[k], ju[k]
        lab = torch.tensor(rel_labels(rp, rm, iu, ju))
        geom = pair_geom(instance, layout, iu, ju) if getattr(net, "rel_geom", False) else None
        logits = net.rel_logits(net._h, torch.tensor(iu), torch.tensor(ju), geom)
        lr_ = nn.functional.cross_entropy(logits, lab)
        loss = loss + REL_W * lr_
        net._last_rel = (float(lr_.detach()), float((logits.argmax(1) == lab).float().mean()))
    return loss, float(lc.detach()), float(la.detach())


def predict(net, instance, noise: float = 0.0, aspect_noise: float = 0.0, rng=None, layout=None):
    """Centres (absolute) and log aspects; optional Gaussian noise (centres in
    pin-box units, aspects in log units) turns the placer into a sampler."""
    with torch.no_grad():
        feats, pin_feats, b2b, p2b, (PX, PY) = features(instance, layout)
        c, a = net(feats, pin_feats, b2b, p2b)
    c, a = c.numpy().copy(), a.numpy().copy()
    if noise > 0 or aspect_noise > 0:
        rng = rng or np.random.default_rng()
        c = c + rng.normal(0, noise, c.shape)
        a = a + rng.normal(0, aspect_noise, a.shape)
    return c[:, 0] * PX, c[:, 1] * PY, a


def rel_sequences(net, instance, layout=None):
    """Sequence pair from the pairwise relation head: rank blocks by their total
    probability of coming first in G+ (left of, or above, the other block) and in
    G- (left of, or below)."""
    n = instance.block_count
    with torch.no_grad():
        feats, pin_feats, b2b, p2b, _ = features(instance, layout)
        net(feats, pin_feats, b2b, p2b)
        iu, ju = np.triu_indices(n, 1)
        geom = pair_geom(instance, layout, iu, ju) if getattr(net, "rel_geom", False) else None
        pr = torch.softmax(net.rel_logits(net._h, torch.tensor(iu), torch.tensor(ju), geom), dim=1).numpy()
    # classes: 0 i left j, 1 j left i, 2 i below j, 3 j below i
    before_p = pr[:, 0] + pr[:, 3]        # i before j in G+ (i left of j, or i above j)
    before_m = pr[:, 0] + pr[:, 2]        # i before j in G- (i left of j, or i below j)
    sp, sm = np.zeros(n), np.zeros(n)
    np.add.at(sp, iu, before_p); np.add.at(sp, ju, 1 - before_p)
    np.add.at(sm, iu, before_m); np.add.at(sm, ju, 1 - before_m)
    return np.argsort(-sp, kind="stable"), np.argsort(-sm, kind="stable")


def to_layout(instance, net, noise: float = 0.0, aspect_noise: float = 0.0, rng=None, pred=None,
              use_rel: Optional[bool] = None, seqs=None, layout=None):
    """Predicted (or sampled) arrangement packed from the origin:
    (positions, (gp, gm), collision overshoot). `seqs` = (gp, gm) given
    (e.g. sampled by scripts/rl_relation.py) instead of predicted."""
    from . import packer as K
    from . import repack as R
    n = instance.block_count
    cx, cy, la = pred if pred is not None else predict(net, instance, noise, aspect_noise, rng, layout)
    con = instance.constraints[:n]
    tp = instance.target_positions[:n].numpy().astype(float)
    area = instance.area_targets[:n].numpy().astype(float)
    fixed = (con[:, 0] != 0).numpy()
    pre = (con[:, 1] != 0).numpy() if con.shape[1] > 1 else np.zeros(n, bool)
    from .repack import MAX_ASPECT
    asp = np.exp(np.clip(la, -math.log(MAX_ASPECT), math.log(MAX_ASPECT)))
    w = np.where(fixed | pre, tp[:, 2], np.sqrt(area * asp))
    h = np.where(fixed | pre, tp[:, 3], area / np.maximum(w, 1e-9))
    mib = con[:, 2].numpy() if con.shape[1] > 2 else np.zeros(n)
    for g in set(mib[mib > 0].tolist()):           # one shape per MIB group (as the solver does)
        m = np.where(mib == g)[0]
        known = m[fixed[m] | pre[m]]
        ms = m[~(fixed[m] | pre[m])]
        if len(known):
            w[ms], h[ms] = tp[known[0], 2], tp[known[0], 3]
        elif len(ms):
            w[ms] = w[ms[0]]
            h[ms] = area[ms] / w[ms]
    cx = np.where(pre, tp[:, 0] + tp[:, 2] / 2, cx)
    cy = np.where(pre, tp[:, 1] + tp[:, 3] / 2, cy)
    if use_rel is None:     # env FLOORSET_USE_REL=0 builds the arrangement from centres even with a relation head
        import os
        use_rel = os.environ.get("FLOORSET_USE_REL", "1") != "0"
    if seqs is not None:
        gp, gm = (np.asarray(q) for q in seqs)
    elif use_rel and getattr(net, "rel", False) and n > 1:
        gp, gm = rel_sequences(net, instance, layout)
    else:
        gp = np.argsort(cx - cy, kind="stable")
        gm = np.argsort(cx + cy, kind="stable")
    start = [(float(cx[i] - w[i] / 2), float(cy[i] - h[i] / 2), float(w[i]), float(h[i])) for i in range(n)]
    P = R._problem(instance, start)
    P["pos"][:, 2], P["pos"][:, 3] = w, h
    Bt = K._Batch(P, 1.0, 1.0, (0.0, 0.0))
    x, y, over = Bt.pack(gm[None], np.argsort(gp)[None], np.argsort(gm)[None], w[None], h[None])
    pos = [(float(x[0, i]), float(y[0, i]), float(w[i]), float(h[i])) for i in range(n)]
    return pos, (gp, gm), float(over[0])


def load(path: str) -> Optional[PlacerNet]:
    import os
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location="cpu")
    net = PlacerNet(**ck.get("cfg", {}))
    net.load_state_dict(ck["state"])
    net.eval()
    return net

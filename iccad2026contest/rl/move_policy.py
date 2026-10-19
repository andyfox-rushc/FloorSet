"""Learned move proposals for the packer (10-12, rl/packer.py compact_pack).

The packer's search proposes BATCH random moves per step (reshape, swap in
G+ / G- / both, ideal-spot swap, move a next to b on one side), packs them
all and keeps the best. Here a small network scores many more candidate
moves from cheap features (where each block's nets pull it, block sizes,
whether a block sets the bounding box, search progress) and only the most
promising BATCH are packed, so each step's packing work goes further.

Training is contextual-bandit RL on the packer's own experience: logged
moves with the cost change they caused (normalized by the search's start
temperature), the network regresses that change; the next round collects
with the trained network filtering, and so on (scripts/train_move_policy.py).
"""
import numpy as np

# move types (descriptor[0])
RESHAPE, SWAP_P, SWAP_M, SWAP_B, SWAP_IDEAL, NEXT, EDGE = range(7)
N_TYPES = 7
N_FEATS = N_TYPES + 4 + 13


def features(desc, ctx):
    """[m, N_FEATS] features for m move descriptors (type, a, b, side, factor).

    ctx: dict with cx, cy (current centres), w, h, ix, iy (ideal centres, nan
    where a block has no nets), deg (normalized net weight), area (normalized),
    edge_x, edge_y (bool: block touches the right / top bbox edge), span
    (bbox max side), frac (search progress)."""
    d = np.asarray(desc, dtype=float)
    m = len(d)
    t = d[:, 0].astype(int)
    a = d[:, 1].astype(int)
    b = np.where(d[:, 2] >= 0, d[:, 2], d[:, 1]).astype(int)
    side = d[:, 3].astype(int)
    fac = d[:, 4]
    cx, cy, w, h, ix, iy = ctx["cx"], ctx["cy"], ctx["w"], ctx["h"], ctx["ix"], ctx["iy"]
    S = max(ctx["span"], 1e-9)

    def dist(px, py, k):
        v = (np.abs(px - ix[k]) + np.abs(py - iy[k])) / S
        return np.where(np.isnan(v), 0.0, v)

    # where a lands: swap -> b's centre; next-to -> beside b on `side`
    sx = np.select([side == 0, side == 1], [cx[b] + (w[a] + w[b]) / 2, cx[b] - (w[a] + w[b]) / 2], cx[b])
    sy = np.select([side == 2, side == 3], [cy[b] + (h[a] + h[b]) / 2, cy[b] - (h[a] + h[b]) / 2], cy[b])
    is_swap = (t >= SWAP_P) & (t <= SWAP_IDEAL)
    is_next = t == NEXT
    ax_new = np.where(is_next, sx, np.where(is_swap, cx[b], cx[a]))
    ay_new = np.where(is_next, sy, np.where(is_swap, cy[b], cy[a]))
    gain_a = dist(cx[a], cy[a], a) - dist(ax_new, ay_new, a)
    gain_b = np.where(is_swap, dist(cx[b], cy[b], b) - dist(cx[a], cy[a], b), 0.0)
    onehot = np.zeros((m, N_TYPES))
    onehot[np.arange(m), np.clip(t, 0, N_TYPES - 1)] = 1.0
    f = np.column_stack([
        onehot,
        np.where(t == RESHAPE, np.log(np.maximum(fac, 1e-9)), 0.0),
        side == 0, side == 1, (side == 2) | (side == 3),
        ctx["area"][a], ctx["area"][b], ctx["deg"][a], ctx["deg"][b],
        dist(cx[a], cy[a], a), gain_a, gain_b, gain_a * ctx["deg"][a],
        ctx["edge_x"][a], ctx["edge_y"][a], ctx["edge_x"][b] | ctx["edge_y"][b],
        np.full(m, ctx["frac"]), np.abs(cx[a] - cx[b]) / S + np.abs(cy[a] - cy[b]) / S,
    ]).astype(np.float32)
    return f


class MovePolicy:
    """Numpy MLP N_FEATS -> H -> H -> 1 (predicted normalized cost change; lower is better)."""

    def __init__(self, path):
        z = np.load(path)
        self.mu, self.sd = z["mu"], z["sd"]
        self.W = [z[f"W{i}"] for i in range(3)]
        self.b = [z[f"b{i}"] for i in range(3)]

    def score(self, f):
        x = (f - self.mu) / self.sd
        for i in range(2):
            x = np.maximum(x @ self.W[i] + self.b[i], 0.0)
        return (x @ self.W[2] + self.b[2])[:, 0]


_POLICY = {}


def load(path):
    if path not in _POLICY:
        import os
        _POLICY[path] = MovePolicy(path) if path and os.path.exists(path) else None
    return _POLICY[path]

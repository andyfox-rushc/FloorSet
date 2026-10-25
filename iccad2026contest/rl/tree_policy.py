"""
Learned top-down slicing-tree builder for the partition flow (10-21, plan "A").

The annealer (rl/partition.py) starts from one tree that is invalid on most
large designs (41-51 bad leaves on test designs 60 and 90) and spends much of
its time repairing it. This policy builds the tree one cut at a time over the
real region, so it sees which splits leave a valid leaf:

  node (blocks, region) -> cut axis; a sort key per block along that axis;
  a split index among the sorted prefixes whose area share is in
  [SPLIT_LO, SPLIT_HI] and whose two sides pass the partitioner's own fit
  checks (failing ones are masked; if all fail, none are).

Zero white space holds by construction (exact-area cuts); only structure is
learned. Untrained, the policy is the annealer's first tree: longer-side cut,
seed order, split nearest half the area -- plus the validity mask. Training
(scripts/train_tree_policy.py): REINFORCE on Partitioner.cost of the finished
expression against the mean of K samples of the same design.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import partition as PT

SPLIT_LO, SPLIT_HI = 0.1, 0.9
N_KEY, N_AXIS, N_SPLIT = 14, 6, 6


def _mlp(n_in: int, hidden: int) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, 1))
    nn.init.zeros_(net[-1].weight)        # untrained = the hand-made first tree
    nn.init.zeros_(net[-1].bias)
    return net


class TreePolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.cfg = dict(hidden=hidden)
        self.key = _mlp(N_KEY, hidden)
        self.axis = _mlp(N_AXIS, hidden)
        self.split = _mlp(N_SPLIT, hidden)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.05)))
        self.axis_scale = nn.Parameter(torch.tensor(3.0))
        self.split_scale = nn.Parameter(torch.tensor(20.0))


def save(net: TreePolicy, path: str, **meta) -> None:
    torch.save({"state": net.state_dict(), "cfg": net.cfg, **meta}, path)


def load(path: str) -> Optional[TreePolicy]:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except OSError:
        return None
    net = TreePolicy(**ck["cfg"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net


# -----------------------------------------------------------------------------
# One node's split, exactly as Partitioner.evaluate does it
# -----------------------------------------------------------------------------
def _nd(part: PT.Partitioner, blocks) -> tuple:
    """Node tuple in Partitioner.evaluate's form (item, area, key); subtrees
    only need a non-int item."""
    a = float(part.P.area[list(blocks)].sum())
    return (int(blocks[0]), a, int(blocks[0])) if len(blocks) == 1 else (None, a, 0)


def node_split(part: PT.Partitioner, op: int, lo_blocks, hi_blocks, R: np.ndarray):
    """(valid, region of the lo side, region of the hi side) for cut `op`
    with lo_blocks left/below. Mirrors Partitioner.evaluate: a rigid leaf
    beside a subtree goes in a corner and the subtree gets the rest; otherwise
    the expression's cut, then the other axis and the swapped order, first
    fitting option kept. A rigid side's region is R itself (it is placed later)."""
    l, r = _nd(part, lo_blocks), _nd(part, hi_blocks)
    is_leaf = lambda nd: isinstance(nd[0], int)
    is_rigid = lambda nd: is_leaf(nd) and nd[0] in part.options

    def fits(nd, Q):
        if is_rigid(nd):
            return True
        if is_leaf(nd):
            return PT.badness_cached(Q) == 0.0
        return PT.components(Q) == 1

    # a rigid leaf can only fill an exact-fit region, so every subtree needs a
    # soft block to take the rest (two rigid siblings: 26 of 26 bad on design 40)
    all_rigid = lambda bs: all(int(b) in part.options for b in bs)
    rigid_only = (len(lo_blocks) > 1 and all_rigid(lo_blocks)) or (len(hi_blocks) > 1 and all_rigid(hi_blocks)) \
        or (is_rigid(l) and is_rigid(r))
    if is_rigid(l) != is_rigid(r):
        rg, other = (l, r) if is_rigid(l) else (r, l)
        side = (0 if rg is l else 1) if op == PT.V else (2 if rg is l else 3)
        bad, s, x, y, rest = part._place(rg[0], R, side)
        if bad:
            alt = part._place(rg[0], R, None)
            if alt[0] < bad:
                bad, s, x, y, rest = alt
        ok = bad == 0 and len(rest) > 0 and fits(other, rest) and not rigid_only
        return (ok, R, rest) if rg is l else (ok, rest, R)
    axis = 0 if op == PT.V else 1
    tot = PT.area(R)
    a = l[1] + r[1]
    first = None
    for ax, (p, q) in ((axis, (l, r)), (1 - axis, (l, r)), (axis, (r, l)), (1 - axis, (r, l))):
        lo, hi = PT.split(R, ax, tot * p[1] / a)
        if first is None:
            first = (p, lo, hi)
        if fits(p, lo) and fits(q, hi):
            return (not rigid_only, lo, hi) if p is l else (not rigid_only, hi, lo)
        lo2, hi2 = part._pocket_cut(R, ax, tot * p[1] / a, p, q)
        if lo2 is not None and fits(p, lo2) and fits(q, hi2):
            return (not rigid_only, lo2, hi2) if p is l else (not rigid_only, hi2, lo2)
    p, lo, hi = first
    return False, lo, hi


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------
class Context:
    """Per-design arrays the features need, built once per Partitioner."""

    def __init__(self, part: PT.Partitioner):
        P = self.P = part.P
        self.part = part
        self.S = np.concatenate([P.bi, P.bj]).astype(int)
        self.D = np.concatenate([P.bj, P.bi]).astype(int)
        self.Wt = np.concatenate([P.bw, P.bw])
        self.pinsum = part.pinsum
        self.code = np.array(part.code)
        self.rigid = np.array([i in part.options for i in range(P.n)])
        self.pos0 = part.seeds.astype(float).copy()
        for i, (s, x, y) in part.pre.items():
            self.pos0[i] = (x + s.cx, y + s.cy)
        for b, R in part.pockets.items():
            self.pos0[b] = PT.K.centroid(R)
        for b, (s, x, y) in part.pocket_rigid.items():
            self.pos0[b] = (x + s.cx, y + s.cy)

    def pulls(self, pos: np.ndarray, inn: np.ndarray):
        """Net-weighted pull on every block from partners inside / outside the
        node (pins count as outside): (sum_in [n,2], w_in [n], sum_out [n,2], w_out [n])."""
        n = self.P.n
        m = inn[self.D]
        s_in = np.stack([np.bincount(self.S[m], self.Wt[m] * pos[self.D[m], k], n) for k in (0, 1)], 1)
        w_in = np.bincount(self.S[m], self.Wt[m], n)
        o = ~m
        s_out = np.stack([np.bincount(self.S[o], self.Wt[o] * pos[self.D[o], k], n) for k in (0, 1)], 1)
        w_out = np.bincount(self.S[o], self.Wt[o], n)
        return s_in, w_in, s_out + self.pinsum[:, :2], w_out + self.pinsum[:, 2]


def _bbox(R):
    return float(R[:, 0].min()), float(R[:, 1].min()), float(R[:, 2].max()), float(R[:, 3].max())


def node_features(ctx: Context, B: np.ndarray, R: np.ndarray, pos: np.ndarray):
    """Axis features [N_AXIS] and a function ax -> key features [m, N_KEY]."""
    P = ctx.P
    x0, y0, x1, y1 = _bbox(R)
    L = (max(x1 - x0, 1e-9), max(y1 - y0, 1e-9))
    lo = (x0, y0)
    inn = np.zeros(P.n, dtype=bool)
    inn[B] = True
    s_in, w_in, s_out, w_out = ctx.pulls(pos, inn)
    wsum = w_in[B] + w_out[B]
    m = len(B)
    a_node = float(P.area[B].sum())

    def norm(v, k):
        return np.clip((v - lo[k]) / L[k], -1.0, 2.0)

    pull = [norm((s_in[B, k] + s_out[B, k]) / np.maximum(wsum, 1e-12), k) for k in (0, 1)]
    for k in (0, 1):
        pull[k] = np.where(wsum > 0, pull[k], 0.5)
    axis_f = np.array([math.log(L[0] / L[1]), math.log(m) / 5, float(np.std(pull[0])), float(np.std(pull[1])),
                       float(ctx.rigid[B].mean()), PT.area(R) / (L[0] * L[1])], dtype=np.float32)

    def key_f(ax):
        o = 1 - ax
        f = np.zeros((m, N_KEY), dtype=np.float32)
        f[:, 0] = norm(pos[B, ax], ax)
        f[:, 1] = norm(pos[B, o], o)
        f[:, 2] = pull[ax]
        f[:, 3] = np.where(w_out[B] > 0, norm(s_out[B, ax] / np.maximum(w_out[B], 1e-12), ax), 0.5)
        f[:, 4] = np.where(wsum > 0, w_out[B] / np.maximum(wsum, 1e-12), 0.0)
        f[:, 5] = np.where(w_in[B] > 0, norm(s_in[B, ax] / np.maximum(w_in[B], 1e-12), ax), 0.5)
        f[:, 6] = np.where(wsum > 0, w_in[B] / np.maximum(wsum, 1e-12), 0.0)
        f[:, 7] = P.area[B] / a_node
        f[:, 8] = ctx.rigid[B]
        f[:, 9] = (ctx.code[B] & (1 if ax == 0 else 8)) != 0
        f[:, 10] = (ctx.code[B] & (2 if ax == 0 else 4)) != 0
        f[:, 11] = math.log(m) / 5
        f[:, 12] = math.log(L[ax] / L[o])
        f[:, 13] = np.sqrt(P.area[B]) / L[ax]
        return f

    return axis_f, key_f


# -----------------------------------------------------------------------------
# Rollout
# -----------------------------------------------------------------------------
def rollout(net: TreePolicy, part: PT.Partitioner, ctx: Optional[Context] = None, greedy: bool = False,
            rng: Optional[torch.Generator] = None, mask: bool = True
            ) -> Tuple[List[int], Optional[torch.Tensor], dict]:
    """Build one slicing tree over part.operands / part.region.
    Returns (Polish expression, summed log-probability or None when greedy,
    stats: nodes, nodes with every split masked)."""
    ctx = ctx or Context(part)
    ops = list(part.operands)
    stats = {"nodes": 0, "all_masked": 0}
    if len(ops) <= 1:
        return ops, None, stats
    pos = ctx.pos0.copy()
    logp = []
    sigma = net.log_sigma.exp()
    tree = {}                                        # node id -> (op, lo id, hi id) or block
    queue = [(0, np.array(ops), part.region)]
    nid = 1
    while queue:
        k, B, R = queue.pop(0)
        if len(B) == 1:
            tree[k] = int(B[0])
            stats.setdefault("regions", {})[int(B[0])] = R
            continue
        stats["nodes"] += 1
        axis_f, key_f = node_features(ctx, B, R, pos)
        # cut axis: + favours a vertical cut (sort along x)
        za = net.axis_scale * float(axis_f[0]) + net.axis(torch.from_numpy(axis_f)).squeeze()
        if greedy:
            ax = 0 if float(za) >= 0 else 1
        else:
            ax = 0 if float(torch.rand((), generator=rng)) < float(torch.sigmoid(za)) else 1
            logp.append(nn.functional.logsigmoid(za if ax == 0 else -za))
        # sort keys along the axis
        f = torch.from_numpy(key_f(ax))
        mu = f[:, 0] + net.key(f).squeeze(-1)
        if greedy:
            keys = mu.detach()
        else:
            keys = (mu + sigma * torch.randn(mu.shape, generator=rng)).detach()
            logp.append(torch.distributions.Normal(mu, sigma).log_prob(keys).sum())
        order = B[np.argsort(keys.numpy(), kind="stable")]
        ks = keys.numpy()[np.argsort(keys.numpy(), kind="stable")]
        # split index: prefixes with area share in [SPLIT_LO, SPLIT_HI], plus the one nearest half
        acc = np.cumsum(part.P.area[order])[:-1] / float(part.P.area[order].sum())
        cand = [i for i in range(len(acc)) if SPLIT_LO <= acc[i] <= SPLIT_HI]
        near = int(np.argmin(np.abs(acc - 0.5)))
        if near not in cand:
            cand.append(near)
            cand.sort()
        op = PT.V if ax == 0 else PT.H
        res = [node_split(part, op, order[:i + 1], order[i + 1:], R) for i in cand]
        valid = np.array([r[0] for r in res])
        if mask and not valid.any():
            stats["all_masked"] += 1
        m = len(B)
        gap = np.diff(ks)
        gmean = max(float(gap.mean()), 1e-9)
        sf = np.array([[acc[i] - 0.5, abs(acc[i] - 0.5), (i + 1) / m, gap[i] / gmean, math.log(m) / 5,
                        float(ctx.rigid[order[0]] if i == 0 else ctx.rigid[order[-1]] if i == m - 2 else 0.0)]
                       for i in cand], dtype=np.float32)
        zs = -net.split_scale * torch.from_numpy(sf[:, 1]) + net.split(torch.from_numpy(sf)).squeeze(-1)
        if mask and valid.any():
            zs = zs.masked_fill(torch.from_numpy(~valid), -1e9)
        if greedy:
            j = int(torch.argmax(zs))
        else:
            pj = torch.softmax(zs, 0)
            j = int(torch.multinomial(pj.detach(), 1, generator=rng))
            logp.append(torch.log_softmax(zs, 0)[j])
        i = cand[j]
        _, Rl, Rh = res[j]
        lo_b, hi_b = order[:i + 1], order[i + 1:]
        for Bc, Rc in ((lo_b, Rl), (hi_b, Rh)):
            if len(Rc):                              # estimates move into the child's box
                bx0, by0, bx1, by1 = _bbox(Rc)
                pos[Bc, 0] = np.clip(pos[Bc, 0], bx0, bx1)
                pos[Bc, 1] = np.clip(pos[Bc, 1], by0, by1)
        tree[k] = (op, nid, nid + 1)
        queue.append((nid, lo_b, Rl))
        queue.append((nid + 1, hi_b, Rh))
        nid += 2

    expr: List[int] = []
    stack = [(0, False)]
    while stack:                                     # post-order: lo, hi, op
        k, done = stack.pop()
        t = tree[k]
        if isinstance(t, int):
            expr.append(t)
        elif done:
            expr.append(t[0])
        else:
            stack += [(k, True), (t[2], False), (t[1], False)]
    return expr, (torch.stack(logp).sum() if logp else None), stats

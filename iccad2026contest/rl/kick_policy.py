"""
Learned kicks for the pair re-cut refiner's iterated local search (10-21 night).

rl/refine.py descends greedily to a local optimum, then kicks it with a few
random re-cuts and descends again, keeping improvements. Here a small network
chooses the kicks instead: every candidate move (touching soft pair, cut
option) is scored from local features and k moves are sampled from the
softmax. Reward: relative score drop after the kick and the greedy descent;
training compares K kicks from the same state (scripts/train_kick_policy.py).
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import partition as PT
from .refine import Refiner, snap

N_F = 10
MAX_PAIRS = 60        # pairs scored per kick


class KickPolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.cfg = dict(hidden=hidden)
        self.net = nn.Sequential(nn.Linear(N_F, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)          # untrained = uniform random kicks
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):
        return self.net(f).squeeze(-1)


def save(net, path, **meta):
    torch.save({"state": net.state_dict(), "cfg": net.cfg, **meta}, path)


def load(path) -> Optional[KickPolicy]:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except OSError:
        return None
    net = KickPolicy(**ck["cfg"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net


def _pull(R: Refiner, b) -> np.ndarray:
    """Net-weighted mean position of b's partners and pins (where its nets pull it)."""
    P = R.P
    sx = sy = sw = 0.0
    for k in R.b2b[b]:
        o = P.bj[k] if P.bi[k] == b else P.bi[k]
        sx += P.bw[k] * R.c[o, 0]
        sy += P.bw[k] * R.c[o, 1]
        sw += P.bw[k]
    for k in R.p2b[b]:
        sx += P.pw[k] * P.px[k]
        sy += P.pw[k] * P.py[k]
        sw += P.pw[k]
    return np.array([sx / sw, sy / sw]) if sw > 0 else R.c[b].copy()


def candidates(R: Refiner, max_pairs: Optional[int] = None, rng=None) -> Tuple[List[tuple], np.ndarray]:
    """Valid (u, v, ru, rv) re-cuts of touching soft pairs (a random sample of
    max_pairs pairs when given) and their features."""
    P = R.P
    s0 = R.score(R.hp, R.viol)
    scale = math.sqrt(P.W * P.H)
    pull = {}
    wsum = float(P.bw.mean()) if len(P.bw) else 1.0
    pair_w = {}
    for k, (i, j) in enumerate(zip(P.bi, P.bj)):
        key = (min(i, j), max(i, j))
        pair_w[key] = pair_w.get(key, 0.0) + float(P.bw[k])
    out, feats = [], []
    pairs = R.pairs()
    if max_pairs is not None and len(pairs) > max_pairs:
        idx = (rng or np.random).choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    for u, v in pairs:
        for b in (u, v):
            if b not in pull:
                pull[b] = _pull(R, b)
        Rr = np.concatenate([R.rects[u], R.rects[v]])
        au = PT.area(R.rects[u])
        blocks = [u, v]
        hp0 = R.hp_local(blocks, R.c)
        sv0 = R.soft_viol(blocks)
        du0 = np.abs(R.c[u] - pull[u]).sum() / scale
        dv0 = np.abs(R.c[v] - pull[v]).sum() / scale
        old = (R.rects[u], R.rects[v], R.c[u].copy(), R.c[v].copy())
        for ax in (0, 1):
            for first_u in (True, False):
                lo, hi = PT.split(Rr, ax, au if first_u else PT.area(Rr) - au)
                lo, hi = snap(lo), snap(hi)
                ru, rv = (lo, hi) if first_u else (hi, lo)
                if not len(ru) or not len(rv) or PT.badness_cached(ru) or PT.badness_cached(rv):
                    continue
                R.rects[u], R.rects[v] = ru, rv
                R.c[u], R.c[v] = PT.K.centroid(ru), PT.K.centroid(rv)
                gain = hp0 - R.hp_local(blocks, R.c)
                sv = R.soft_viol(blocks)
                du1 = np.abs(R.c[u] - pull[u]).sum() / scale
                dv1 = np.abs(R.c[v] - pull[v]).sum() / scale
                R.rects[u], R.rects[v] = old[0], old[1]
                R.c[u], R.c[v] = old[2], old[3]
                s1 = R.score(R.hp - gain, R.viol + sv - sv0)
                if abs(s1 - s0) < 1e-12 and (ru.tobytes() == old[0].tobytes()):
                    continue                          # the current cut itself
                out.append((u, v, ru, rv, gain, sv - sv0))
                feats.append([100 * (s0 - s1) / s0, hp0 / max(R.hp, 1e-9) * 10,
                              pair_w.get((min(u, v), max(u, v)), 0.0) / wsum, du0 + dv0, (du1 + dv1) - (du0 + dv0),
                              math.log(au / PT.area(R.rects[v])), float(sv0 > 0), float(sv - sv0),
                              float(ax), math.log(len(ru) + len(rv))])
    return out, np.array(feats, dtype=np.float32).reshape(-1, N_F)


def apply(R: Refiner, m) -> None:
    u, v, ru, rv, gain, dv = m
    R.rects[u], R.rects[v] = ru, rv
    R.c[u], R.c[v] = PT.K.centroid(ru), PT.K.centroid(rv)
    R.hp -= gain
    R.viol += dv


def policy_kick(R: Refiner, net: KickPolicy, k: int, gen: torch.Generator, greedy: bool = False,
                max_pairs: Optional[int] = MAX_PAIRS):
    """One scoring pass, then k moves sampled from it without replacement (a move
    touching a block already moved is skipped). Returns the summed log-prob.
    v1 re-scored every candidate after each move: too slow to pay off (10-22)."""
    rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen)))
    cands, f = candidates(R, max_pairs, rng)
    if not cands:
        return None
    z = net(torch.from_numpy(f))
    logp, used, avail = [], set(), torch.ones(len(cands), dtype=torch.bool)
    for _ in range(k):
        zz = z.masked_fill(~avail, -1e9)
        if not bool(avail.any()):
            break
        if greedy:
            j = int(torch.argmax(zz))
        else:
            j = int(torch.multinomial(torch.softmax(zz, 0).detach(), 1, generator=gen))
            logp.append(torch.log_softmax(zz, 0)[j])
        avail[j] = False
        u, v = cands[j][0], cands[j][1]
        if u in used or v in used:
            continue
        apply(R, cands[j])
        used.update((u, v))
    return torch.stack(logp).sum() if logp else None


def run_learned(R: Refiner, net: KickPolicy, time_budget: float, seed: int = 0, k: int = 3) -> dict:
    """refine.Refiner.run with policy-chosen kicks."""
    import time
    t0 = time.time()
    t_end = t0 + time_budget
    R.hp = R.hp_ref = max(R.P.hpwl(R.c[:, 0], R.c[:, 1]), 1e-9)
    R.viol = R.soft_viol(range(R.P.n))
    moves = R._greedy(t_end, 20)
    gen = torch.Generator().manual_seed(seed)
    best = (R.score(R.hp, R.viol), dict(R.rects), R.c.copy(), R.hp, R.viol)
    kicks = wins = 0
    with torch.no_grad():
        while time.time() < t_end:
            policy_kick(R, net, k, gen)
            moves += R._greedy(t_end, 20)
            kicks += 1
            sc = R.score(R.hp, R.viol)
            if sc < best[0] - 1e-12:
                best = (sc, dict(R.rects), R.c.copy(), R.hp, R.viol)
                wins += 1
            else:
                _, rects, c, R.hp, R.viol = best
                R.rects, R.c = dict(rects), c.copy()
    return dict(moves=moves, kicks=kicks, wins=wins, seconds=time.time() - t0, hpwl=R.hp, viol=R.viol,
                score=R.score(R.hp, R.viol))

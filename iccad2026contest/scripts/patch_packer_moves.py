"""Patch rl/packer.py for learned move proposals (apply only when no run is live).
   python patch_packer_moves.py [--check]"""
import sys

P = "/home/afox/floorplanner_prime/FloorSet-Prime/iccad2026contest/rl/packer.py"
s = open(P).read()
pairs = [
# settings
('''WALLBOX = os.environ.get("FLOORSET_WALLBOX", "0") != "0"
''', '''WALLBOX = os.environ.get("FLOORSET_WALLBOX", "0") != "0"
# Learned move scorer (10-12, rl/move_policy.py): FLOORSET_MOVE_POLICY = weights
# (.npz, "" = off); each step FILTER x BATCH moves are proposed and the best-scored
# BATCH packed, MOVE_EXPLORE of them picked at random from the rest.
MOVE_POLICY = os.environ.get("FLOORSET_MOVE_POLICY", "")
MOVE_FILTER = int(os.environ.get("FLOORSET_MOVE_FILTER", "4"))
MOVE_EXPLORE = float(os.environ.get("FLOORSET_MOVE_EXPLORE", "0.25"))
MOVE_LOG = []           # scripts/collect_moves.py: (features, cost change / start temperature, step)
MOVE_LOGGING = False
'''),
# propose: descriptors
('''    def propose(gp, gm, w, h):
        gp, gm, w, h = gp.copy(), gm.copy(), w, h
        r = rng.random()''', '''    def propose(gp, gm, w, h):
        """-> (gp, gm, w, h, (move type, a, b, side, factor)), types as in rl/move_policy.py."""
        gp, gm, w, h = gp.copy(), gm.copy(), w, h
        desc = (0, 0, -1, -1, 1.0)
        r = rng.random()'''),
('''                for i in u:
                    a = w[i] * h[i]
                    nw = min(max(w[i] * f, math.sqrt(a / R.MAX_ASPECT)), math.sqrt(a * R.MAX_ASPECT))
                    w[i], h[i] = nw, a / nw
            return gp, gm, w, h''', '''                for i in u:
                    a = w[i] * h[i]
                    nw = min(max(w[i] * f, math.sqrt(a / R.MAX_ASPECT)), math.sqrt(a * R.MAX_ASPECT))
                    w[i], h[i] = nw, a / nw
                desc = (6, int(u[0]), -1, -1, f)
            return gp, gm, w, h, desc'''),
('''            f = math.exp(rng.gauss(0.0, 0.3))
            for i in units[rng.randrange(len(units))]:
                a = w[i] * h[i]
                nw = min(max(w[i] * f, math.sqrt(a / R.MAX_ASPECT)), math.sqrt(a * R.MAX_ASPECT))
                w[i], h[i] = nw, a / nw''', '''            f = math.exp(rng.gauss(0.0, 0.3))
            u = units[rng.randrange(len(units))]
            for i in u:
                a = w[i] * h[i]
                nw = min(max(w[i] * f, math.sqrt(a / R.MAX_ASPECT)), math.sqrt(a * R.MAX_ASPECT))
                w[i], h[i] = nw, a / nw
            desc = (0, int(u[0]), -1, -1, f)'''),
('''            which = rng.randrange(3)
            if rng.random() < IDEAL and not P["preplaced"][a]:     # a with the block at its ideal spot
                b2 = ideal_partner(a)
                if b2 is not None:
                    b, which = b2, 2''', '''            which = rng.randrange(3)
            desc = (1 + which, a, b, -1, 1.0)
            if rng.random() < IDEAL and not P["preplaced"][a]:     # a with the block at its ideal spot
                b2 = ideal_partner(a)
                if b2 is not None:
                    b, which = b2, 2
                    desc = (4, a, b, -1, 1.0)'''),
('''            if a == b:
                return gp, gm, w, h
            side = rng.randrange(4)      # 0 right of b, 1 left, 2 above, 3 below''', '''            if a == b:
                return gp, gm, w, h, (5, a, b, 0, 1.0)
            side = rng.randrange(4)      # 0 right of b, 1 left, 2 above, 3 below
            desc = (5, a, b, side, 1.0)'''),
('''            gp, gm = out
        return gp, gm, w, h
''', '''            gp, gm = out
        return gp, gm, w, h, desc

    policy = None
    if MOVE_POLICY or MOVE_LOGGING:
        from . import move_policy as MP
        policy = MP.load(MOVE_POLICY) if MOVE_POLICY else None
        e_i, e_j, e_w = (np.asarray(P[k], dtype=int if k != "b2b_w" else float) for k in ("b2b_i", "b2b_j", "b2b_w"))
        sw_tot = np.bincount(e_i, e_w, n) + np.bincount(e_j, e_w, n) + pin_sum[:, 2]
        has_nets = sw_tot > 0
        deg = sw_tot / max(float(sw_tot[has_nets].mean()) if has_nets.any() else 1.0, 1e-12)
        area_n = (w0 * h0) / max(float((w0 * h0).mean()), 1e-12)

    def move_context(frac):
        cx, cy = cur_xy
        w, h = cur[3], cur[4]
        sx = np.bincount(e_i, e_w * cx[e_j], n) + np.bincount(e_j, e_w * cx[e_i], n) + pin_sum[:, 0]
        sy = np.bincount(e_i, e_w * cy[e_j], n) + np.bincount(e_j, e_w * cy[e_i], n) + pin_sum[:, 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            ix, iy = np.where(has_nets, sx / sw_tot, np.nan), np.where(has_nets, sy / sw_tot, np.nan)
        X1, Y1 = float((cx + w / 2).max()), float((cy + h / 2).max())
        tol = 1e-6 * max(X1, Y1)
        return dict(cx=cx, cy=cy, w=w, h=h, ix=ix, iy=iy, deg=deg, area=area_n,
                    edge_x=cx + w / 2 >= X1 - tol, edge_y=cy + h / 2 >= Y1 - tol,
                    span=max(X1 - float((cx - w / 2).min()), Y1 - float((cy - h / 2).min())), frac=frac)
'''),
# main loop
('''        props = [propose(cur[1], cur[2], cur[3], cur[4]) for _ in range(BATCH)]
        cs, xs, ys, ovs = evaluate(''', '''        frac = (time.time() - start) / time_budget
        feats = None
        if policy is not None:
            pool_ = [propose(cur[1], cur[2], cur[3], cur[4]) for _ in range(MOVE_FILTER * BATCH)]
            fp = MP.features([p[4] for p in pool_], move_context(frac))
            order = np.argsort(policy.score(fp))
            k_rand = int(round(MOVE_EXPLORE * BATCH))
            keep = order[:BATCH - k_rand].tolist()
            rest = order[BATCH - k_rand:]
            keep += rng.sample(rest.tolist(), min(k_rand, len(rest)))
            props = [pool_[i] for i in keep]
            feats = fp[keep]
        else:
            props = [propose(cur[1], cur[2], cur[3], cur[4]) for _ in range(BATCH)]
            if MOVE_LOGGING:
                feats = MP.features([p[4] for p in props], move_context(frac))
        cs, xs, ys, ovs = evaluate('''),
('''                slot[0], slot[1] = float(tc[kt]), (props[kt], xs[kt], ys[kt])''',
 '''                slot[0], slot[1] = float(tc[kt]), (props[kt][:4], xs[kt], ys[kt])'''),
('''        frac = (time.time() - start) / time_budget
        temp = t0 * (0.001 ** frac)''', '''        if MOVE_LOGGING and feats is not None:
            MOVE_LOG.append((feats, ((cs - cur[0]) / t0).astype(np.float32), steps))
        temp = t0 * (0.001 ** frac)'''),
('''            cur = (float(cs[k]),) + props[k]''', '''            cur = (float(cs[k]),) + tuple(props[k][:4])'''),
]
bad = [a for a, _ in pairs if s.count(a) != 1]
if bad:
    print("NOT UNIQUE / MISSING:", *[b[:80] for b in bad], sep="\n  ")
    sys.exit(1)
if "--check" in sys.argv:
    print("all", len(pairs), "anchors found once"); sys.exit(0)
for a, b in pairs:
    s = s.replace(a, b)
open(P, "w").write(s)
print("patched")

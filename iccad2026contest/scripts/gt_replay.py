"""Ground-truth replay (10-07): can the placement env reproduce the
ground-truth layouts? Step 0 of imitation pre-training.

For each design, drive GridPlacementEnv with "expert" actions taken from the
ground truth: the aspect choice nearest the block's true aspect ratio, the
grid cell holding its true lower-left corner (nearest legal cell when that
one is masked). Records how often the true cell is legal, how far snapping
moves blocks, whether the replay finishes, and the replayed layout's cost.

  python scripts/gt_replay.py --ids 0-99 --out gt_replay_val.json [--canvas pin|default]
"""
import argparse
import json
import math
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import polish_bench as PB


def replay(inst, gt, pin_canvas=True, gt_canvas=False):
    """Drive the env with ground-truth actions. Returns (positions or None, stats)."""
    import numpy as np
    from rl.env import GridPlacementEnv
    if gt_canvas:       # test only: canvas = true outline (one unused dummy pin at its corner)
        import dataclasses
        import torch
        W = max(q[0] + q[2] for q in gt)
        H = max(q[1] + q[3] for q in gt)
        pins = inst.pins_pos[(inst.pins_pos[:, 0] >= 0) & (inst.pins_pos[:, 1] >= 0)]
        pins = torch.cat([pins.clamp(max=torch.tensor([W, H], dtype=pins.dtype)),
                          torch.tensor([[W, H]], dtype=pins.dtype)])
        inst = dataclasses.replace(inst, pins_pos=pins)
        pin_canvas = True
    env = GridPlacementEnv(inst, pin_canvas=pin_canvas)
    st = dict(pos_steps=0, exact=0, snap_cells=[], snap_rel=[], aspect_err=[], auto=0, fail=None)
    try:
        while not env.done():
            step = env.current_step()
            i = step.block_idx
            gx0, gy0, gw, gh = gt[i]
            if env.needs_aspect():
                ratios = np.array(env.aspect_ratios)
                want = math.log(max(gw, 1e-9) / max(gh, 1e-9))
                mask = env.aspect_mask().numpy()
                if not mask.any():
                    raise RuntimeError("no shape fits")
                d = np.where(mask, np.abs(np.log(ratios) - want), np.inf)
                a = int(np.argmin(d))
                st["aspect_err"].append(float(abs(math.log(ratios[a]) - want)))
                w, h = env.choose_aspect(a)
            else:
                w, h = env.current_shape()
            res = env.position_mask(w, h)
            if res is None:
                st["auto"] += 1
                continue
            mask = res[0].numpy()
            ty = (gy0 - env.y_min) / env.cell_h
            tx = (gx0 - env.x_min) / env.cell_w
            cy, cx = int(math.floor(ty)), int(math.floor(tx))
            st["pos_steps"] += 1
            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx]:
                gy, gx = cy, cx
                st["exact"] += 1
            else:
                ys, xs = np.nonzero(mask)
                if not len(ys):
                    raise RuntimeError("no legal cell")
                k = int(np.argmin((ys + 0.5 - ty) ** 2 + (xs + 0.5 - tx) ** 2))
                gy, gx = int(ys[k]), int(xs[k])
            env.place(gy, gx)
            px, py = env.positions[i][0], env.positions[i][1]
            dist = math.hypot(px - gx0, py - gy0)
            st["snap_cells"].append(dist / math.hypot(env.cell_w, env.cell_h))
            st["snap_rel"].append(dist / math.sqrt(max(gw * gh, 1e-9)))
        return env.finalize(), st
    except (RuntimeError, AssertionError) as e:
        st["fail"] = f"{type(e).__name__}: {e} at step {env._cursor}/{len(env._pending)}"
        return None, st


def _one(args):
    tid, mode = args
    pin_canvas, gt_canvas = mode != "default", mode == "gt"
    from rl.data import from_validation_sample
    inst = from_validation_sample(PB.EV, tid)
    _, gt = PB.EV._extract_baseline(tid, PB.EV.dataset[tid]["label"], inst.b2b_connectivity,
                                    inst.p2b_connectivity, inst.pins_pos, inst.block_count)
    gt = [tuple(map(float, q[:4])) for q in gt]
    pos, st = replay(inst, gt, pin_canvas, gt_canvas)
    out = dict(design=tid, n=inst.block_count, pos_steps=st["pos_steps"], exact=st["exact"], auto=st["auto"],
               snap_cells_mean=sum(st["snap_cells"]) / max(len(st["snap_cells"]), 1),
               snap_rel_mean=sum(st["snap_rel"]) / max(len(st["snap_rel"]), 1),
               aspect_err_mean=sum(st["aspect_err"]) / max(len(st["aspect_err"]), 1), fail=st["fail"])
    if pos is not None:
        s = PB._score(inst, pos)
        out.update(cost=s.cost, feasible=s.is_feasible, hpwl_gap=s.hpwl_gap, area_gap=s.area_gap,
                   vrel=s.violations_relative)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="0-99")
    ap.add_argument("--canvas", choices=["pin", "default", "gt"], default="pin")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, _, hi = a.ids.partition("-")
    ids = list(range(int(lo), int(hi or lo) + 1))
    with get_context("forkserver").Pool(a.workers, initializer=PB._init) as pool:
        res = pool.map(_one, [(t, a.canvas) for t in ids])
    json.dump(res, open(a.out, "w"), indent=1)
    done = [r for r in res if r["fail"] is None]
    steps = sum(r["pos_steps"] for r in res)
    W = sum(math.exp(r["n"] / 12) for r in done)
    print(f"canvas={a.canvas}: finished {len(done)}/{len(res)}; true cell legal on "
          f"{sum(r['exact'] for r in res) / max(steps, 1):.1%} of {steps} position steps; "
          f"auto-placed {sum(r['auto'] for r in res)}")
    print(f"snap distance mean {sum(r['snap_cells_mean'] for r in res) / len(res):.2f} cells "
          f"({sum(r['snap_rel_mean'] for r in res) / len(res):.2f} block sizes); "
          f"aspect error mean {sum(r['aspect_err_mean'] for r in res) / len(res):.3f} (log ratio)")
    if done:
        m = lambda k: sum(r[k] for r in done) / len(done)
        print(f"replayed layouts: cost {m('cost'):.3f} (weighted "
              f"{sum(math.exp(r['n'] / 12) * r['cost'] for r in done) / W:.3f}) hpwl {m('hpwl_gap'):.3f} "
              f"area {m('area_gap'):.3f} vrel {m('vrel'):.3f}; feasible {sum(r['feasible'] for r in done)}/{len(done)}")
    fails = [r for r in res if r["fail"]]
    for r in fails[:8]:
        print(f"  fail design {r['design']} (n {r['n']}): {r['fail']}")


if __name__ == "__main__":
    main()

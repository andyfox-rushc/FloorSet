"""Live-frame scan (rl/env.py frame_util): per frame utilization, over the
validation cases, greedy rollout finish rate and compacted hpwl_gap /
area_gap / violations / cost, plus how often the frame had to grow.

  python scripts/frame_scan.py --frames none 0.8 0.85 0.9 --util 0.70 --out frame_scan.json
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch

from iccad2026_evaluate import ContestEvaluator
from rl.data import from_validation_sample
from rl.env import GridPlacementEnv
from rl.finetune import choose_grid_dim
from rl.networks import ActorCritic, load_state
from rl.ppo import collect_episode
from rl.reward import _evaluate

EV = NET = None
LAST = {}


def _init(ckpt):
    global EV, NET
    torch.set_num_threads(1)
    EV = ContestEvaluator(data_path="../", verbose=False)
    EV._load_dataset()
    NET = ActorCritic()
    load_state(NET, ckpt)
    orig = GridPlacementEnv.finalize

    def fin(self):
        LAST['growths'] = self.frame_growths
        return orig(self)
    GridPlacementEnv.finalize = fin


def _run(args):
    case, frame, util = args
    inst = from_validation_sample(EV, case)
    LAST.clear()
    ep = collect_episode(NET, inst, choose_grid_dim(inst.block_count), use_baseline=True, greedy=True,
                         env_kwargs={"utilization": util, "frame_util": frame})
    if ep.positions is None:
        return dict(case=case, frame=frame, finished=False)
    m = _evaluate(inst, ep.positions)
    return dict(case=case, frame=frame, finished=True, feasible=m.is_feasible, cost=m.cost,
                hpwl_gap=m.hpwl_gap, area_gap=m.area_gap, vrel=m.violations_relative,
                bnd=m.boundary_violations, grp=m.grouping_violations, growths=LAST.get('growths', 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", nargs="+", default=["none", "0.8", "0.85", "0.9"])
    p.add_argument("--util", type=float, default=0.70)
    p.add_argument("--cases", type=int, default=100)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--checkpoint", default="checkpoints/policy.pt")
    p.add_argument("--out", default="frame_scan.json")
    a = p.parse_args()
    frames = [None if f == "none" else float(f) for f in a.frames]
    jobs = [(c, f, a.util) for f in frames for c in range(a.cases)]
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.checkpoint,)) as pool:
        rows = pool.map(_run, jobs, chunksize=1)
    print(f"{'frame':>6} {'finish':>6} {'cost':>6} {'hpwl':>6} {'area':>6} {'vrel':>6} {'bnd':>5} {'grp':>5} {'grow':>5}")
    for f in frames:
        rs = [r for r in rows if r['frame'] == f]
        ok = [r for r in rs if r['finished']]
        avg = lambda k: sum(r[k] for r in ok) / max(len(ok), 1)
        print(f"{str(f):>6} {len(ok)/len(rs):6.0%} {avg('cost'):6.3f} {avg('hpwl_gap'):6.3f} {avg('area_gap'):6.3f} "
              f"{avg('vrel'):6.3f} {avg('bnd'):5.2f} {avg('grp'):5.2f} {avg('growths'):5.2f}")
    # Paired comparison against the first config, on cases both finished.
    base = {r['case']: r for r in rows if r['frame'] == frames[0] and r['finished']}
    for f in frames[1:]:
        pairs = [(base[r['case']], r) for r in rows if r['frame'] == f and r['finished'] and r['case'] in base]
        if pairs:
            d = lambda k: sum(x[k] - b[k] for b, x in pairs) / len(pairs)
            print(f"  {f} vs {frames[0]} on {len(pairs)} shared cases: "
                  f"cost {d("cost"):+.3f} area {d("area_gap"):+.3f} hpwl {d("hpwl_gap"):+.3f}")
    print(f"{time.time() - t0:.0f} s")
    with open(a.out, "w") as fh:
        json.dump(rows, fh)


if __name__ == "__main__":
    main()

"""Scan working-canvas utilization targets: per config, over validation
cases, how often random and policy rollouts finish, and the compacted
greedy rollout's hpwl_gap / area_gap / violations / cost.

  python scripts/canvas_scan.py --configs 0.5:1:0 0.6:4:1 ...   (U:fine:aspect_mask[:die_anchor[:shape_fit[:lookahead_k[:edge_last]]]])
"""
import argparse, json, random, sys, time
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


def _init(ckpt):
    global EV, NET
    torch.set_num_threads(1)
    EV = ContestEvaluator(data_path="../", verbose=False)
    EV._load_dataset()
    NET = ActorCritic()
    load_state(NET, ckpt)


def _random_rollout(env, seed, use_mask):
    rng = random.Random(seed)
    while not env.done():
        if env.needs_aspect():
            if use_mask:
                ok = env.aspect_mask().nonzero().flatten().tolist()
                if not ok:
                    raise RuntimeError("no shape fits")
                idx = rng.choice(ok)
            else:
                idx = rng.randrange(len(env.aspect_ratios))
            w, h = env.choose_aspect(idx)
        else:
            w, h = env.current_shape()
        res = env.position_mask(w, h)
        if res is None:
            continue
        valid = res[0].nonzero().tolist()
        for attempt in range(9):  # same re-draw budget as collect_episode
            gy, gx = valid[rng.randrange(len(valid))]
            if attempt == 8 or len(valid) == 1 or env.lookahead_ok(gy, gx):
                break
            valid.remove([gy, gx])
        env.place(gy, gx)
    return env.finalize()


def run_case(args):
    idx, util, fine, use_mask, die, fit, look, edge, n_random, n_sampled = args
    inst = from_validation_sample(EV, idx)
    g = choose_grid_dim(inst.block_count)
    kw = dict(utilization=util, fine_factor=fine, die_anchor=die, shape_fit=fit, lookahead=look, edge_last=edge)
    out = dict(idx=idx, rand_ok=0, samp_ok=0)
    env = GridPlacementEnv(inst, grid_dim=g, **kw)
    out['canvas_area'] = (env.x_max - env.x_min) * (env.y_max - env.y_min)
    for s in range(n_random):
        env.reset()
        try:
            _random_rollout(env, s, use_mask)
            out['rand_ok'] += 1
        except RuntimeError:
            pass
    for s in range(n_sampled):
        torch.manual_seed(s)
        ep = collect_episode(NET, inst, g, False, env_kwargs=kw, use_aspect_mask=use_mask)
        out['samp_ok'] += ep.positions is not None
    ep = collect_episode(NET, inst, g, False, greedy=True, env_kwargs=kw, use_aspect_mask=use_mask)
    out['greedy_ok'] = ep.positions is not None
    if ep.positions is not None:
        r = _evaluate(inst, ep.positions)
        out.update({k: float(getattr(r, k)) for k in ('hpwl_gap', 'area_gap', 'violations_relative', 'cost',
                                                       'is_feasible', 'boundary_violations', 'bbox_area')})
        p = inst.pins_pos
        v = p[(p[:, 0] != -1) | (p[:, 1] != -1)]
        out['bbox_over_pinbox'] = out['bbox_area'] / float(v[:, 0].max() * v[:, 1].max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', required=True)
    ap.add_argument('--cases', type=int, default=100)
    ap.add_argument('--random', type=int, default=5)
    ap.add_argument('--sampled', type=int, default=3)
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--checkpoint', default='checkpoints/policy.pt')
    ap.add_argument('--out', default='canvas_scan_results.json')
    a = ap.parse_args()
    results = {}
    with Pool(a.workers, initializer=_init, initargs=(a.checkpoint,)) as pool:
        for cfg in a.configs:
            util, fine, m, *extra = cfg.split(':')
            extra += ['0'] * (4 - len(extra))
            die, fit, look, edge = extra[0] == '1', extra[1] == '1', int(extra[2]), extra[3] == '1'
            t = time.time()
            jobs = [(i, float(util), int(fine), m == '1', die, fit, look, edge, a.random, a.sampled)
                    for i in range(a.cases)]
            rows = pool.map(run_case, jobs)
            results[cfg] = rows
            n = len(rows)
            g = [r for r in rows if r['greedy_ok']]
            avg = lambda k: sum(r[k] for r in g) / max(len(g), 1)
            print(f"U={util} F{fine} mask={m} die={int(die)} fit={int(fit)} look={look} edge={int(edge)} | random {sum(r['rand_ok'] for r in rows)}/{n*a.random} "
                  f"| greedy {len(g)}/{n} | sampled {sum(r['samp_ok'] for r in rows)}/{n*a.sampled} "
                  f"| greedy hpwl_gap {avg('hpwl_gap'):.3f} area_gap {avg('area_gap'):.3f} "
                  f"vrel {avg('violations_relative'):.3f} bnd {avg('boundary_violations'):.2f} "
                  f"bbox/pinbox {avg('bbox_over_pinbox'):.3f} cost {avg('cost'):.3f} | {time.time()-t:.0f}s",
                  flush=True)
            json.dump(results, open(a.out, 'w'))


if __name__ == '__main__':
    main()

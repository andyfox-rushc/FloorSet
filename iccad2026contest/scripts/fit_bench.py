"""fit_shapes on finished layouts (10-07): python scripts/fit_bench.py official_eval_1007_tol.json [iters lr]"""
import json, math, sys, time
from multiprocessing import get_context
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import polish_bench as PB

def one(args):
    r, iters, lr = args
    from rl.data import from_validation_sample
    from rl.sizing import fit_shapes
    from rl.packer import in_box, pin_box
    inst = from_validation_sample(PB.EV, r['test_id'])
    pos = [tuple(p) for p in r['positions']]
    t0 = time.time()
    out = fit_shapes(inst, pos, iters=iters, lr=lr)
    s = PB._score(inst, out)
    return (r['block_count'], r['cost'], s.cost, r['hpwl_gap'], s.hpwl_gap, r['area_gap'], s.area_gap,
            r['violations_relative'], s.violations_relative, time.time() - t0, in_box(out, pin_box(inst)))

if __name__ == '__main__':
    f = sys.argv[1]; iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1000; lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    T = json.load(open(f))['test_results']
    with get_context('forkserver').Pool(3, initializer=PB._init) as pool:
        res = pool.map(one, [(r, iters, lr) for r in T])
    W = sum(math.exp(r[0] / 12) for r in res)
    w = lambda k: sum(math.exp(r[0] / 12) * r[k] for r in res) / W
    m = lambda k: sum(r[k] for r in res) / len(res)
    print('fit_shapes(%d, lr %.3f) on %s: weighted %.4f -> %.4f | hpwl %.3f -> %.3f area %.3f -> %.3f vrel %.4f -> %.4f | improved %d | in pin box %d | %.1f s avg' % (
        iters, lr, f, w(1), w(2), m(3), m(4), m(5), m(6), m(7), m(8), sum(r[2] < r[1] - 1e-9 for r in res), sum(r[10] for r in res), m(9)))

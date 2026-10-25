"""Final step of the 10-21 night: per design, the best layout over several flows
(picked without the ground truth, as generate_prime_report_1022.best_of does),
then a longer pair re-cut refinement with iterated local search (rl/refine.py)
from a fresh seed; never worse than its input.

  python scripts/final_polish.py --flow label=file ... --seconds 60 --jobs 4 --out final.json
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _polish(job):
    import prime_evaluate as E
    from rl import partition as PT
    from rl.refine import Refiner
    r, seconds, seed, kick = job
    d = E.load_designs(ids=[r["id"]])[0]
    if not r.get("feasible") or not r.get("layout"):
        return r
    args = (d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"], d["constraints"],
            d["target_polygons"])
    R = Refiner(PT.Problem(*args), r["layout"])
    if kick:                                  # learned kicks (rl/kick_policy.py)
        import torch
        from rl import kick_policy as KP
        torch.set_num_threads(1)
        st = KP.run_learned(R, KP.load(kick), seconds, seed=seed)
    else:
        st = R.run(time_budget=seconds, seed=seed)
    lay = R.layout()
    n = E.evaluate(d, lay, runtime=r.get("runtime", 0.0) + st["seconds"])
    q = lambda x: x.get("quality_cost", x["cost"])
    if not n["feasible"] or q(n) > q(r):
        return r
    n["layout"] = [list(map(list, v)) for v in lay]
    n["picked"] = r.get("picked")
    n["polish"] = st
    return n


def main():
    import generate_prime_report_1022 as G
    import prime_evaluate as E
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", action="append", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kick-policy", help="learned kicks instead of random ones")
    a = ap.parse_args()
    flows = {}
    for spec in a.flow:
        label, f = spec.split("=", 1)
        flows[label] = {r["id"]: r for r in json.loads(Path(f).read_text())}
    ids = sorted(set().union(*flows.values()))            # a subset run (IDS=...) has fewer rows
    designs = {d["id"]: d for d in E.load_designs(ids=ids)}
    B = G.best_of(flows, designs)
    jobs = [(B[i], a.seconds, a.seed + i, a.kick_policy) for i in sorted(B)]
    with Pool(a.jobs) as pool:
        out = pool.map(_polish, jobs, chunksize=1)
    Path(a.out).write_text(json.dumps(out))
    E._summary(list(B.values()), "best of flows")
    E._summary(out, "polished")


if __name__ == "__main__":
    main()

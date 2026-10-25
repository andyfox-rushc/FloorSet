#!/bin/bash
# 10-09 overnight: after the big-net and RL placer runs, evaluate both with the full
# pipeline metric next to placer_c (same machine conditions) and print diversity stats.
cd "$(dirname "$0")/.."
until grep -q "^done:" logs/rl_placer.log && grep -q "^done:" logs/train_placer_big.log; do sleep 60; done
grep "^done:" logs/rl_placer.log logs/train_placer_big.log
for ck in placer_c placer_d placer_rl; do
  ../venv/bin/python scripts/placer_eval.py --ckpt checkpoints/$ck.pt --budget 6 --workers 6 --out placer_eval_1009_$ck.json 2>&1 | grep -v Warn | tail -1
done
../venv/bin/python - <<'PY' 2>&1 | grep -v Warn
import json, math
names = ['placer_c', 'placer_d', 'placer_rl']
ev = {n: {x['design']: x for x in json.load(open(f'placer_eval_1009_{n}.json'))} for n in names}
ev['A'] = {x['design']: x for x in json.load(open('placer_eval_snap1b.json'))}
ev['B'] = {x['design']: x for x in json.load(open('placer_eval_final1.json'))}
ids = [k for k in ev['placer_c'] if all(ev[n][k]['cost'] is not None for n in ev)]
W = sum(math.exp(ev['A'][k]['n'] / 12) for k in ids)
w = lambda f: sum(math.exp(ev['A'][k]['n'] / 12) * f(k) for k in ids) / W
for n in ev:
    print('%-10s alone %.4f' % (n, w(lambda k, n=n: ev[n][k]['cost'])))
base = ['A', 'B', 'placer_c']
print('min(A,B,C) %.4f' % w(lambda k: min(ev[n][k]['cost'] for n in base)))
for extra in ('placer_d', 'placer_rl'):
    print('min(A,B,C,%s) %.4f' % (extra, w(lambda k: min(ev[n][k]['cost'] for n in base + [extra]))))
print('min(all) %.4f' % w(lambda k: min(ev[n][k]['cost'] for n in ev)))
PY

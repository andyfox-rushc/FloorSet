#!/bin/bash
# 10-10: (A) continue relation net from placer_r.pt, (B) second relation net from placer_d.pt
# (different seed) for diversity; both 8 h on 4 threads. Then official runs: A alone, A + B.
cd "$(dirname "$0")/.."
PY=../venv/bin/python
$PY scripts/train_placer.py --hours 8 --threads 4 --seed 21 --hidden 256 --layers 8 --norm --rel \
    --resume checkpoints/placer_r.pt --lr 1e-4 --val-every 5000 --out checkpoints/placer_rel3.pt > logs/train_placer_rel3.log 2>&1 &
$PY scripts/train_placer.py --hours 8 --threads 4 --seed 31 --hidden 256 --layers 8 --norm --rel \
    --resume checkpoints/placer_d.pt --lr 2e-4 --val-every 5000 --out checkpoints/placer_relB.pt > logs/train_placer_relB.log 2>&1 &
wait
P3=$PWD/checkpoints/placer_rel3.pt; PB=$PWD/checkpoints/placer_relB.pt
FLOORSET_PLACER2=$P3 FLOORSET_PLACER3=$P3 \
    $PY iccad2026_evaluate.py --evaluate my_optimizer.py --output official_eval_1011_rel3.json > logs/official_eval_1011_rel3.log 2>&1
FLOORSET_PLACER2=$P3 FLOORSET_PLACER3=$P3 FLOORSET_PLACER4=$PB \
    FLOORSET_CANDIDATES="base,scratch0.75:it600:po1.0,scratch0.85:it600:po1.0,scratch0.90:it600:po1.0,scratch0.95:it600:po1.0,scratch1.0:it600:po1.0,learned,learned2,learned3,learned4" \
    $PY iccad2026_evaluate.py --evaluate my_optimizer.py --output official_eval_1011_relB.json > logs/official_eval_1011_relB.log 2>&1
echo chain done

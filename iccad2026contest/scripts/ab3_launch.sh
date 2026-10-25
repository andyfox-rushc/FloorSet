#!/bin/bash
# 10-04: 3-arm policy A/B from scratch, 4000 iterations each, same data seed:
# A = current network, B = whole-canvas position net (FLOORSET_POS_NET=unet),
# C = B + analytic-placer target map (FLOORSET_TARGET_MAP=1). No frame.
# Relaunching resumes each arm from its own policy.pt.
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
arm() {  # name env
    mkdir -p checkpoints/ab_$1
    R=""; [ -f checkpoints/ab_$1/policy.pt ] && R="--resume checkpoints/ab_$1/policy.pt"
    env $2 nohup python -u -m rl.train --iterations 4000 --episodes-per-iter 4 --ppo-epochs 4 \
        --checkpoint checkpoints/ab_$1/policy.pt --checkpoint-every 500 --threads 3 --data-seed 12345 $R \
        >> logs/train_ab_$1.log 2>&1 &
}
arm A "FLOORSET_POS_NET=cnn"
arm B "FLOORSET_POS_NET=unet"
arm C "FLOORSET_POS_NET=unet FLOORSET_TARGET_MAP=1"

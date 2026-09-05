#!/usr/bin/env python3
"""
Offline pretraining loop for the shared ActorCritic policy.

Uses the 1M-sample training set (LiteTensorData/, ~15GB) when it has been
downloaded (see README.md "Dataset Downloads"); otherwise falls back to
cycling the 100-sample validation set (LiteTensorDataTest/, already local)
so this script is runnable and testable today without that download.

Both sources carry their own ground-truth baseline metrics, so pretraining
always uses the exact contest cost (rl.reward.pretraining_reward) rather
than the no-baseline proxy used for per-instance fine-tuning at contest time
(see rl/finetune.py).

Before the main PPO loop (unless --resume-ing an existing checkpoint), this
also runs a staged encoder warm-start via supervised reward-prediction
pretraining across a diverse batch of instances -- see rl/pretrain.py for
why this is a separate phase, not just the reward-approx auxiliary loss PPO
already trains online.

Usage:
    python -m rl.train --iterations 200 --episodes-per-iter 8 \\
        --checkpoint checkpoints/policy.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from iccad2026_evaluate import ContestEvaluator  # noqa: E402
from rl.data import from_training_batch_item, from_validation_sample  # noqa: E402
from rl.networks import ActorCritic  # noqa: E402
from rl.ppo import collect_batch, ppo_update  # noqa: E402
from rl.pretrain import collect_reward_prediction_corpus, pretrain_encoder_on_reward_prediction  # noqa: E402


def iter_training_instances(data_path: str = "../"):
    """Yields FloorplanInstance objects, preferring the real 1M-sample
    training set when present, else cycling the local validation set.

    NOTE: the README documents the training set as living under
    `LiteTensorData/`, but the actual downloader (lite_dataset.py's
    download_dataset(), invoked by get_training_dataloader() ->
    FloorplanDatasetLite()) fetches LiteTensorData_v2.tar.gz and extracts it
    to `floorset_lite/worker_*/` -- that's the real presence check to use
    (lite_dataset.is_dataset_downloaded), not the README's path."""
    from lite_dataset import is_dataset_downloaded

    if is_dataset_downloaded(data_path):
        from iccad2026_evaluate import get_training_dataloader
        loader = get_training_dataloader(data_path=data_path, batch_size=1, shuffle=True)
        while True:
            for batch in loader:
                area_target, b2b_conn, p2b_conn, pins_pos, constraints, _, _, metrics = batch
                yield from_training_batch_item(
                    area_target.squeeze(0), b2b_conn.squeeze(0), p2b_conn.squeeze(0),
                    pins_pos.squeeze(0), constraints.squeeze(0), metrics.squeeze(0),
                )
    else:
        print("NOTE: 1M-sample training set (floorset_lite/) not found -- "
              "cycling the 100-sample validation set instead. Download the "
              "training set (see README.md) for real pretraining.",
              file=sys.stderr)
        ev = ContestEvaluator(data_path=data_path, verbose=False)
        ev._load_dataset()
        n = len(ev.dataset)
        i = 0
        while True:
            yield from_validation_sample(ev, i % n)
            i += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="../")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--episodes-per-iter", type=int, default=4)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--grid-dim", type=int, default=48)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--checkpoint", default="checkpoints/policy.pt")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--resume", default=None)
    p.add_argument("--pretrain-instances", type=int, default=20,
                    help="Number of diverse instances used for the staged encoder "
                         "reward-prediction warm-start (see rl/pretrain.py); 0 to skip. "
                         "Ignored when --resume-ing an existing checkpoint.")
    p.add_argument("--pretrain-rollouts-per-instance", type=int, default=4)
    p.add_argument("--pretrain-epochs", type=int, default=5)
    args = p.parse_args()

    net = ActorCritic(hidden_dim=args.hidden_dim)
    instances = iter_training_instances(args.data_path)
    if args.resume and os.path.exists(args.resume):
        net.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"resumed from {args.resume}")
    elif args.pretrain_instances > 0:
        print(f"Staged encoder warm-start: collecting rollouts from "
              f"{args.pretrain_instances} instances...")
        corpus_instances = [next(instances) for _ in range(args.pretrain_instances)]
        corpus = collect_reward_prediction_corpus(
            corpus_instances, net=net,
            rollouts_per_instance=args.pretrain_rollouts_per_instance,
            grid_dim=args.grid_dim,
        )
        print(f"  collected {len(corpus)} (state, reward) samples; "
              f"training encoder on reward prediction...")
        pretrain_encoder_on_reward_prediction(net, corpus, epochs=args.pretrain_epochs, verbose=True)
        print("  encoder warm-started; reward-approx head reset for PPO")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    ckpt_dir = os.path.dirname(args.checkpoint)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    t0 = time.time()
    for it in range(1, args.iterations + 1):
        inst = next(instances)
        episodes = collect_batch(net, inst, grid_dim=args.grid_dim, use_baseline=True,
                                  num_episodes=args.episodes_per_iter)
        stats = ppo_update(net, optimizer, episodes, epochs=args.ppo_epochs)
        avg_reward = sum(e.reward for e in episodes) / len(episodes)
        print(f"iter {it}/{args.iterations} blocks={inst.block_count} "
              f"avg_reward={avg_reward:.4f} loss={stats['loss']:.4f} "
              f"elapsed={time.time() - t0:.1f}s")

        if it % args.checkpoint_every == 0 or it == args.iterations:
            torch.save(net.state_dict(), args.checkpoint)
            print(f"  saved checkpoint -> {args.checkpoint}")


if __name__ == "__main__":
    main()

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
import gc
import os
import resource
import sys
import time
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from iccad2026_evaluate import ContestEvaluator  # noqa: E402
from rl.checkpoint_meta import read_checkpoint_meta, write_checkpoint_meta  # noqa: E402
from rl.data import from_training_batch_item, from_validation_sample  # noqa: E402
from rl.networks import ActorCritic, load_state  # noqa: E402
from rl.ppo import collect_batch, make_episode_pool, ppo_update  # noqa: E402
from rl.pretrain import collect_reward_prediction_corpus, pretrain_encoder_on_reward_prediction  # noqa: E402


def iter_shuffled_training_files(data_path: str, seed: int, buffer_files: int = 32):
    """Yields FloorplanInstances from the 1M-sample training set in random
    order: file order is shuffled with `seed`, and samples are drawn at
    random from a buffer holding `buffer_files` whole files (~1.5MB each).
    Each file holds 112 layouts of one block count, so reading in file
    order (the old behaviour) trained ~112 iterations on one size at a
    time, and every launch replayed the same first files."""
    import glob
    import random
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(data_path, "floorset_lite", "worker_*", "layouts*")))
    while True:
        rng.shuffle(files)
        buffer, next_file, since_load = [], 0, 0

        def load_next():
            nonlocal next_file
            c = torch.load(files[next_file])
            next_file += 1
            for li in range(len(c[0])):
                buffer.append((c[0][li][:, 0], c[1][li], c[2][li], c[3][li], c[0][li][:, 1:],
                               c[6][li], c[5][li]))

        while next_file < min(buffer_files, len(files)):
            load_next()
        while buffer:
            j = rng.randrange(len(buffer))
            buffer[j], buffer[-1] = buffer[-1], buffer[j]
            area_target, b2b, p2b, pins, constraints, metrics, fp_sol = buffer.pop()
            yield from_training_batch_item(area_target, b2b, p2b, pins, constraints, metrics, fp_sol)
            since_load += 1
            if since_load >= 112 and next_file < len(files):
                load_next()
                since_load = 0


def iter_training_instances(data_path: str = "../", seed: Optional[int] = None):
    """Yields FloorplanInstance objects, preferring the real 1M-sample
    training set when present, else cycling the local validation set.

    NOTE: the README documents the training set as living under
    `LiteTensorData/`, but the actual downloader (lite_dataset.py's
    download_dataset(), invoked by get_training_dataloader() ->
    FloorplanDatasetLite()) fetches LiteTensorData_v2.tar.gz and extracts it
    to `floorset_lite/worker_*/` -- that's the real presence check to use
    (lite_dataset.is_dataset_downloaded), not the README's path."""
    from lite_dataset import is_dataset_downloaded

    if is_dataset_downloaded(data_path) and seed is not None:
        print(f"training data: shuffled files, seed {seed}")
        yield from iter_shuffled_training_files(data_path, seed)
    elif is_dataset_downloaded(data_path):
        from iccad2026_evaluate import get_training_dataloader
        # shuffle=False: FloorplanDatasetLite caches one file at a time;
        # shuffled indices rarely reuse a file across the 24GB corpus, which
        # doesn't fit this machine's 15GB RAM, thrashing the page cache.
        loader = get_training_dataloader(data_path=data_path, batch_size=1, shuffle=False)
        while True:
            for batch in loader:
                area_target, b2b_conn, p2b_conn, pins_pos, constraints, _, fp_sol, metrics = batch
                yield from_training_batch_item(
                    area_target.squeeze(0), b2b_conn.squeeze(0), p2b_conn.squeeze(0),
                    pins_pos.squeeze(0), constraints.squeeze(0), metrics.squeeze(0),
                    fp_sol.squeeze(0),
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
    p.add_argument("--lr", type=float, default=3e-4,
                    help="Adam learning rate at the start of this launch (was a fixed 1e-3 "
                         "until 09-27; that made finish rates swing 90%%->45%% and back).")
    p.add_argument("--lr-final", type=float, default=3e-5,
                    help="Cosine-decayed to this by --iterations.")
    p.add_argument("--checkpoint", default="checkpoints/policy.pt")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--time-limit-hours", type=float, default=None,
                    help="Stop after this many wall-clock hours (checkpointing first), "
                         "regardless of --iterations. Default: no limit.")
    p.add_argument("--resume", default=None)
    p.add_argument("--pretrain-instances", type=int, default=20,
                    help="Number of diverse instances used for the staged encoder "
                         "reward-prediction warm-start (see rl/pretrain.py); 0 to skip. "
                         "Ignored when --resume-ing an existing checkpoint.")
    p.add_argument("--pretrain-rollouts-per-instance", type=int, default=4)
    p.add_argument("--pretrain-epochs", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=1,
                    help="Worker processes for parallel episode collection within each "
                         "iteration (see rl/ppo.py's make_episode_pool). 1 (default) "
                         "keeps the original single-process, sequential behavior.")
    p.add_argument("--util-start", type=float, default=0.6,
                    help="Canvas utilization target to start from (see rl/env.py).")
    p.add_argument("--util-max", type=float, default=0.75)
    p.add_argument("--util-step", type=float, default=0.025)
    p.add_argument("--util-window", type=int, default=400,
                    help="Episodes at a level before judging its finish rate.")
    p.add_argument("--probe-frac", type=float, default=0.25,
                    help="Share of iterations run at the next (tighter) utilization level. "
                         "The level advances when those probe episodes finish >= --probe-up.")
    p.add_argument("--probe-up", type=float, default=0.85)
    p.add_argument("--util-up", type=float, default=0.9,
                    help="Tighten when the finish rate over the window reaches this.")
    p.add_argument("--util-down", type=float, default=0.6,
                    help="Loosen when the finish rate over the window falls below this.")
    p.add_argument("--data-seed", type=int, default=None,
                    help="Seed for the shuffled training-data order (default: random per "
                         "launch, so a resume sees new designs). -1 = old file order.")
    p.add_argument("--no-normalize-advantages", action="store_true",
                    help="Use raw advantages (the pre-09-26 behaviour). Normalizing keeps "
                         "failed rollouts from dominating updates at the canvas edge.")
    p.add_argument("--threads", type=int, default=None,
                    help="torch intra-op threads (default: torch's choice). Rollout ops are "
                         "small, so few threads avoid oversubscription.")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    net = ActorCritic(hidden_dim=args.hidden_dim)
    data_seed = args.data_seed if args.data_seed is not None else int.from_bytes(os.urandom(4), "little")
    instances = iter_training_instances(args.data_path, seed=None if data_seed == -1 else data_seed)
    utilization = args.util_start
    start_it = 0
    lr_origin = 0
    if args.resume and os.path.exists(args.resume):
        load_state(net, args.resume)
        print(f"resumed from {args.resume}")
        meta = read_checkpoint_meta(args.resume)
        if "utilization" in meta:
            utilization = meta["utilization"]
            print(f"  resuming canvas utilization {utilization:.3f}")
        # Continue the iteration count so history snapshots aren't overwritten.
        start_it = int(meta.get("iteration", 0))
        lr_origin = int(meta.get("lr_origin", start_it))
        if start_it:
            print(f"  resuming at iteration {start_it}")
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
    pool = make_episode_pool(args.num_workers, hidden_dim=args.hidden_dim) if args.num_workers > 1 else None
    if pool is not None:
        # NOTE: worker processes are only busy *during* collect_batch's
        # pool.starmap call, which blocks this process until it returns --
        # they're idle (no CPU use) while this process runs ppo_update
        # afterward, so this process keeps its full default thread count for
        # that phase rather than being capped alongside the workers.
        print(f"episode collection: {args.num_workers} worker processes")

    ckpt_dir = os.path.dirname(args.checkpoint)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    def save_checkpoint(it):
        torch.save(net.state_dict(), args.checkpoint)
        write_checkpoint_meta(args.checkpoint, {"utilization": utilization, "iteration": it,
                                                 "lr_origin": lr_origin})
        # Versioned snapshot alongside the canonical "latest" path above
        # (which my_optimizer.py and --resume always load): a training
        # run that suddenly destabilizes (see module docstring history --
        # an overnight run's fallback rate jumped from ~3% to ~29%+ and
        # never recovered) would otherwise have already overwritten the
        # last healthy state by the time anyone notices.
        history_dir = os.path.join(ckpt_dir or ".", "history")
        os.makedirs(history_dir, exist_ok=True)
        snapshot_path = os.path.join(history_dir, f"policy_iter{it:06d}.pt")
        torch.save(net.state_dict(), snapshot_path)
        write_checkpoint_meta(snapshot_path, {"utilization": utilization, "iteration": it,
                                              "lr_origin": lr_origin})
        print(f"  saved checkpoint -> {args.checkpoint} (+ {snapshot_path})")

    time_limit_s = args.time_limit_hours * 3600 if args.time_limit_hours else None
    # Adaptive canvas curriculum with gradual probing (09-27): most iterations
    # run at the current utilization, --probe-frac at the next level up. Jumping
    # the whole run to the next level at once knocked the policy's finish rate
    # at the old level from ~91% to ~60% every time. Advance when the probe
    # episodes finish >= --probe-up over a window; loosen when the current
    # level falls below --util-down.
    level_outcomes: list = []
    probe_outcomes: list = []
    import math as _math
    import random as _random
    probe_rng = _random.Random(data_seed)
    last_it = start_it

    t0 = time.time()
    try:
        if start_it >= args.iterations:
            # Same line shape as a normal last iteration, so the watchdog sees "done".
            print(f"iter {start_it}/{args.iterations} already complete; nothing to do")
        for it in range(start_it + 1, args.iterations + 1):
            if time_limit_s is not None and time.time() - t0 >= time_limit_s:
                print(f"time limit of {args.time_limit_hours}h reached at iter {last_it}; stopping")
                if last_it % args.checkpoint_every != 0:
                    save_checkpoint(last_it)
                break
            inst = next(instances)
            # Decay measured from the iteration this schedule started at (kept in
            # the checkpoint meta), so a watchdog relaunch continues it.
            span = max(args.iterations - lr_origin, 1)
            frac = min(max((it - lr_origin) / span, 0.0), 1.0)
            lr = args.lr_final + (args.lr - args.lr_final) * 0.5 * (1 + _math.cos(_math.pi * frac))
            for g in optimizer.param_groups:
                g["lr"] = lr
            next_u = round(min(args.util_max, utilization + args.util_step), 4)
            probing = next_u > utilization and probe_rng.random() < args.probe_frac
            run_u = next_u if probing else utilization
            episodes = collect_batch(net, inst, grid_dim=args.grid_dim, use_baseline=True,
                                      num_episodes=args.episodes_per_iter, pool=pool,
                                      env_kwargs={"utilization": run_u})
            outcomes = [e.positions is not None for e in episodes]
            if probing:
                probe_outcomes += outcomes
                if len(probe_outcomes) >= args.util_window // 2:
                    rate = sum(probe_outcomes) / len(probe_outcomes)
                    adv = rate >= args.probe_up
                    print(f"  canvas probe {next_u:.3f}: finish rate {rate:.3f} over "
                          f"{len(probe_outcomes)} episodes -> {'advance' if adv else 'stay'} "
                          f"(level {next_u if adv else utilization:.3f})")
                    if adv:
                        utilization, level_outcomes = next_u, []
                    probe_outcomes = []
            else:
                level_outcomes += outcomes
                if len(level_outcomes) >= args.util_window:
                    rate = sum(level_outcomes) / len(level_outcomes)
                    new_u = utilization
                    if rate < args.util_down:
                        new_u = max(args.util_start, utilization - args.util_step)
                        probe_outcomes = []
                    print(f"  canvas utilization {utilization:.3f}: finish rate {rate:.3f} "
                          f"over {len(level_outcomes)} episodes -> {new_u:.3f}")
                    utilization = round(new_u, 4)
                    level_outcomes = []
            stats = ppo_update(net, optimizer, episodes, epochs=args.ppo_epochs,
                               normalize_advantages=not args.no_normalize_advantages)
            avg_reward = sum(e.reward for e in episodes) / len(episodes)
            del episodes  # drop references before the next iteration's allocations

            rss_mb = None
            if it % 50 == 0:
                gc.collect()
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

            line = (f"iter {it}/{args.iterations} blocks={inst.block_count} "
                    f"avg_reward={avg_reward:.4f} loss={stats['loss']:.4f} "
                    f"grad_norm={stats['grad_norm']:.4f} util={run_u:.3f}{'p' if probing else ''} "
                    f"lr={lr:.2e} "
                    f"elapsed={time.time() - t0:.1f}s")
            if rss_mb is not None:
                line += f" rss_mb={rss_mb:.1f}"
            print(line)

            last_it = it
            if it % args.checkpoint_every == 0 or it == args.iterations:
                save_checkpoint(it)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()

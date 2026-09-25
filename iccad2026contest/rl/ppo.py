"""
PPO rollout collection + clipped-surrogate update for the sequential
placement policy.

The terminal reward (rl/reward.py -- exact contest cost when a baseline is
available, the no-baseline proxy otherwise) is exactly decomposable into a
per-placement-step piece, since HPWL is a sum over independent edges each
resolved the moment their second endpoint is placed, and bbox area only
ever grows monotonically as blocks are added (see
GridPlacementEnv._step_deltas / rl/reward.py's step_quality_delta). Each
Transition's `return_to_go` is the sum of its own and every later step's
piece, so PPO's advantage differentiates between the specific actions that
actually caused wirelength/area, instead of every action in an episode
sharing one identical whole-episode advantage (the previous design, which
made it impossible to reinforce good position choices over bad ones within
a single ~100-step rollout). The value head predicts this per-step
return-to-go, conditioned on the static graph embedding + progress --
still not a discounted multi-reward return (no per-step discounting; γ=1
throughout, matching the exact, un-discounted decomposition), just a
target that now actually varies within an episode instead of being
constant.

The reward-approximation head (for AlphaChip fidelity) is unaffected by
this -- it still regresses the whole-episode realized reward (`ep.reward`),
matching its documented purpose of predicting the final realized outcome
from any state; it does not itself drive the policy gradient.

2026-09-14: both scalar heads also take `reward_so_far` (the running sum of
every prior transition's `step_reward` in this rollout) -- without it,
`net.value(global_emb, progress)` is a function of only two things that are
identical across every stochastic rollout of the same instance at the same
step (rl/ordering.py's placement order never depends on which action was
taken), so it provably cannot explain any of a specific rollout's actual
outcome. A live diagnostic measured the resulting gap directly: return_to_go's
std across rollouts, at a fixed step, was 10x-300x (avg ~44x-82x across two
instances) the size of the per-step signal a position choice actually
controls -- meaning the advantage estimate PPO trains on was almost pure
downstream noise. See rl/networks.py's `ScalarHead` docstring and
algorithm.md's "Reward" section.

2026-09-20: found the actual source of most of that variance, downstream of
`reward_so_far` rather than fixed by it. The terminal reward is
MULTIPLICATIVE (quality_factor * violation_factor), but step_quality_delta
only ever computed the additive quality piece with no violation_factor
scaling -- so collect_episode's one residual correction was dumping the
entire quality*violation interaction (unbounded, scales with the whole
episode's quality) onto whichever transition happened to be last, instead
of distributing it across the steps that actually caused it. Fixed by
rescaling every step's tracked delta by the same violation_factor the
terminal reward used before computing the residual, which is now a small,
bounded, Q-independent leftover (exactly zero for inference_reward). This
doesn't make `reward_so_far` redundant -- the value head still benefits from
seeing progress -- but it should substantially shrink the variance that
feature was compensating for. See rl/reward.py's
`_reward_and_violation_factor` docstring for the exact derivation.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from iccad2026_evaluate import M_PENALTY

from .encoder import build_block_features, build_pin_features
from .env import GridPlacementEnv
from .networks import NEG_INF, ActorCritic, masked_log_softmax
from .reward import _reward_and_violation_factor, step_quality_delta


MIN_TEMPERATURE = 1e-3


def _temp_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature (floored, never truly 0) before
    softmax -- lower temperature makes the distribution more peaked. Used
    identically at collection time and update time (see Transition.temperature)
    so the PPO importance ratio always compares the same tempered
    distribution before/after a policy update, not two different ones."""
    return logits / max(temperature, MIN_TEMPERATURE)


def _temp_scale_batch(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """Batched form of _temp_scale: temperatures is [N] (one per row of
    logits' leading dim), broadcast over logits' remaining dims."""
    shape = (temperatures.shape[0],) + (1,) * (logits.dim() - 1)
    return logits / temperatures.clamp(min=MIN_TEMPERATURE).view(shape)


def _masked_log_softmax_rows(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Batched form of networks.masked_log_softmax: logits, mask both
    [N, H, W] (mask True = valid); softmaxes each row independently over its
    own H*W cells (matching the unbatched version's single flattened
    softmax, generalized from dim=0 to per-row dim=1)."""
    n = logits.shape[0]
    filled = logits.masked_fill(~mask, NEG_INF)
    return F.log_softmax(filled.reshape(n, -1), dim=1).reshape(logits.shape)


@dataclass
class Transition:
    kind: str  # 'aspect' | 'position'
    block_idx: int
    progress: float
    action: int
    old_log_prob: float
    temperature: float = 1.0
    occupancy: Optional[torch.Tensor] = None
    cluster_grid: Optional[torch.Tensor] = None
    wiremask: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    step_reward: float = 0.0    # this step's own causal piece (0 for 'aspect' steps)
    return_to_go: float = 0.0   # filled in after the episode ends; see collect_episode


@dataclass
class Episode:
    transitions: List[Transition]
    reward: float
    positions: list
    block_feats: torch.Tensor
    pin_feats: torch.Tensor
    b2b_edges: torch.Tensor
    p2b_edges: torch.Tensor


def canvas_scale_of(instance) -> float:
    total_area = float(instance.area_targets.clamp(min=0).sum().item())
    return max(total_area ** 0.5, 1.0)


def collect_episode(net: ActorCritic, instance, grid_dim: int, use_baseline: bool,
                     temperature: float = 1.0, greedy: bool = False) -> Episode:
    """`temperature` controls how peaked the sampling distribution is
    (annealed lower over training -- see finetune.py -- to reduce output
    variance as the policy converges); `greedy` picks each distribution's
    argmax instead of sampling from it at all (used for the final
    deterministic rollout in finetune.py, not during PPO training).

    The working canvas is sized by GridPlacementEnv itself, directly from
    this instance's own pins bounding box (see rl/env.py's
    _estimate_canvas_size) -- no scouting or separate estimation pass
    needed here."""
    env = GridPlacementEnv(instance, grid_dim=grid_dim)
    scale = canvas_scale_of(instance)
    block_feats = build_block_features(instance, scale)
    pin_feats = build_pin_features(instance, scale)

    with torch.no_grad():
        block_emb, global_emb = net.encode(block_feats, pin_feats,
                                            instance.b2b_connectivity, instance.p2b_connectivity)

    transitions: List[Transition] = []
    total_steps = max(len(env._pending), 1)
    positions = None

    try:
        while not env.done():
            step = env.current_step()
            progress = env._cursor / total_steps
            progress_t = torch.tensor(progress, dtype=torch.float32)

            if env.needs_aspect():
                with torch.no_grad():
                    logits = net.aspect_logits(block_emb, global_emb, step.block_idx, progress_t)
                    log_probs = F.log_softmax(_temp_scale(logits, temperature), dim=0)
                    action = (int(torch.argmax(log_probs).item()) if greedy
                              else torch.multinomial(log_probs.exp(), 1).item())
                transitions.append(Transition('aspect', step.block_idx, progress, action,
                                               log_probs[action].item(), temperature=temperature))
                w, h = env.choose_aspect(action)
            else:
                w, h = env.current_shape()

            cid = step.cluster_id
            occupancy_before = env.occupancy.clone()
            cluster_grid = torch.zeros_like(env.occupancy)
            if cid and env.cluster_cells.get(cid):
                for (gy, gx) in env.cluster_cells[cid]:
                    cluster_grid[gy, gx] = 1.0
            wiremask = env.wiremask(w, h)

            # A block whose chosen (extreme aspect ratio) shape no longer
            # fits anywhere is a real possibility with a stochastic,
            # partially-trained policy exploring aspect choices -- not an
            # env bug. Treat it the same as any other infeasible outcome
            # (see below) rather than letting it crash the training loop.
            result = env.position_mask(w, h)
            if result is None:
                continue  # auto-placed deterministically (cluster touch); no action taken

            mask, _, _ = result
            out_h, out_w = mask.shape
            with torch.no_grad():
                full_logits = net.position_logits(occupancy_before, cluster_grid, wiremask, block_emb,
                                                   global_emb, step.block_idx, progress_t)
                cropped = _temp_scale(full_logits[:out_h, :out_w], temperature)
                log_probs = masked_log_softmax(cropped, mask).reshape(-1)
                flat_action = (int(torch.argmax(log_probs).item()) if greedy
                               else torch.multinomial(log_probs.exp(), 1).item())

            gy, gx = divmod(flat_action, out_w)
            tr = Transition('position', step.block_idx, progress, flat_action,
                             log_probs[flat_action].item(), temperature=temperature,
                             occupancy=occupancy_before, cluster_grid=cluster_grid,
                             wiremask=wiremask, mask=mask)
            transitions.append(tr)
            env.place(gy, gx)
            delta_wl, delta_area = env.last_step_deltas
            tr.step_reward = step_quality_delta(delta_wl, delta_area, use_baseline,
                                                 instance.baseline_metrics)

        positions = env.finalize()
    except RuntimeError:
        pass  # ran out of grid space for this rollout's action sequence

    if positions is None:
        reward, violation_factor = -M_PENALTY, 1.0
    else:
        reward, violation_factor = _reward_and_violation_factor(instance, positions, use_baseline)

    # The terminal reward is MULTIPLICATIVE (-(quality_factor)*(violation_factor)),
    # but step_quality_delta only ever computes the additive quality piece --
    # so every tracked step_reward must first be rescaled by the same
    # violation_factor the terminal reward used, or the one residual
    # correction below would have to absorb the whole quality*violation
    # interaction (unbounded, scales with the entire episode's quality) onto
    # a single transition instead of just the bounded, Q-independent `-V`
    # leftover from the pretraining formula's "+1" baseline term (exactly
    # zero leftover for inference_reward, which has no such term). See
    # rl/reward.py's _reward_and_violation_factor docstring.
    for tr in transitions:
        tr.step_reward *= violation_factor

    # One residual correction makes the per-step decomposition land on
    # `reward` exactly -- it absorbs the max(0, gap) floor (step_quality_delta
    # never clips) and any cost from auto-placed cluster-touch blocks (which
    # commit a position with no matching Transition to attribute it to). See
    # rl/reward.py's step_quality_delta docstring.
    #
    # Must land on the last 'position' transition, never plain transitions[-1]:
    # if the episode's last block never got a matching position Transition
    # (auto-cluster-touch skipped it, or a RuntimeError aborted mid-attempt
    # after its aspect choice was already logged), transitions[-1] is a
    # dangling 'aspect' entry -- putting the residual there would violate
    # "aspect steps carry no immediate reward" (tests/test_ppo_credit_assignment.py).
    if transitions:
        tracked = sum(tr.step_reward for tr in transitions)
        target = next((tr for tr in reversed(transitions) if tr.kind == 'position'), transitions[-1])
        target.step_reward += reward - tracked
        running = 0.0
        for tr in reversed(transitions):
            running += tr.step_reward
            tr.return_to_go = running

    return Episode(transitions, reward, positions, block_feats, pin_feats,
                    instance.b2b_connectivity, instance.p2b_connectivity)


_worker_net: Optional[ActorCritic] = None


def _worker_init(net_kwargs: dict) -> None:
    """Pool initializer: one persistent ActorCritic per worker process, built
    once at pool startup and reused (loading a fresh state_dict per call is
    far cheaper than reconstructing the module graph every episode). Caps
    this process's own intra-op thread pool at 1 -- collect_episode's ops are
    all small (single-instance rollout, not batched training), so BLAS
    multithreading buys nothing and N workers x 8 torch threads would
    massively oversubscribe an 8-core machine."""
    global _worker_net
    torch.set_num_threads(1)
    _worker_net = ActorCritic(**net_kwargs)


def _worker_collect_episode(state_dict, instance, grid_dim, use_baseline, temperature):
    _worker_net.load_state_dict(state_dict)
    return collect_episode(_worker_net, instance, grid_dim, use_baseline, temperature=temperature)


def make_episode_pool(num_workers: int, hidden_dim: int = 64, num_gnn_layers: int = 3,
                       num_aspects: int = 9, cnn_channels: int = 16):
    """A persistent multiprocessing.Pool for collect_batch's `pool` argument
    -- created once by the caller (e.g. rl/train.py's main()) and reused
    across every iteration, since spawning fresh processes per iteration
    would eat most of the parallelism gain. Network shape must match what
    the caller's own ActorCritic was constructed with."""
    import multiprocessing
    net_kwargs = dict(hidden_dim=hidden_dim, num_gnn_layers=num_gnn_layers,
                       num_aspects=num_aspects, cnn_channels=cnn_channels)
    return multiprocessing.Pool(processes=num_workers, initializer=_worker_init, initargs=(net_kwargs,))


def collect_batch(net: ActorCritic, instance, grid_dim: int, use_baseline: bool,
                   num_episodes: int, temperature: float = 1.0,
                   deadline: Optional[float] = None, pool=None) -> List[Episode]:
    """`deadline` (an absolute time.time() value), if given, stops starting
    new episodes once passed -- so a slow per-episode rollout can't blow the
    caller's time budget by a full batch's worth of episodes (see
    finetune.py, which was previously only checking its budget between full
    iterations, letting a single iteration overshoot by 6+ episodes).

    `pool`, if given (see make_episode_pool), runs all num_episodes rollouts
    concurrently across worker processes instead of one at a time in this
    process -- collect_episode holds the GIL for negligible time relative to
    its own env-stepping/tensor work, so real OS processes (not threads) are
    needed to get wall-clock parallelism. `deadline` with `pool` only gates
    whether the whole batch is submitted, not individual episodes within it
    (they all run concurrently, so there's nothing to skip mid-batch)."""
    if pool is not None:
        if deadline is not None and time.time() >= deadline:
            return []
        state_dict = net.state_dict()
        args = [(state_dict, instance, grid_dim, use_baseline, temperature)] * num_episodes
        return pool.starmap(_worker_collect_episode, args)

    episodes = []
    for _ in range(num_episodes):
        if deadline is not None and time.time() >= deadline:
            break
        episodes.append(collect_episode(net, instance, grid_dim, use_baseline, temperature=temperature))
    return episodes


def ppo_update(
    net: ActorCritic,
    optimizer: torch.optim.Optimizer,
    episodes: List[Episode],
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    reward_approx_coef: float = 0.5,
    entropy_coef: float = 0.01,
    epochs: int = 4,
    max_grad_norm: float = 1.0,
) -> dict:
    """Batches every transition across all episodes into a handful of single
    forward calls per epoch, instead of one Python-level call per transition.
    Measured directly: the old per-transition loop was ~90% of a training
    iteration's wall time (5.98s of 6.74s on a 97-block instance), dwarfing
    episode collection -- each of the ~block_count x episodes_per_iter tiny
    forward calls pays PyTorch's per-op/autograd dispatch overhead
    independent of its (minuscule) actual FLOP count. The loss formula
    itself (clipped surrogate, value MSE, reward-approx MSE, entropy bonus)
    is unchanged; see tests/test_ppo_batched_equivalence.py for a direct
    numerical check against the original per-transition computation."""
    last_stats = {}
    for _ in range(epochs):
        optimizer.zero_grad()

        # One encode() per episode (cheap -- only num_episodes calls, not
        # num_transitions -- so left as-is). reward_so_far is a plain
        # data-derived running sum (see ScalarHead docstring), computed here
        # exactly as before; only the network calls below are batched.
        rows = []  # one entry per transition, in original iteration order
        for ep in episodes:
            block_emb, global_emb = net.encode(ep.block_feats, ep.pin_feats, ep.b2b_edges, ep.p2b_edges)
            reward_so_far = 0.0
            for tr in ep.transitions:
                rows.append(dict(tr=tr, block_emb=block_emb, global_emb=global_emb,
                                  ep_reward=ep.reward, reward_so_far=reward_so_far))
                reward_so_far += tr.step_reward

        count = max(len(rows), 1)

        global_emb_batch = torch.stack([r['global_emb'] for r in rows])
        progress_batch = torch.tensor([r['tr'].progress for r in rows], dtype=torch.float32)
        reward_so_far_batch = torch.tensor([r['reward_so_far'] for r in rows], dtype=torch.float32)
        return_batch = torch.tensor([r['tr'].return_to_go for r in rows], dtype=torch.float32)
        reward_t_batch = torch.tensor([r['ep_reward'] for r in rows], dtype=torch.float32)

        value_batch = net.value_batch(global_emb_batch, progress_batch, reward_so_far_batch)
        reward_pred_batch = net.reward_approx_batch(global_emb_batch, progress_batch, reward_so_far_batch)
        advantage_batch = (return_batch - value_batch).detach()

        value_loss_sum = ((value_batch - return_batch) ** 2).sum()
        reward_loss_sum = ((reward_pred_batch - reward_t_batch) ** 2).sum()

        policy_terms = []
        entropy_terms = []

        aspect_idx = [i for i, r in enumerate(rows) if r['tr'].kind == 'aspect']
        if aspect_idx:
            a_rows = [rows[i] for i in aspect_idx]
            block_emb_a = torch.stack([r['block_emb'][r['tr'].block_idx] for r in a_rows])
            temp_a = torch.tensor([r['tr'].temperature for r in a_rows], dtype=torch.float32)
            action_a = torch.tensor([r['tr'].action for r in a_rows], dtype=torch.long)
            old_lp_a = torch.tensor([r['tr'].old_log_prob for r in a_rows], dtype=torch.float32)
            adv_a = advantage_batch[aspect_idx]

            logits_a = net.aspect_logits_batch(block_emb_a, global_emb_batch[aspect_idx], progress_batch[aspect_idx])
            log_probs_a = F.log_softmax(_temp_scale_batch(logits_a, temp_a), dim=-1)
            new_lp_a = log_probs_a.gather(1, action_a.unsqueeze(1)).squeeze(1)
            probs_a = log_probs_a.exp()
            entropy_a = -(probs_a * log_probs_a).sum(dim=1)

            ratio_a = torch.exp(new_lp_a - old_lp_a)
            surr1_a = ratio_a * adv_a
            surr2_a = torch.clamp(ratio_a, 1 - clip_eps, 1 + clip_eps) * adv_a
            policy_terms.append(-torch.min(surr1_a, surr2_a))
            entropy_terms.append(entropy_a)

        position_idx = [i for i, r in enumerate(rows) if r['tr'].kind != 'aspect']
        if position_idx:
            # Grouped by mask shape -- always one group given the current
            # callers (one instance per collect_batch, so every position
            # transition shares the same working-canvas crop), but bucketing
            # keeps this correct if that ever changes.
            by_shape = {}
            for i in position_idx:
                by_shape.setdefault(tuple(rows[i]['tr'].mask.shape), []).append(i)

            for (out_h, out_w), idxs in by_shape.items():
                p_rows = [rows[i] for i in idxs]
                block_emb_p = torch.stack([r['block_emb'][r['tr'].block_idx] for r in p_rows])
                temp_p = torch.tensor([r['tr'].temperature for r in p_rows], dtype=torch.float32)
                action_p = torch.tensor([r['tr'].action for r in p_rows], dtype=torch.long)
                old_lp_p = torch.tensor([r['tr'].old_log_prob for r in p_rows], dtype=torch.float32)
                adv_p = advantage_batch[idxs]
                occupancy_p = torch.stack([r['tr'].occupancy for r in p_rows])
                cluster_p = torch.stack([r['tr'].cluster_grid for r in p_rows])
                wiremask_p = torch.stack([r['tr'].wiremask for r in p_rows])
                mask_p = torch.stack([r['tr'].mask for r in p_rows])

                full_logits_p = net.position_logits_batch(
                    occupancy_p, cluster_p, wiremask_p, block_emb_p,
                    global_emb_batch[idxs], progress_batch[idxs])
                cropped_p = _temp_scale_batch(full_logits_p[:, :out_h, :out_w], temp_p)
                log_probs_p = _masked_log_softmax_rows(cropped_p, mask_p).reshape(len(idxs), -1)
                new_lp_p = log_probs_p.gather(1, action_p.unsqueeze(1)).squeeze(1)
                flat_mask_p = mask_p.reshape(len(idxs), -1).to(log_probs_p.dtype)
                probs_p = log_probs_p.exp()
                entropy_p = -(probs_p * log_probs_p * flat_mask_p).sum(dim=1)

                ratio_p = torch.exp(new_lp_p - old_lp_p)
                surr1_p = ratio_p * adv_p
                surr2_p = torch.clamp(ratio_p, 1 - clip_eps, 1 + clip_eps) * adv_p
                policy_terms.append(-torch.min(surr1_p, surr2_p))
                entropy_terms.append(entropy_p)

        policy_loss_sum = torch.cat(policy_terms).sum()
        entropy_sum = torch.cat(entropy_terms).sum()

        loss = (policy_loss_sum + value_coef * value_loss_sum
                + reward_approx_coef * reward_loss_sum
                - entropy_coef * entropy_sum) / count
        loss.backward()
        # Guards against the kind of sudden PPO instability found in an
        # overnight run (fallback rate jumped from ~3% to ~29%+ within a few
        # hundred iterations, never recovering): an occasional large/noisy
        # gradient pushing the policy into a degenerate, overconfident state.
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        optimizer.step()
        last_stats = {
            'loss': loss.item(),
            'policy_loss': (policy_loss_sum / count).item(),
            'value_loss': (value_loss_sum / count).item(),
            'reward_approx_loss': (reward_loss_sum / count).item(),
            'entropy': (entropy_sum / count).item(),
            'grad_norm': float(grad_norm),
        }
    return last_stats

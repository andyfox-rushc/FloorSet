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
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from iccad2026_evaluate import M_PENALTY

from .encoder import build_block_features, build_pin_features
from .env import GridPlacementEnv
from .networks import ActorCritic, masked_log_softmax
from .reward import inference_reward, pretraining_reward, step_quality_delta


MIN_TEMPERATURE = 1e-3


def _temp_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature (floored, never truly 0) before
    softmax -- lower temperature makes the distribution more peaked. Used
    identically at collection time and update time (see Transition.temperature)
    so the PPO importance ratio always compares the same tempered
    distribution before/after a policy update, not two different ones."""
    return logits / max(temperature, MIN_TEMPERATURE)


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
    deterministic rollout in finetune.py, not during PPO training)."""
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
        reward = -M_PENALTY
    else:
        reward = (pretraining_reward(instance, positions) if use_baseline
                  else inference_reward(instance, positions))

    # One residual correction, added to the final transition, makes the
    # per-step decomposition land on `reward` exactly -- it absorbs the
    # max(0, gap) floor (step_quality_delta never clips) and any cost from
    # auto-placed cluster-touch blocks (which commit a position with no
    # matching Transition to attribute it to). See rl/reward.py's
    # step_quality_delta docstring.
    if transitions:
        tracked = sum(tr.step_reward for tr in transitions)
        transitions[-1].step_reward += reward - tracked
        running = 0.0
        for tr in reversed(transitions):
            running += tr.step_reward
            tr.return_to_go = running

    return Episode(transitions, reward, positions, block_feats, pin_feats,
                    instance.b2b_connectivity, instance.p2b_connectivity)


def collect_batch(net: ActorCritic, instance, grid_dim: int, use_baseline: bool,
                   num_episodes: int, temperature: float = 1.0,
                   deadline: Optional[float] = None) -> List[Episode]:
    """`deadline` (an absolute time.time() value), if given, stops starting
    new episodes once passed -- so a slow per-episode rollout can't blow the
    caller's time budget by a full batch's worth of episodes (see
    finetune.py, which was previously only checking its budget between full
    iterations, letting a single iteration overshoot by 6+ episodes)."""
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
    last_stats = {}
    for _ in range(epochs):
        optimizer.zero_grad()
        policy_loss_sum = torch.tensor(0.0)
        value_loss_sum = torch.tensor(0.0)
        reward_loss_sum = torch.tensor(0.0)
        entropy_sum = torch.tensor(0.0)
        count = 0

        for ep in episodes:
            block_emb, global_emb = net.encode(ep.block_feats, ep.pin_feats, ep.b2b_edges, ep.p2b_edges)
            reward_t = torch.tensor(ep.reward, dtype=torch.float32)

            for tr in ep.transitions:
                progress_t = torch.tensor(tr.progress, dtype=torch.float32)
                return_t = torch.tensor(tr.return_to_go, dtype=torch.float32)
                value = net.value(global_emb, progress_t)
                reward_pred = net.reward_approx(global_emb, progress_t)
                advantage = (return_t - value).detach()

                if tr.kind == 'aspect':
                    logits = net.aspect_logits(block_emb, global_emb, tr.block_idx, progress_t)
                    log_probs = F.log_softmax(_temp_scale(logits, tr.temperature), dim=0)
                    new_log_prob = log_probs[tr.action]
                    probs = log_probs.exp()
                    entropy = -(probs * log_probs).sum()
                else:
                    full_logits = net.position_logits(tr.occupancy, tr.cluster_grid, tr.wiremask,
                                                        block_emb, global_emb, tr.block_idx, progress_t)
                    out_h, out_w = tr.mask.shape
                    cropped = _temp_scale(full_logits[:out_h, :out_w], tr.temperature)
                    log_probs = masked_log_softmax(cropped, tr.mask).reshape(-1)
                    new_log_prob = log_probs[tr.action]
                    flat_mask = tr.mask.reshape(-1)
                    probs = log_probs.exp()
                    entropy = -(probs[flat_mask] * log_probs[flat_mask]).sum()

                ratio = torch.exp(new_log_prob - tr.old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
                policy_loss_sum = policy_loss_sum - torch.min(surr1, surr2)
                value_loss_sum = value_loss_sum + (value - return_t) ** 2
                reward_loss_sum = reward_loss_sum + (reward_pred - reward_t) ** 2
                entropy_sum = entropy_sum + entropy
                count += 1

        count = max(count, 1)
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

# The Algorithm (Goldie/Mirhoseini AlphaChip) and Our Implementation

Notes on the actual algorithm from Mirhoseini/Goldie et al., "Chip Placement
with Deep Reinforcement Learning" (the Nature paper) and Anna Goldie's
dissertation, and how our implementation in `rl/` maps to it.

## The problem

Chip placement -- deciding where every macro (memory block, compute unit)
and cluster of standard cells goes on the physical die -- is framed as a
sequential decision process. An agent places one node of the netlist at a
time onto a discretized grid canvas, largest first. Once every macro is
placed, a separate fast force-directed method places the standard-cell
clusters (grouped ~1,000-way via hMETIS graph partitioning) using simple
spring physics -- attraction proportional to connectivity weight x
distance, repulsion to avoid overlap. The RL agent never touches individual
standard cells directly.

## State representation

The netlist becomes a graph -- macros, standard-cell clusters, and ports as
nodes, nets as edges -- encoded by a GNN. Each edge gets its own learned
embedding:

```
e_ij = fc1(concat(fc0(v_i), fc0(v_j), w_ij))
```

where `w_ij` is the net's edge weight. Node embeddings update by
mean-aggregating the embeddings of incident edges, repeated for a few
rounds. The graph-level embedding is the mean over edge representations
(not node representations).

## Policy and value networks

Both take the same input: graph embedding, current macro's embedding, and a
small metadata vector (routing capacity, net/macro/cluster counts, canvas
size). The value head is just a small MLP on that vector. The policy head
is the more distinctive piece -- 5 deconvolution layers (channels
16->8->4->2->1) that *generate* a spatial probability map directly from
that compact vector, rather than convolving over an actual image of the
current layout. The generated map is then masked by a density-based
legality check (occupancy below a 60% threshold, not exact geometric
overlap) before sampling a placement.

Deviating from the paper here: our `PositionCNN` (`rl/networks.py`)
*convolves* over the actual occupancy/cluster grid rather than generating
one from a compact vector, and adds a third spatial channel -- `wiremask`
(`rl/env.py`) -- with no equivalent in the paper's design. Occupancy and
cluster_grid say where placement is legal/clustered, not where it's cheap;
without an explicit wirelength channel, the only path from "connectivity"
to "good HPWL position" was through a spatially-uniform embedding vector
broadcast identically to every grid cell, forcing the conv layers to
recover spatial HPWL structure with no direct geometric hint. `wiremask`
gives each candidate cell the weighted-Manhattan cost (the exact b2b/p2b
formula the contest scores with) of placing the current block's center
there against every already-placed connected block/pin, normalized
per-step into an attraction map. This mirrors the "wiremask"/"position
mask" fix later chip-placement RL papers (e.g. MaskPlace) made to this
same gap in the original AlphaChip design.

## Reward

Sparse -- zero every step except the last, where it's:

```
R = -Wirelength - 0.01 x Congestion
```

(HPWL-based wirelength, a routing-congestion estimate from a smoothed
demand map). Density is enforced as a hard constraint via the masking
above, not as a reward term.

Deviating from the paper here too: their terminal-only reward is standard
for macro placement, where congestion genuinely can't be attributed to a
specific earlier action. Ours doesn't have that excuse -- HPWL is a literal
sum over independent b2b/p2b edges, each fully resolved (both its
endpoints known) the instant the second one is placed, and bbox area only
ever grows monotonically as blocks are added. That means the exact same
terminal cost can be redistributed, with zero approximation, into a
per-placement-step piece (`GridPlacementEnv._step_deltas` in `rl/env.py`,
consumed by `rl/reward.py`'s `step_quality_delta`). Before this
(2026-09-09), `rl/ppo.py`'s advantage used the identical whole-episode
reward for every single transition -- with ~60-200 sequential decisions
per episode and only a handful of episodes per training iteration, PPO had
no way to tell which specific position choices caused the final
wirelength/area versus which didn't, only whether the whole trajectory was
better or worse than the value baseline predicted. This was diagnosed as
the actual bottleneck behind a training plateau that survived a separate,
also-real fix to the `wiremask` channel's normalization (see "Policy and
value networks" above) -- the position head had a strong instantaneous
HPWL signal available and still couldn't learn to use it, because nothing
in training could differentially reward acting on it well versus poorly.
`collect_episode` now gives each `Transition` a `return_to_go` (suffix sum
of its own and every later step's exact piece, plus one residual
correction on the final step for the `max(0, gap)` floor and any
untracked auto-placed-cluster-touch cost), and the value head predicts
that instead of a flat per-episode constant. No architecture or checkpoint
change was needed -- the value head's inputs are unchanged; it's just
fitting a target that now actually varies within an episode.

**2026-09-12: the reward was scoring the wrong geometry.** The real
submission (`rl/finetune.py`'s `finetune_and_solve`) always runs the raw
RL rollout through `rl/compaction.py`'s deterministic gravity-compaction
before returning it -- that's the only thing that ever closes the gap
between the RL policy's naturally spread-out placement (see "Policy and
value networks" -- `CANVAS_PADDING` gives it far more room than it needs)
and a tightly packed one. But `rl/reward.py`'s `_evaluate` scored the raw,
*pre*-compaction positions, so `area_gap` (and the HPWL contribution from
that same spread-out geometry) got zero gradient signal for the entire
project -- training could run indefinitely and the policy would never
learn to pack tighter, because the thing it was being scored on didn't
reflect what actually got submitted. Confirmed directly: `area_gap` had
been flat at ~1.12-1.22 (i.e. ~2.1-2.2x baseline area) across every
training run since the project started (see training-history memory).
Fixed by compacting inside `_evaluate` before scoring, for both
`pretraining_reward` and `inference_reward` -- now the training signal and
the actual submission are scored on the same geometry. Verified this
doesn't change any existing reward test (the hand-built test fixtures
happen to already be fully compacted, so `compact()` is a no-op on them).

**Also 2026-09-12, found while validating the above: compaction itself had
a real bug that was making `boundary_violations` *worse*, not better.**
`rl/compaction.py`'s right/top-pinned-edge handling used to move every
block sharing an edge pin by the same shared amount (the least any single
member could move alone), reasoning they "must stay flush." That's wrong
-- they don't need to move together, only to each independently land on
whatever the final bbox edge turns out to be, and forcing lockstep motion
let one more-blocked member cap how far every other member could go,
stranding it short of the edge. Verified directly on real validation
cases: post-compaction `boundary_violations` were *higher* than
pre-compaction on 13 of the first 15 cases checked (e.g. 5->8). Fixed:
`_shift_group_x`/`_shift_group_y` now compute the tightest edge the group
could jointly reach *and* the tightest edge non-group content already
occupies (the term the old version was missing entirely), and pull every
member to whichever is larger, independently. Once this no longer fights
itself, compaction's area_gap improvement got much bigger on
boundary-heavy cases too -- e.g. one validation case went from
`area_gap` 1.199 (pre-fix) to 0.642 (post-fix) purely from compaction now
being able to shrink every edge correctly.

**Also 2026-09-12: `rl/ordering.py`'s placement order had no notion of
boundary urgency.** A boundary pin (especially a corner -- two bits set)
needs a specific cell (`env.py`'s `_boundary_mask`; a corner pin is the
intersection of two edge masks, i.e. exactly one cell), so whichever block
gets there first wins it and every later block needing the same pin falls
back to an unconstrained placement (`GridPlacementEnv.boundary_fallback_count`)
-- a real, permanent soft-constraint violation, since nothing later
(including compaction) can retroactively grant it that cell. The order was
purely descending-area / connectivity-driven, with no preference for
placing these blocks while the canvas is emptiest. Fixed: unit ordering
now seeds each unit's connectivity score with a boundary-priority bonus
(corner > single edge > none -- see `_boundary_priority`), well below the
same-cluster bonus (so cluster cohesion still wins when both apply) but
enough to move boundary-critical units to the front whenever they'd
otherwise have no connectivity pulling them there. Measured on the first
15 validation cases: average `violations_relative` (boundary + grouping +
MIB, the term `compute_cost` multiplies cost by `exp(BETA * v)` with
`BETA=2.0`) dropped from ~0.34 to ~0.25, with zero regressions across the
sample.

All three fixes verified: 175/175 tests pass, and none of them touch
`rl/ppo.py`'s exact-decomposition invariant (`return_to_go` still sums to
whatever `reward` comes out to, whatever that now is).

**2026-09-13/14: after ~6200 more training iterations under the three
fixes above produced no further improvement (tracked in
floorset-training-history memory), found a fourth gap -- this time in
`rl/env.py`'s `_try_cluster_touch`, the likely dominant source of
`grouping_violations`.** It only tried one candidate offset per neighbor
per side (the neighbor's own coordinate) before falling through to
`position_mask()`'s `_adjacency_mask`, a grid-dilation fallback that uses
a full 3x3 (diagonal-inclusive) kernel -- meaning it can select a cell
that's merely grid-adjacent, even corner-only touching, to the cluster.
`position_mask()` treats that as legal, but `evaluate_solution`'s real
shapely-based connected-components check does not count it as touching.
This matches the data: `grouping_violations` were consistently several
times higher than `grouping_fallback_count` across every diagnostic run,
meaning most disconnections were happening through this "successful" but
imprecise path, not through outright fallback. Fixed by making
`_try_cluster_touch` search every grid-aligned offset along the whole
legal touching range for each neighbor/side (a new `_grid_values`
helper), not just one -- same per-candidate `_rect_overlap` safety check
as before, so this can only find more real touches, never introduce an
overlap. Not yet measured whether this moves `grouping_violations` in
practice; that's the next thing to check once training has run under it
for a while.

## The key trick: why it generalizes to new chips

Before any RL training happens, a *separate supervised pretraining phase*
runs: collect ~10,000 (placement-state, realized wirelength+congestion)
pairs by running vanilla RL across 5 real chip blocks at varying congestion
weights, then train a GNN encoder to predict that realized cost directly
from the state, as a regression task. Once that encoder is good at
predicting placement quality, its *prediction head is thrown away* and the
trained encoder becomes the initialization for the actual policy/value
networks. This is what "grounds representation learning in the supervised
task of predicting placement quality" means in the paper's own
description -- the embeddings are pre-shaped to be quality-predictive
before a single step of actual RL happens.

Then the real training loop: pretrain the full RL policy across a large,
diverse set of chip blocks (the paper's headline experiments use 20 TPU
blocks), and the central empirical result is that as you pretrain across
*more* chips, the policy gets better at rapidly generalizing to a
*previously unseen* chip block -- either placing it well in a single pass
with no further training, or converging in a handful of fine-tuning steps
rather than the hours a from-scratch run would need. That's the whole point
of the method: amortize the hard search once, across many chips, so each
new chip is cheap.

## Our version, concretely

| Piece | Their approach | Ours |
|---|---|---|
| `rl/pretrain.py` | Supervised reward-prediction pretraining -> discard head -> keep encoder | Same mechanism |
| `rl/train.py` | Pretrain across many chips (20 TPU blocks) | Pretrain across many instances from the real 1M-sample FloorSet training set |
| `rl/finetune.py` | Fast fine-tuning on a new, unseen chip at deploy time | Per-instance PPO fine-tuning inside `solve()`, since the hidden test set gives no baseline to pretrain against directly |
| Legality | Soft density threshold + downstream legalizer | Exact geometric overlap masking (must be exact -- the contest scores any overlap as instantly infeasible) |
| Macros vs. standard cells | RL places macros; force-directed method places clustered cells | Not applicable -- FloorSet gives one flat list of blocks, no macro/cell hierarchy to split |

The value of a long training run (many diverse instances, many iterations)
is a direct test of the paper's central result: the more diverse real
instances the policy sees, the better it should get at handling a *new*
one it hasn't seen. A handful of pretraining instances and a few thousand
iterations is a modest test of the idea, not a full demonstration of it --
real confirmation needs the scale the paper itself used (tens of diverse
chips, tens of thousands of episodes).

## Running from the command line

All commands run from `iccad2026contest/`, with the project's venv active:

```bash
cd /home/afox/floorplan/FloorSet/iccad2026contest
source ../venv/bin/activate
```

If that venv doesn't exist yet, see "Setting up Python and the virtual
environment" below to create it -- it's a one-time step.

### Setting up Python and the virtual environment

This only needs doing once. The venv lives at `FloorSet/venv/` (one level
above `iccad2026contest/`), separate from the system Python.

**1. Check Python is available.** Python 3.10+ works; this was set up
against 3.14.

```bash
python3 --version
```

**2. Make sure the `venv` module is actually installable.** On some
Debian/Ubuntu systems the standard library's `venv` module is split into a
separate package and isn't there by default. If step 3 fails with
"ensurepip is not available", install it (this needs `sudo`, so run it
yourself if the assistant can't):

```bash
sudo apt install -y python3.14-venv   # match your python3 --version
```

**3. Create the virtual environment and activate it:**

```bash
cd /home/afox/floorplan/FloorSet
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

Don't skip the venv and `pip install` directly with the system Python --
modern Debian/Ubuntu Pythons refuse this on purpose ("externally-managed-
environment", PEP 668), and forcing it with `--break-system-packages` risks
breaking the system Python install. The venv sidesteps the whole issue.

**4. Install the project's dependencies, plus `pytest` for the test suite:**

```bash
pip install -r iccad2026contest/requirements.txt pytest
```

This pulls in torch, numpy, shapely, matplotlib, tqdm, and requests. On a
CPU-only machine the default `pip install torch` still pulls the CUDA
wheels along with it (several GB of unused NVIDIA packages) -- harmless,
just a bigger download than strictly necessary; torch itself correctly
falls back to CPU at runtime.

**5. Verify it worked:**

```bash
python -c "import torch, numpy, shapely, matplotlib, tqdm, requests, pytest; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Should print a torch version and `cuda False` (or `True` if you actually
have a GPU set up). From here on, every session just needs:

```bash
cd /home/afox/floorplan/FloorSet/iccad2026contest
source ../venv/bin/activate
```

### Tests

```bash
pytest tests/ -q
```

### Inference (running the optimizer)

`my_optimizer.py` is the contest entry point. It loads
`checkpoints/policy.pt` automatically if present (falling back to a
from-scratch policy otherwise), then runs `rl/finetune.py`'s per-instance
PPO fine-tuning inside `solve()`.

```bash
# One validation case (0-99), with live fine-tuning progress
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0 --verbose

# All 100 validation cases (slow: ~20s x 100 by default)
python iccad2026_evaluate.py --evaluate my_optimizer.py

# Check submission format only (fast, doesn't run a full solve)
python iccad2026_evaluate.py --validate my_optimizer.py --quick
```

Per-instance fine-tuning speed/quality is controlled by `MyOptimizer`'s
constructor in `my_optimizer.py` (`time_budget`, `max_iterations`,
`episodes_per_iter`, `ppo_epochs`) -- lower `time_budget` for a faster
sweep across all 100 cases, raise it once a good checkpoint exists and you
want the best score.

### Training (`rl/train.py`)

Pretrains the shared policy: a staged encoder warm-start (`rl/pretrain.py`)
followed by the main PPO loop. Uses the real 1M-sample training set
(`floorset_lite/`) if downloaded, else falls back to cycling the 100-sample
validation set with a printed warning.

```bash
# Fresh run: staged pretraining (40 instances) + 2000 PPO iterations
python -m rl.train --iterations 2000 --episodes-per-iter 4 --ppo-epochs 4 \
  --pretrain-instances 40 --pretrain-rollouts-per-instance 4 --pretrain-epochs 5 \
  --checkpoint checkpoints/policy.pt --checkpoint-every 50

# Continue training from an existing checkpoint (skips staged pretraining --
# it already has a trained encoder)
python -m rl.train --iterations 20000 --episodes-per-iter 4 --ppo-epochs 4 \
  --resume checkpoints/policy.pt \
  --checkpoint checkpoints/policy.pt --checkpoint-every 200
```

Every checkpoint save writes two files: the canonical `--checkpoint` path
(what `my_optimizer.py` and `--resume` load), and a versioned snapshot at
`checkpoints/history/policy_iter<NNNNNN>.pt`. Keep the versioned history --
it's the only way to roll back if a run destabilizes (as one did; see the
gradient-clipping note in `rl/ppo.py`). To roll back:

```bash
cp checkpoints/history/policy_iter009000.pt checkpoints/policy.pt
```

Any history snapshot from before the `wiremask` channel was added (2026-09-08,
`position_cnn.conv1` in_channels 18 -> 19) needs migrating first or it won't
load -- run it through `scripts/migrate_checkpoint_wiremask.py`, which
expands conv1's weight tensor and zero-inits the new channel (so the
migrated checkpoint's behavior is unchanged until further training adapts
it):

```bash
python scripts/migrate_checkpoint_wiremask.py \
  checkpoints/history/policy_iter009000.pt checkpoints/policy.pt
```

For a long run, launch it detached so it survives the terminal closing, and
tee the log somewhere persistent:

```bash
nohup python -u -m rl.train --iterations 20000 --resume checkpoints/policy.pt \
  --checkpoint checkpoints/policy.pt --checkpoint-every 200 \
  > logs/train_run.log 2>&1 &
disown
```

Then check progress with `tail -f logs/train_run.log`, or watch for a
climbing `avg_reward=-10.0000` failure rate / rising `grad_norm` values as
an early warning sign of instability.

### Downloading the real training set

The 100-sample validation set (`LiteTensorDataTest/`) is small enough to
ship with the repo, but the 1M-sample training set is not -- it has to be
downloaded separately (a real one-time, ~6.6GB transfer). Without it,
`rl/train.py` still runs, but falls back to cycling the 100 validation
cases instead, which is fine for testing the pipeline but not for real
pretraining.

Run this from the `FloorSet/` repo root (one level up from
`iccad2026contest/`), not from inside `iccad2026contest/`:

```bash
cd /home/afox/floorplan/FloorSet

curl -C - -L -o LiteTensorData_v2.tar.gz \
  --retry 10 --retry-delay 5 --retry-all-errors \
  'https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/LiteTensorData_v2.tar.gz'

tar xzf LiteTensorData_v2.tar.gz -C .
rm LiteTensorData_v2.tar.gz
```

Notes:
- `-C -` makes curl resume a partial download rather than restart from
  zero -- useful since this is a large enough transfer that a connection
  reset partway through is a real possibility (it happened once while
  setting this up). If curl exits with an error, just rerun the exact same
  command; it picks up where it left off.
- The extracted data lands at `FloorSet/floorset_lite/worker_*/`, **not**
  `FloorSet/LiteTensorData/` despite what the top-level `README.md` says --
  that's a stale path in the README; the actual downloader
  (`lite_dataset.py`) uses `floorset_lite/`. `rl/train.py` already checks
  the correct path via `lite_dataset.is_dataset_downloaded()`.
- Expect ~100 `worker_*` directories totaling ~24GB once extracted.

Verify it worked:

```bash
python3 -c "
from lite_dataset import is_dataset_downloaded, FloorplanDatasetLite
print('downloaded:', is_dataset_downloaded('./'))
print('sample count:', len(FloorplanDatasetLite('./')))
"
```

Should print `downloaded: True` and a sample count around `1008000`. Once
this is done, `rl/train.py` picks up the real dataset automatically on its
next run -- no flags needed, no code changes.

## Directory structure

Everything lives under `iccad2026contest/` (in the `FloorSet/` repo root):

```
iccad2026contest/
  my_optimizer.py          # contest entry point -- MyOptimizer(FloorplanOptimizer)
  iccad2026_evaluate.py    # the contest's own evaluator (not ours -- reused, never reimplemented)
  optimizer_template.py    # the original provided baseline (B*-tree simulated annealing)
  algorithm.md             # this document
  rl/                      # our implementation
  tests/                   # pytest suite
  checkpoints/             # trained policy weights
  logs/                    # training run logs
```

### `rl/`

| File | Purpose |
|---|---|
| `data.py` | Adapts contest data (validation samples, training batches, synthetic test fixtures) into one plain `FloorplanInstance` struct used everywhere else |
| `ordering.py` | Computes placement order (a greedy, connectivity-aware walk over the b2b graph) and each block's hard-constraint role (`free` / `fixed` / `preplaced` / `mib_follower`) |
| `env.py` | `GridPlacementEnv` -- the sequential grid placement mechanics that guarantee overlap-free, exact-area, exact-fixed/preplaced-dimension, and exact-MIB-shape placement *by construction*, independent of any learned policy |
| `encoder.py` | Hand-rolled edge-weighted GNN encoder over the block/pin connectivity graph (no torch_geometric dependency) |
| `networks.py` | `ActorCritic` -- policy (aspect-ratio + position heads), value, and reward-approximation heads sharing the encoder trunk |
| `ppo.py` | Episode rollout collection and the PPO clipped-surrogate update, including temperature-annealed/greedy sampling and gradient clipping |
| `pretrain.py` | Staged encoder warm-start: supervised regression to predict realized episode reward, then the prediction head is discarded and only the trained encoder is kept (mirrors the actual paper's mechanism -- see above) |
| `finetune.py` | Per-instance PPO fine-tuning used inside `my_optimizer.py`'s `solve()`, since the hidden test set exposes no ground-truth baseline to pretrain against |
| `anneal.py` | Post-RL simulated-annealing polish (`anneal_polish`), run in `my_optimizer.py`'s `solve()` after `finetune_and_solve` returns -- locally perturbs (translate/swap/rotate) the already-feasible, already-compacted layout and keeps whatever lowers `inference_reward`'s baseline-free cost, since the RL policy places each block once, in order, and never revisits an earlier choice in light of later ones |
| `train.py` | The offline pretraining CLI documented above |

### `tests/`

One `test_*.py` file per `rl/` module, plus `conftest.py` for shared fixtures
(notably a `rollout` helper that drives an env to completion with random
actions, used to fuzz-test hard-constraint guarantees). Coverage includes:
placement ordering, hard-constraint guarantees under both synthetic
fixtures and every real validation-set instance, encoder/network shapes and
action masking, reward-formula correctness against the real evaluator,
temperature/greedy sampling behaviour, the staged pretraining warm-start,
an actual PPO overfit test (proof the policy learns, not just that the code
runs), and an end-to-end integration test through the real contest
evaluator.

### `checkpoints/`

- `policy.pt` -- the canonical "latest" checkpoint. This is what
  `my_optimizer.py` loads automatically and what `--resume` reads.
- `history/` -- versioned snapshots (`policy_iter<NNNNNN>.pt`), one saved
  alongside every periodic checkpoint during training. This is the only way
  to roll back to a specific point if a run destabilizes -- `policy.pt`
  itself is overwritten in place and has no history of its own. Kept
  indefinitely; the files are small (a few hundred KB each).

### `logs/`

One log file per training run that's been launched (named for what the run
was), left in place rather than deleted -- they're the record of what
happened, including the run that destabilized and the diagnosis that led
to the gradient-clipping fix. Watch for a climbing `avg_reward=-10.0000`
rate or rising `grad_norm` values as an early warning sign in any of these.

## Feature set and reward function -- a 2026-09-14 review

A ground-up review of exactly what's encoded and what's optimized, checked
directly against the current code (not just this doc) since the two had
drifted in one place -- see the correction at the end.

**Three tiers of input, computed at different frequencies:**

1. **Static netlist features, GNN-encoded once per instance**
   (`rl/encoder.py`). Per-block, 9-dim (`BLOCK_FEAT_DIM`):
   `[log_area, is_fixed, is_preplaced, has_mib, has_cluster, boundary_left,
   boundary_right, boundary_top, boundary_bottom]` -- purely the block's
   *constraint identity*, not its geometry or placement status. Per-pin,
   2-dim normalized `(x, y)`. These feed the hand-rolled edge-weighted GNN
   (message-passed for `num_gnn_layers=3` rounds, mean-aggregated,
   residual) into a per-block embedding (`hidden_dim=64`) plus a global
   embedding = mean over block embeddings. This encoder runs exactly once
   per episode -- it has no notion of which blocks are already placed or
   where; that's carried entirely by tiers 2 and 3.

2. **Per-step spatial channels feeding `PositionCNN`** (`rl/env.py`,
   48x48 grid by default): `occupancy` (binary, cells taken), `cluster_grid`
   (binary, current block's cluster group), and `wiremask` (the only
   channel carrying an actual wirelength signal -- see "Policy and value
   networks" above). The current block's embedding + global embedding +
   progress are projected and broadcast as a spatially-uniform extra
   channel, concatenated with the three spatial channels, then 2 conv
   layers + a 1x1 output conv produce the position logit map.

3. **Scalar context fed to every head**: `progress` = fraction of blocks
   placed so far (`env._cursor / total_steps`) -- the only explicit "how
   far along are we" signal. Aspect-ratio choice is a separate small head
   (`AspectHead`) over 9 fixed log-spaced buckets (`[0.2, 0.3, 0.45, 0.67,
   1.0, 1.5, 2.22, 3.33, 5.0]`), chosen before position for blocks that
   need one (MIB followers copy their leader's shape instead).

**Reward** (`rl/reward.py`): two modes, both built on the real
`evaluate_solution` so training reward and actual scoring always agree on
violation accounting.

- `pretraining_reward` (ground truth available: training/validation) =
  `-quality_factor * violation_factor`, uncapped -- `quality_factor = 1 +
  ALPHA * (max(0,hpwl_gap) + max(0,area_gap))` against the ground-truth
  baseline, `violation_factor = exp(BETA * violations_relative)`,
  `BETA=2.0`.
- `inference_reward` (contest time, no ground truth -- also what the new
  SA polish pass optimizes, see below) = same violation accounting, but
  quality is *absolute* `hpwl_total + 0.01*bbox_area` instead of a gap
  against an unknown baseline; flat `-M_PENALTY` if infeasible.

Both score `compact(positions)`, not the raw rollout (the 2026-09-12 fix
described above). Credit assignment is exact, not approximated: HPWL
resolves edge-by-edge and bbox area only grows monotonically, so
`_step_deltas`/`step_quality_delta` attribute the exact per-step
contribution to whichever step caused it, and PPO's advantage uses
`return_to_go` (suffix sum) instead of the flat episode reward -- the
single biggest fix in the project's history (see "Reward" above).

**Correction to this doc:** "Policy and value networks" above describes
the value/policy input as "graph embedding, current macro's embedding, and
a small metadata vector (routing capacity, net/macro/cluster counts,
canvas size)" -- that's the *paper's* design, not what's implemented here.
In this codebase, `ScalarHead`/`AspectHead` only ever see
`[block_embedding, global_embedding, progress]` or `[global_embedding,
progress]` -- no separate routing-capacity/count/canvas-size vector exists
in the code. Also, the global embedding is the mean over *block*
embeddings (`ActorCritic.encode`), not "the mean over edge
representations" as the state-representation section says. Neither is a
functional bug -- the network trains and the tests pass either way -- but
anyone touching `rl/networks.py` or `rl/encoder.py` should trust the code
over that older prose.


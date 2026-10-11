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
PPO fine-tuning inside `solve()` (2 s, a hard deadline), then builds six
candidate layouts (`rl/analytic.py`'s `best_of`: the policy layout plus five
analytic placements, each polished for 1 s with the edge/cluster repair) and
runs the compact packer (`rl/packer.py`) on the best four; the cheapest
result wins, and a wirelength relaxation (`relax_hpwl`) slides its blocks
inside their slack. Candidates and packer runs execute in a pool of worker
processes (`FLOORSET_PROCS`). About 7 s per design on average, 15 s at most
(official evaluator, 10-06 evening: weighted 1.225). See "Solve pipeline
and policy analysis, 2026-09-15 to 10-05", "Solve stage, 10-05 to 10-06"
and "Preplaced walls and wirelength relaxation, 10-06" below.

```bash
# One validation case (0-99), with live fine-tuning progress
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0 --verbose

# All 100 validation cases (~8 s x 100, ~13 min)
python iccad2026_evaluate.py --evaluate my_optimizer.py

# Check submission format only (fast, doesn't run a full solve)
python iccad2026_evaluate.py --validate my_optimizer.py --quick
```

Per-instance fine-tuning speed/quality is controlled by `MyOptimizer`'s
constructor in `my_optimizer.py` (`time_budget`, `max_iterations`,
`episodes_per_iter`, `ppo_epochs`, `anneal_time_budget` for the polish).
Raising the budgets barely helps: a 09-28 sweep on 100 designs gave 20+8 s
1.607 at 28.9 s vs 2+1 s 1.643 at 4.3 s, and the contest multiplies cost
by `max(0.7, (runtime/median)^0.3)`. `FLOORSET_CANDIDATES=""` turns off
best-of and falls back to the policy layout + slicing/re-pack polish
(1.548 at 4.5 s); `FLOORSET_PACK=0` turns off the packer, `FLOORSET_RELAX=0`
the wirelength relaxation.

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
| `compaction.py` | Deterministic gravity compaction (slide every block toward the origin, pinned blocks to their edges); `reward.py` scores the compacted layout |
| `slicing.py` | Slicing re-pack polish: rebuilds the layout as a guillotine floorplan in the policy's order, soft blocks at their exact area, then legalizes and snaps edge pins |
| `repack.py` | Constraint-graph re-pack (left/right/above/below relations from geometry, tightest coordinates) + annealing; `_fix_boundary` snaps blocks still off their required edge |
| `analytic.py` | Analytic global placer (gradient descent on smoothed wirelength + overlap/outline penalties), `build_candidate` and `best_of` (the candidate layouts) |
| `legalize.py` | `repair` (blocks past the walls revealed by pinned preplaced blocks into holes, cut-off cluster members back to their cluster), run after the edge snaps in slicing and re-pack; also an experimental Tetris legalizer (off) |
| `packer.py` | Compact packer: sequence pair from the layout, batched numpy packing and contest cost, annealing; `pack_budget` scales its time with block count |
| `anneal.py` | The original simulated-annealing polish (`anneal_polish`); replaced by `repack.py` on 09-26, still selectable with `FLOORSET_POLISH=anneal` |
| `checkpoint_meta.py` | `<checkpoint>.meta.json` sidecar: the canvas utilization the training curriculum reached, learning-rate origin, iteration |
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
  `my_optimizer.py` loads automatically and what `--resume` reads. Since
  10-05 it's iteration 4000 of the 10-04 "arm A" run
  (`ab_A/history/policy_iter004000.pt`); the previous one (09-28,
  iteration 12000) is kept as `policy_installed_0928_iter12000.pt`.
  `FLOORSET_CHECKPOINT` points the solver at a different file.
- Experiment runs write to their own subdirectories (`scratch/`, `ab_A/`,
  `exp_*/`). Some only load or only behave correctly with the environment
  switches they were trained with -- see the switch table below.
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


## Solve pipeline and policy analysis, 2026-09-15 to 10-05

Three weeks of work since the review above, condensed. The day-by-day
record is in the progress reports (`progress_report_0927.pdf`,
`progress_report_0928.pdf`, `progress_report_1003.pdf` -- the last one
covers 9/28 to 10/5). All costs below are the contest cost averaged over
the 100 validation designs (lower is better, ground truth = 1.0), without
the runtime factor.

**Where the score came from:**

| Date  | Setup                                                    | Avg cost | Runtime  |
|-------|----------------------------------------------------------|----------|----------|
| 09-22 | Policy + 20 s fine-tune + anneal polish                  | 1.913    | ~30 s    |
| 09-27 | Fine grid, curriculum, step features, re-pack polish     | 1.663    | ~28 s    |
| 09-28 | Iteration 12000 policy, 20 s fine-tune + 8 s re-pack     | 1.644    | 30-40 s  |
| 09-28 | 2 s fine-tune + 1 s slicing/re-pack, boundary snaps      | 1.552    | 4.5 s    |
| 10-04 | Best-of: policy + two analytic placements                | 1.431    | 8.2 s    |
| 10-05 | Same, policy replaced by arm A iteration 4000            | 1.424    | ~8 s     |
| 10-05 | + edge/cluster repair, grouped-shape fix in slicing      | 1.354    | 8.5 s    |
| 10-05 | + compact packer on the best candidate                   | 1.267    | 10.3 s   |
| 10-06 | 6 candidates, best 4 packed, size-scaled packer time     | 1.199    | 23 s     |
| 10-06 | Same in parallel workers (official evaluator)            | 1.212    | 7.8 s    |

Nearly all of the gain since 09-28 is in the solve stage. The policy
itself has moved the result by about 0.01 in that time.

**The pipeline as it stands:**

1. `rl/ordering.py`: connectivity walk over the blocks, boundary-pinned
   and clustered blocks pulled earlier.
2. `rl/env.py`: grid placement. The policy picks a cell on the action grid
   (48x48 in training, 32-96 by block count at solve time,
   `rl/finetune.py` `choose_grid_dim`); free space is tracked 4x finer (`DEFAULT_FINE_FACTOR=4`).
   Hard rules are code, not learned: overlap, exact areas/shapes, edge
   pins, cluster contact, an aspect mask (only shapes that fit somewhere
   are offered), `shape_fit` (a free block's exact w/h is adjusted within
   1.4x so it ends flush with its neighbours) and a lookahead check (a
   cell is rejected if the next 3 largest unplaced blocks would no longer
   fit). The canvas is sized from the pin box at a target utilization;
   training tightens it adaptively (0.60 -> 0.70, stored in
   `<checkpoint>.meta.json`) and the solver starts at the trained level and
   loosens only if every rollout fails.
3. Policy (`rl/networks.py`, ~74k parameters): GNN over the netlist, once
   per design; per step, a `PositionCNN` over occupancy + cluster-mates +
   wiremask, with the block/design embeddings and step features (progress
   plus 6 numbers about what's still to come: remaining vs free area, the
   largest unplaced block, how many of its shapes still fit) broadcast in.
   The aspect head is an MLP and sees no maps.
4. `rl/finetune.py`: 2 s of PPO on the design, then a greedy rollout. The
   budget is a hard deadline.
5. `rl/analytic.py` `best_of`: six candidates -- the policy layout and
   five analytic placements (utilization 0.75-1.0, 600 iterations) --
   each legalized and polished for 1 s by `rl/slicing.py` then
   `rl/repack.py`, both ending with `rl/legalize.py`'s repair.
6. `rl/packer.py`: the best four candidates by contest cost (no ground
   truth needed) each go through the compact packer; the cheapest wins.
   (Steps 5-6 were three candidates and no packer until 10-05; see
   "Solve stage, 10-05 to 10-06".)

**Policy experiments, and why each was dropped.** Every one was measured
on the full pipeline against the installed policy under the same
conditions, unless noted.

| Experiment (switch)                                | Result                                                    |
|----------------------------------------------------|-----------------------------------------------------------|
| Area weight x3 / x6 in training reward             | Area gap unchanged (0.350 / 0.355), greedy cost worse     |
| Train on slicing-packed reward (`SLICE_REWARD`)    | Stopped by a reboot at 13866, never scored                |
| Edge-pinned blocks placed last (`EDGE_LAST`)       | Edge violations 3 -> 14 per design; retrain arm crashed   |
| Pin box as the canvas edge (`die_anchor`)          | Area gap 0.50 -> 0.43, violations doubled, cost worse     |
| Per-design canvas ratchet (`RATCHET`)              | Tighter canvases almost never feasible; 1.687 vs 1.663    |
| Live frame, shrinks the canvas (`FRAME_UTIL=0.80`) | Area 0.25 -> 0.21, HPWL worse; retrained 1.582 vs 1.545   |
| Large blocks first (`BIG_FRAC=0.4`) + frame        | Area 0.20, HPWL 0.55; retrained 1.647 vs 1.551            |
| Same + HPWL weight x2 (`HPWL_WEIGHT=2`)            | Recovers some HPWL: 1.576 vs 1.545                        |
| From-scratch retrain, frame, 20000 iterations      | All 40 checkpoints 1.579-1.634; best 1.580 vs 1.551       |
| Whole-canvas U-Net position net (`POS_NET=unet`)   | 4000 iterations: 1.672 vs 1.528 for the current CNN       |
| U-Net + analytic target map (`TARGET_MAP=1`)       | 4000 iterations: 1.706                                    |

**What the experiments say about the policy:**

- It has no lever on area. On a typical rollout the bounding box reaches
  99% of its final size after 11% of the placements, and the final box
  equals the canvas. The area term in the reward is zero for ~89% of the
  decisions and its total is fixed by the canvas size, which is why
  weighting it more did nothing. Every setup that took the canvas away as
  a container (frame, die anchor, edge last, big-first) did shrink the
  area, then lost more on wirelength and edge violations than it gained.
  The slicing re-pack and the analytic placer get the area back after the
  fact, so the policy's area matters less than it used to.
- Raw-policy gains don't survive the solve stage. Greedy-rollout scores
  improved from 1.862 to 1.816 over the 09-28 run while the full-pipeline
  result stayed at 1.645-1.667. Pick checkpoints by the full pipeline only,
  run more than once: run-to-run noise is about +-0.01 on 100 designs and
  up to 0.5 on a single design.
- Training stops paying off at about 4000 iterations. The 10-04 three-arm
  test (fixed data seed, learning rate decayed 3e-4 -> 3e-5 within 4000
  iterations) produced a 4000-iteration policy that beat the 12000-
  iteration one (1.528 vs 1.551 without best-of; 1.424 vs 1.431 with it),
  which is why it's installed. Continuing that run from 4000 (learning
  rate restarted at 2.7e-4, stopped by a reboot at 14620) moved nothing:
  training reward -2.8 to -3.2 throughout, 100-design greedy cost in a
  1.81-1.93 band with no trend, anchor designs bouncing 1.55-1.67. The
  10-03 from-scratch run looked the same.
- The current position CNN is two 3x3 convolutions and a 1x1, so each
  cell's score sees a 5x5-cell neighbourhood. The obvious fix, a
  whole-canvas view, lost badly at 4000 iterations. It may need much
  longer training, but given the plateau above that's not a cheap bet.

Practical conclusion: more training of this network isn't worth the CPU
time. Next work is in the solve stage -- a constraint-aware legalizer for
the analytic candidates (edge pins and clusters handled exactly; vrel
0.06 -> 0.03 target) and a compact packer (sequence pair / B*-tree on the
contest cost; area gap 0.21 -> ~0.1 target). A policy idea that might
still pay off: imitation, i.e. training the policy toward best-of layouts
so one rollout lands nearer the best-of result.

**Environment switches.** All are read at import time. Defaults are what
the solver uses; the rest are experiment knobs left in for reproducibility.
A checkpoint trained with `POS_NET`, `TARGET_MAP`, `FRAME_UTIL`, `BIG_*` or
`EDGE_LAST` set needs the same settings when it's used.

| Switch (`FLOORSET_...`) | Default          | What it does                                                   |
|-------------------------|------------------|----------------------------------------------------------------|
| `CHECKPOINT`            | `policy.pt`      | Checkpoint the solver loads                                    |
| `CANDIDATES`            | base + 5 placers | Best-of candidates; empty = policy + polish only               |
| `POLISH`                | `slice+repack`   | Polish without best-of: `slice+repack`, `repack`, `anneal`     |
| `POS_NET`               | `cnn`            | Position network: `cnn` (5x5 view) or `unet` (whole canvas)    |
| `TARGET_MAP`            | `0`              | Analytic-placer target as a second wiremask channel            |
| `FRAME_UTIL`            | off              | Live frame utilization (e.g. 0.80); `FRAME_GROW`, `FRAME_SLIDE` |
| `BIG_FRAC`              | `0`              | Share of blocks (by area) moved early in the ordering          |
| `BIG_WEIGHT`            | `3`              | Strength of that move (x the boundary bonus)                   |
| `BIG_MODE`              | `seed`           | `seed`, `connected` or `scale`                                 |
| `EDGE_LAST`             | `0`              | Place edge-pinned blocks after the interior                    |
| `KNOWN_DIMS`            | `0`              | Size the canvas from dimensions revealed by preplaced blocks   |
| `HPWL_WEIGHT`           | `1.0`            | Training reward wirelength weight                              |
| `AREA_WEIGHT`           | `1.0`            | Training reward area weight                                    |
| `SLICE_REWARD`          | `0`              | Training reward on the slicing-packed layout (seconds budget)  |
| `RATCHET`               | `0`              | Solve-time canvas tightening step after a feasible batch       |
| `SNAP`                  | `reshape`        | Edge-snap mode in `repack.py`                                  |
| `SLICE_MODES`           | `tree,shift`     | Slicing pre-modes (also `clamp`, `inplace`)                    |
| `SLICE_NONEG`           | `0`              | Reject slicing candidates past the origin (worse; kept off)    |
| `SLICE_PULLBACK`        | `0`              | Slicing pull-back variant (no net gain)                        |
| `REPAIR`                | `0.3`            | Share of each polish budget for `legalize.repair`; 0 = off     |
| `OUTLINE_WALLS`         | `0`              | Placer outline = true width/height from preplaced pins (worse) |
| `LEGALIZER`             | `graph`          | Placer legalizer; `tetris` = experimental (worse)              |
| `PACK`                  | `1.5`            | Packer seconds per candidate at 77 blocks (x (n/77)^2); 0 = off |
| `PACK_TOPK`             | `4`              | Candidates packed                                              |
| `PACK_TARGET`           | `0.5`            | Share of "move next to" steps aimed at a wired/cluster partner |
| `PACK_BATCH`            | `32`             | Neighbours scored per packer step                              |
| `PACK_T`                | `1.0`            | Packer start-temperature multiplier                            |
| `PACK_EDGE`             | `0`              | Edge-thinning move share (worse)                               |
| `PACK_WALL`             | `0`              | Search penalty past preplaced walls (no gain)                  |
| `PROCS`                 | min(6, cpus)     | Worker processes for candidates and packer; 1 = in-process     |

**Corrections to the 09-14 review above:** the position grid is 48x48 only in
training; at solve time it's 32-96 cells by block count, and occupancy is
tracked on a 4x finer grid. Every head now also gets the 6 step features
alongside `progress`.

## Solve stage, 10-05 to 10-06

**Scoring note.** The official total isn't a plain average: each design is
weighted by `exp(blocks / 12)` (README, "Scoring"), so the 116-120-block
designs carry about a third of it. All results from here on are given as
this weighted score too (`scripts/polish_bench.py` and
`scripts/packer_bench.py` print it). Weighted: 9/28 report 1.619, 10/5
report 1.466, 10/6 1.233 (official evaluator; 1.208 when the same solver
ran sequentially -- run-to-run noise is about +-0.01-0.02).

**Diagnosis first.** `scripts/bestof_diag.py` records every candidate's
soft violations after every stage. On the 100 best-of winners of 10/5:
112 edge violations (71 of them preplaced blocks -- they can't move, so
they're off their edge whenever the layout sticks out past it), 100
cluster and 44 grouped-shape (MIB) violations. The MIB ones were all made
by `rl/slicing.py`, which gave a group's soft members a shape of their own
instead of the fixed member's.

**Repair (`rl/legalize.py`).** The ground-truth bbox starts at the origin,
so pinned preplaced blocks reveal walls (left/bottom pin: x/y >= 0;
right/top pin: the true width/height). `repair` moves every block past a
wall into free holes inside the walls at once (one at a time can't help:
the pin only counts once the last block is out) and reattaches cut-off
cluster members beside their cluster's largest component, keeping a move
only if the contest cost drops. With the MIB fix: 1.424 -> 1.354 (mean).
The preplaced-wall violations mostly remain: the constraint-graph
legalizer resolves overlaps by pushing blocks outward, past the wall
(design 2: slicing root exactly 102 wide = the wall, legalized 105-118).
Tried and dropped: a pair-relation bias toward stacking (fits the width by
growing much taller), stacking past-wall blocks on top (new violations), a
Tetris legalizer for the placer (tighter but worse wirelength: 1.432 vs
1.368), the true outline in the placer (1.362 vs 1.354).

**Compact packer (`rl/packer.py`).** A sequence pair (G+, G-) represents
any packing: a before b in both -> a left of b; after in G+ but before in
G- -> a below b. The start comes from the layout itself: G- orders pairs
that are left-of with overlapping y-ranges, below with overlapping
x-ranges, or fully lower-left; G+ the same with above/upper-left. Every
relation it implies is true of the layout, so packing it reproduces the
layout or something smaller (100/100 designs). Picking one relation per
pair by the larger gap -- the obvious way -- makes the orders cyclic and
scrambled 80/100 starts. Packing is the longest path in G- order: the
blocks before position k are k's only possible predecessors, left of it if
also earlier in G+, else below it. Preplaced blocks keep their spot (a
conflict makes that sequence pair infeasible); right/top-pinned blocks
with nothing beyond them slide out to the edge. numpy has no fast per-SP
loop, so each step scores 32 neighbours together ([batch, n] arrays) and
takes the best under an annealing rule. Moves: swaps in one or both
sequences, reshaping a soft unit (MIB groups together), and "move a next
to b" (half the time b is a wired or cluster partner of a). The fast cost
is the contest formula relative to the start layout with exact boundary
and grouping counts; the result is re-checked with the real evaluator and
returned only if better.

Results (mean of 100 designs unless marked weighted):

| Change                                             | Result                                 |
|----------------------------------------------------|----------------------------------------|
| Packer on the best-of winner, 1.5 s                | 1.354 -> 1.267                         |
| Pack the best 3 candidates x 1.5 s                 | 1.238 (best one x 4.5 s: 1.253 cached) |
| 6 candidates (placer utilization 0.75-1.0), best 4 | 1.205; weighted 1.252                  |
| Packer time x (n/77)^2 instead of flat             | weighted 1.250 -> 1.229 (cached)       |
| Placer 600 iterations instead of 250               | weighted 1.233 -> 1.226 (cached)       |
| Parallel worker processes                          | same quality, 23 s -> 7.8 s            |

Not helpful: batch 16/64, start temperature x0.1/x10, an edge-thinning
move, a search penalty for layouts past the preplaced walls (the true
outline has ~3% white space; pulling into it costs more wirelength than it
saves), nine candidates instead of six (1.205 vs 1.207, +3 s), unpolished
candidates (1.337 vs 1.248), half the polish (1.262 vs 1.248).

**Parallel workers.** `my_optimizer.py` keeps a forkserver pool for the
candidate builds and packer runs. Pool functions must live in `rl/`: the
contest evaluator imports `my_optimizer.py` under another module name, so
anything defined there can't be pickled (the first attempt failed every
design this way). Any pool failure falls back to running in-process. Each
stage has a wall-clock budget, so parallel processes do less work each (on
this laptop's 4 performance + 4 efficiency cores the packer makes 1113
steps in 3 s alone, ~955 with 4 at once, ~520 with 6), but on the 12
largest designs parallel (1.230-1.234) and sequential (1.218-1.238) runs
are within noise, and doubling the budgets in parallel didn't help either.

**Where it stands (10-06).** Weighted 1.233, mean 1.212, HPWL gap 0.17,
area gap 0.15, vrel 0.02, all feasible, 7.8 s per design. The largest
designs are still the worst (mean 1.24 for 100+ blocks) and weigh the
most; 30 s of packer time only gains 0.02-0.08 on them, so the next gain
needs better moves (aimed at specific violations and critical paths) or a
packer that builds inside the preplaced walls, not more time. The runtime
factor is unknown until the official run: if the field is much faster,
`FLOORSET_PACK` and the candidate count are the knobs.

## Preplaced walls and wirelength relaxation, 10-06

**Where the remaining cost is.** On the official 10-06 run the weighted
excess over ground truth (0.233) splits into HPWL 0.094, area 0.084 and
violations 0.055. The violations left are 60 preplaced blocks off their
edge, 21 movable blocks off their edge, 17 cluster breaks, no MIB.
Pinned preplaced blocks reveal walls of the true outline. Nine designs
reveal both width and height (12, 31, 53, 57, 61, 69, 88, 89, 99: 16% of
the weight, a quarter of the excess) and our layouts overshoot them; 28
more reveal one side and overshoot it.

**Why fitting the outline is hard.** Block area divided by the true
outline is 0.96-1.006 on those nine designs: the ground truth has 0-4%
white space (our layouts ~15%). What was tried, all on the official
layouts:

| Attempt                                                       | Result                                                     |
|---------------------------------------------------------------|------------------------------------------------------------|
| Slicing at the true outline, then `rl/repack._legalize`       | raw overlap only 2-8%, but the legalizer blows it up (88: ~300 x 600) |
| Slicing leaves -> sequence pair -> bottom-left pack           | feasible on 12 and 57 only (12: 1.296 -> 1.151 with 5 s packer); hard blocks overflow their slices and hit preplaced blocks on the rest |
| + slack-based reshaping of soft blocks on the critical path    | stalls: both axes over the outline, nothing left to trade  |
| + continuous optimizer (torch: x, y, aspect under the sequence pair, outline penalty) | can't satisfy overlap and outline together at ~1% white space |
| Packer with a wall penalty (x5, x50) for 15 s                 | still 15-20% past the outline, cost not better             |

Exact fitting needs either a much better arrangement than the slicing /
packer starts give or a real fixed-outline floorplanner; none of the
prototypes went into the solver.

**Wirelength relaxation (`rl/packer.py relax_hpwl`, default on).** The
packer's bottom-left packing puts every block at the low end of its slack.
`relax_hpwl` keeps the final layout's sequence pair and bbox and moves each
block, one axis at a time, to the weighted median of what it connects to
(the exact optimum on that axis with everything else fixed), clamped to
its slack window. Preplaced, edge-pinned (on their axis) and cluster blocks
stay put, so no violation can appear; the result is kept only if the
contest cost drops. On the 100 official 10-06 layouts: weighted 1.2335 ->
1.2308, 98/100 designs better, ~46 ms per design. Letting cluster members
move too: 1.2320 (cluster breaks cost more than the wirelength gained).

Official evaluator with the relaxation (`official_eval_1006b.json`):
weighted 1.2250, mean 1.2008, HPWL gap 0.151, area gap 0.146, vrel 0.021,
100/100 feasible, 7.2 s average, 15.1 s max (morning run without it:
1.2335 / 1.212 / 7.8 s; the gap beyond the 0.003 measured on fixed layouts
is run-to-run noise). Anchors, median of 3 (`results_report_1006b`): 0
1.370, 49 1.315, 98 1.129.

**Origin rule (`FLOORSET_ORIGIN=1`, off by default).** 78 of 100 official
layouts had blocks below / left of (0, 0), outside the box the pins sit on
(designs 0 and 49 in the report). The policy never does it; the placer's
`_legalize_grow` and the slicing re-pack lower the floor when blocks don't
fit in front of a preplaced block. The switch forbids it everywhere
(placer legalizer without floor lowering, slicing NONEG, best-of ranks
overhanging candidates last, packer packs from the origin). All 100 layouts
stay in the box, but the official score drops 1.2250 -> 1.2653
(`official_eval_1006c_origin.json`): 73 designs worse, mostly 100+ blocks
(design 94 HPWL gap 0.18 -> 0.39). Repacking finished layouts from the
origin is worse still (1.2478, 34 never fit).

**Leaderboard (10-06, `leaderboard_1006.csv`), sorted by cost.** On raw
score (weighted cost before the runtime factor) our 1.225 places 6th of 11,
0.141 behind the best entry (1.084). The leaders are also faster (0.1-1.3 s
per design vs our 7.2 s); the pipeline can be sped up, but the focus is
quality: closing the cost gap.

## Big designs and the bounding box, 10-07

Quality over runtime (user, 10-06/07). The 21 designs with 100+ blocks are
83% of the weighted score.

| Change (official evaluator unless noted)                  | Weighted | Area gap | Time per design |
|-----------------------------------------------------------|----------|----------|-----------------|
| 10-06 solver                                              | 1.2250   | 0.146    | 7 s             |
| Packer x27 (`FLOORSET_PACK=40`) + squeeze                 | 1.1352   | 0.086    | 45 s            |
| + origin repair (`FLOORSET_ORIGIN`, default on)           | 1.1318   | 0.079    | 45 s            |

Big-design bench (`scripts/big_bench.sh`, weighted total with the rest from
the 10-06 run): packer 1.5 / 6 / 15 / 40+squeeze -> 1.225 / 1.192 / 1.170 /
1.147, still improving. Packing all six candidates at once is worse (CPU
contention on 4P+4LP cores). Packer speed-ups (merged nets, pin nets via
sorted prefix sums, blocks x batch layout) give 1.8x steps on design 99 but
no measurable score change.

`squeeze` (rl/packer.py): keeps the sequence pair; soft units on the
height-critical path are widened into their x-slack (vice versa for width),
all variants packed exactly, best kept while the cost drops.

`_origin_repair` (rl/packer.py): when a start overhangs (x or y < 0), pack
from the origin; blocks that run into a preplaced block they are left of /
below are moved to another side of it (or next to a random block when
stuck), until nothing conflicts (81/81 official layouts). The search then
stays inside, and my_optimizer prefers in-box results. Official: 0/100
layouts below/left of the origin (was 81).

**Pin box (top / right edges).** The pins sit on the true outline, so no
block of the ground truth ends past the pins' highest x / y; 52/100 of our
layouts did (up to 24%), mostly from a wrong overall shape (design 80: 172 x
178 against a 139 x 240 box; blocks fill only 69-93% of the pin box, so
there is room). Changes, default on: `rl/slicing.py` also tries roots with
the pin box's aspect (`FLOORSET_SLICE_PINASPECT`), `rl/packer.py` penalizes
running past the box (`FLOORSET_PINBOX=3`, per sqrt(area)), tracks the best
in-box layout first and prefers in-box results, and `my_optimizer.py` picks
in-box layouts first. Neither the penalty nor a box-aware `squeeze` alone
fixes an already-packed layout (the right-pinned column would have to move
together; width-critical paths run through preplaced blocks).

| Official, 100 designs        | Weighted | HPWL gap | Area gap | vrel  | Outside pin box |
|------------------------------|----------|----------|----------|-------|-----------------|
| Origin repair only           | 1.1318   | 0.105    | 0.079    | 0.014 | 52              |
| + pin box (default)          | 1.1482   | 0.121    | 0.068    | 0.020 | 8 (1-10%)       |

Kept as default by the user's choice (layout inside the box over the last
0.016 of score). Penalty 10 fits 2 more of the last 7 but costs much more
(design 4: 1.315 -> 1.536).

## Shape fitting, learned placer, packer 100 -- 10-07 / 10-08

**The ground truth scores 1.108 weighted** on the contest evaluator (219
edge-pin and 10 cluster violations over 100 designs; only its wirelength and
area define 1.0). The packer started from the exact ground truth reaches
1.016, so the optimizer is good enough; the ground truth's arrangement with
soft shapes off by ~20% only reached ~1.24 with the packer alone.

**A -- `rl/sizing.py fit_shapes`.** Keeps a sequence pair and optimizes every
soft unit's log aspect (MIB groups share one) by Adam on a differentiable
packing (longest paths over the left-of / below relations, preplaced fixed),
loss = contest quality term + edge-pin + preplaced-collision + cluster-gap
penalties; snapshots are packed exactly (`_Batch.pack`) and scored with the
real evaluator. GT arrangement + shapes x exp(N(0, 0.2)) -> fit (1000 steps,
lr 0.02) -> packer 6 -> squeeze -> relax: **1.048 weighted, 100/100 feasible**
(`gt_bound_shape0.2_fit.json`, `scripts/gt_arrangement_bound.py FIT=1`).

**B -- `rl/learned_placer.py`.** PlacerNet = NetlistEncoder (15 block
features incl. preplaced centres and sizes in pin-box units; edge weights
mean-normalized; 6 layers x 128) + MLP head -> block centre in [0, 1]^2 of
the pin box and log aspect (bounded to 1/8..8). Trained on the training
set's ground truth (`fp_sol` = w, h, x, y) with L1 losses
(`scripts/train_placer.py`, ~1,900 designs/min on 2 threads; val centre L1
0.92 -> 0.061 in 7 h). `to_layout`: sequence pair from the predicted centres
(G+ by x - y, G- by x + y), predicted shapes, packed from the origin;
`rl/analytic.py learned_candidate` resolves preplaced collisions (fit_shapes,
else `_origin_repair`) and fits shapes. Default best-of candidate `learned`,
always packed (`best_of(force=...)`). Placer checkpoint: `FLOORSET_PLACER`
(default `checkpoints/placer.pt`).

| Official, 100 designs                | Weighted | HPWL gap | Area gap | vrel  |
|--------------------------------------|----------|----------|----------|-------|
| 10/7, in-box within 5%, packer 40    | 1.1358   | 0.116    | 0.068    | 0.017 |
| C: packer 100                        | 1.1232   | 0.110    | 0.062    | 0.018 |
| B: + learned candidate (snapshot)    | 1.1027   | 0.076    | 0.056    | 0.015 |
| + second checkpoint ("learned2")     | 1.0924   | 0.058    | 0.055    | 0.014 |
| + second training run ("learned3")   | 1.0795   | 0.041    | 0.055    | 0.015 |

Ground truth as scored 1.108; leaderboard leader 1.084.

Two checkpoints of one training run (placer_a.pt at ~290k designs, val
0.067; placer_b.pt at 910k, val 0.060) make different mistakes: alone ~1.21
each with packer 6, the better of the two per design 1.164; the final one
alone is not better than the snapshot. Both are candidates (`learned`,
`learned2`, env `FLOORSET_PLACER` / `FLOORSET_PLACER2`), both always packed.


## Honest rewards, block relations -- 10-09

**Reward alignment** (`scripts/reward_alignment.py`, 34 designs x 8 sampled
arrangements of placer_d, centre noise 0.02 / aspect 0.2): within-design rank
correlation with the true final cost (fit 1000 + packer 6 + squeeze + relax):
raw repaired layout 0.38, + fit 200 0.43, + fit 200 + 1 s packer 0.54 (pick
regret 0.029 vs 0.098 for the plain prediction; best of 8 beats the prediction
on 26/34). The 10/8 RL fine-tune (`scripts/rl_placer.py`, raw reward) failed
for that reason. Sample-and-rank in the solver (`FLOORSET_PLACER_SAMPLES`,
default 1 = off): official 1.0801 vs 1.0815 -- noise; the packer at budget 100
evens out variants of one arrangement.

**Larger placer** (256 x 8): stalled (outputs saturated) until
`NetlistEncoder(norm=True)`: messages divided by (edge-weight sum + 1) and
LayerNorm per layer (`placer_d.pt`, val centre L1 0.052).

**Relation head** (`PlacerNet(rel=True)`): for block pairs (i, j) an MLP on
[hi + hj, hi - hj, hi * hj] -> {i left of j, j left of i, i below j, j below
i}, labels from the ground truth's sequence pair (2048 sampled pairs per
design, weight 0.5); `rel_sequences` ranks blocks by their summed probability
of coming first in G+ / G- -> sequence pair. Trained from placer_d
(`train_placer.py --norm --rel --resume`): val relation accuracy 0.906 after
~50k designs; centre predictions also improved (packer-6 test 1.185 -> 1.167).
Arrangement from relations 1.159 (better on 100+ blocks), from centres 1.167
(better on smaller designs) -> both are candidates (`PLACER_FILES` entries
`(file, use_rel)`).

| Official, 100 designs                              | Weighted | HPWL gap | Area gap |
|----------------------------------------------------|----------|----------|----------|
| three placers (10/8 report)                        | 1.0795   | 0.041    | 0.055    |
| + refit / larger network / sample-and-rank (runs)  | 1.080-1.082 | 0.040 | 0.051-0.055 |
| placer A + relation net (relations, centres)       | 1.0727   | 0.034    | 0.054    |
| same, relation net trained to 312k designs (10-10) | 1.0575   | 0.032    | 0.044    |

## More search and a second relation network, 10-11

| Official, 100 designs                                     | Weighted | HPWL gap | Area gap | Beats GT |
|-----------------------------------------------------------|----------|----------|----------|----------|
| 10-10 default (packer 100)                                | 1.0575   | 0.032    | 0.044    | 66       |
| relation net trained on to 358k designs (`placer_rel3`)   | 1.0709   | 0.029    | 0.047    | 66       |
| 10-10 + second relation net (`placer_relB`) as learned4   | 1.0596   | 0.026    | 0.048    | 69       |
| same + packer budget 200 (new default)                    | 1.0465   | 0.022    | 0.036    | 72       |

Longer training lowered validation error (0.050 -> 0.049) but not the
official score: validation error doesn't track the final cost. Run-to-run
variation is large (best of three runs per design: 1.046 at packer 100),
so more search pays; runtime 110 -> 209 s per design. The pin box tightened
to right/top-pinned preplaced edges (`FLOORSET_WALLBOX`, the true outline;
40 of the 58 remaining violations are such blocks just past it) was neutral
on 12 large designs and stays off.

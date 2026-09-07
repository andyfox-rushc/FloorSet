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

## Reward

Sparse -- zero every step except the last, where it's:

```
R = -Wirelength - 0.01 x Congestion
```

(HPWL-based wirelength, a routing-congestion estimate from a smoothed
demand map). Density is enforced as a hard constraint via the masking
above, not as a reward term.

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


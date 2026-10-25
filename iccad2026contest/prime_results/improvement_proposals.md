# FloorSet-Prime: improvement proposals (Team ICB, 10/24)

Research pass over `rl/window.py`, `rl/refine.py`, `scripts/lns_final.py`, the 10/24 LNS logs, the wirelength LNS round now running (`logs/lns_wl_r1.log`), and nine short experiments (prototypes `scripts/*_proto.py`, outputs `prime_results/proto_lns/`, `prime_results/proto_pocket/`). Experiments ran 2 processes at a time beside the 6-process LNS batch, so absolute speeds are those of a loaded machine; every comparison below is like-for-like (same start layout, same seconds, same seed unless stated).

## Bottom line

1. The LNS spends 75% of its time in the greedy descent after each window anneal, and 73% of that in three-block re-splits (`recut3`) re-checked over the whole layout. A "don't-look bits" greedy (skip a move whose blocks haven't changed since it last failed) cuts greedy time 2.7–3.1× with the same number of wins. Together with a 1 s anneal and windows of at most 14 blocks, the LNS got 3× the tries and 31% more HPWL gain in 600 s (76% more in the first 150 s) on 4 large designs: better on 3, tied on 1. That's the cheapest real gain, roughly a 50-line change.
2. LNS is far from saturated on most designs. The running 900 s round gained a further 1.1% HPWL (mean) on the 25 designs (n ≥ 85) that never had LNS and 0.3–1.0% on designs that already had one round. Only designs 76 and 75, which got the most CPU (10 h on 76), are flat (0.04–0.05% in 900 s).
3. LNS from one partition flow's layout catches up with LNS from the best-of-six-flows + polish layout within 600 s (2 designs tested). That points to a much faster pipeline: one flow + LNS, instead of 16 h of flows before the LNS.
4. GT-free time allocation: proportional to the contest weight, plus "keep going while a soft violation remains", would have cost about +0.0004 weighted on the 10/24 run. `lns_final.py` without `--stop-at-gt` doesn't work as a rule today: its min-gain test never triggers, so it would run for days.
5. Robustness: two published layouts (designs 23 and 73) meet an edge pin only through a 1e-6-high sliver; LNS on 73 merged it away and got a boundary violation (1.000 → 1.080). The pocket swap succeeds only in the smallest window (4 of 72 anneals; 0 of 101 in larger ones), so its seeds should go there.
6. A learned window chooser is still a reasonable RL target, but the signals I could measure so far are weak (window size is the only clear one). It should come after items 1–4, with a held-out protocol fixed in advance.

## Where things stand

- `final_1024.json`: all 100 at 1.000; HPWL gap mean −3.1%, contest-weighted −1.6% (weight = exp((n−120)/12)). The 20 designs with n ≥ 101 carry 81% of the weight; designs 99, 98, 97 alone carry 22%.
- Running wirelength round (`lns_wl_r1`, 900 s, seed 20, from `final_1024.json`), 54 designs done when this was written: weighted gap over those designs −1.61% → −2.48%. Gains on large designs:

| Design | n   | Gap before | After 900 s | Gain  | Note                              |
| ------ | --- | ---------- | ----------- | ----- | --------------------------------- |
| 99     | 120 | −4.21%     | −4.60%      | 0.39% | no earlier LNS                    |
| 98     | 119 | −0.51%     | −0.81%      | 0.30% | earlier LNS (stopped at GT)       |
| 97     | 118 | −0.06%     | −1.04%      | 0.98% | earlier LNS (stopped at GT)       |
| 95     | 116 | −1.45%     | −3.49%      | 2.04% | no earlier LNS                    |
| 93     | 114 | −1.83%     | −4.06%      | 2.23% | no earlier LNS                    |
| 84     | 105 | −3.41%     | −6.20%      | 2.79% | no earlier LNS                    |
| 76     | 97  | −0.11%     | −0.16%      | 0.05% | 10 CPU-h of LNS before: flat      |
| 75     | 96  | −2.12%     | −2.16%      | 0.04% | 1 win in 127 tries: flat          |
| 73     | 94  | −2.19%     | −3.41%      | 1.22% | not usable: edge pin lost, see R2 |

  Fresh designs (25 so far): mean gain 1.08%, median 0.78%. Previously LNS'd designs: 97 0.98%, 87 0.47%, 77 0.45%, 86 0.42%, 83 0.40%, 98 0.30%, 80 0.28%, 81 0.26%, 76 0.05%, 75 0.04%.

## Where the time goes

### The full rebuild (21 h)

| Stage                            | Wall clock | Final layouts it supplies                |
| -------------------------------- | ---------- | ---------------------------------------- |
| Packer flow + refine (steps 1–2) | 2.7 h      | none since 10/24                         |
| Four partition flows (steps 3–6) | 12.1 h     | all 100 (61 from the RL-tree flow alone) |
| Best-of + 60 s polish (step 7)   | 1 h        | —                                        |
| LNS stage, stop at GT (step 8)   | ~5 h       | 17 designs changed                       |

Per design, the RL-tree flow alone takes about 220 s wall on 6 processes at n = 120 (`partition_full_refine_tree_merged.json` runtimes).

### Inside one LNS try (cProfile, design 99, 6 tries, `prime_results/proto_lns/profile_99.txt`)

| Part                             | Seconds | Share |
| -------------------------------- | ------- | ----- |
| Window anneal (3 s each)         | 18.1    | 20%   |
| Greedy descent, total            | 74.4    | 80%   |
| Greedy: `recut3` (19,341 calls)  | 54.1    | 58%   |
| Greedy: `recut` (19,470 calls)   | 10.6    | 11%   |
| Greedy: `move_rigid` (4,943)     | 7.6     | 8%    |
| Greedy: `pairs()` scans          | 1.8     | 2%    |

Design 99 has 222 touching soft pairs and 928 triples; each greedy re-scans all 928 triples about 3.5 times (triples run whenever a pair pass finds nothing, then the pass repeats if a triple won). Low-level cost is numpy overhead on tiny arrays: `snap` 18 s, `hp_local` 11 s, `soft_viol` 8 s, `thin` 8 s, `split` 7 s, `badness_cached` 6 s of the 92 s.

Over all 505 traced tries (8 runs, designs 91/93/97/99): the anneal result alone beat the current layout in 3 of 505 tries. Every win comes from the greedy descent after the anneal, so the LNS is "big structured kick + descent". 13% of tries got no valid partition or a rejected shape (5% of the time).

### What the LNS curves say about diminishing returns

- Within one 600 s chain, gains are front-loaded: by quarter, 0.81/0.32/0.16/0.11% (design 93), 0.49/0.37/0.04/0.01% (97), 0.41/0.20/0.26/0.00% (91).
- But a fresh seed from the same start finds about as much again: design 97 gained 0.91% (seed 7, 600 s) and 0.98% (seed 20, 900 s) from the same layout. Their results can't be combined, though (see N1).
- Only after many CPU-hours does a design go flat: 76 (≈10 CPU-h) gained 0.05% in 900 s.
- Window size matters: windows of 6–14 blocks won 18–32% of tries at 7–9 s/try; 15–20 blocks won 8–12% at 10–11 s/try, about half the HPWL gain per second.

## Ranked proposals

Ranked by expected gain per unit of effort. "Margin" = weighted HPWL gap below the ground truth; "GT-free" = quality without the ground truth steering; "runtime" = time to a given quality.

| #   | Proposal                                        | Helps               | Evidence                                  | Effort    | Risk   |
| --- | ----------------------------------------------- | ------------------- | ----------------------------------------- | --------- | ------ |
| 1   | Don't-look-bits greedy in LNS (and refine)      | runtime, margin     | measured: 2.7–3.1× greedy, same wins      | ~0.5 day  | low    |
| 2   | LNS settings: 1 s anneal, windows ≤ 14 blocks   | runtime, margin     | measured with 1: +31% gain, 3 of 4 better | 1 line    | low    |
| 3   | One flow + LNS instead of six flows + polish    | runtime             | measured on 2 designs                     | 1–2 days  | medium |
| 4   | GT-free allocation: by weight + soft violations | GT-free             | estimate from 10/24 logs: +0.0004         | ~0.5 day  | low    |
| 5   | Robustness fixes (slivers, pocket seeds, guard) | GT-free, robustness | measured                                  | ~0.5 day  | low    |
| 6   | numba port of the pair/triple re-cut kernels    | runtime             | profile: numpy overhead dominates         | 2–3 days  | medium |
| 7   | Learned window chooser (contextual bandit / RL) | margin, runtime     | weak signals so far                       | 1–2 weeks | high   |

### 1. Don't-look-bits greedy

What: in `Refiner._greedy`, remember for each pair / rigid pair / triple the clock at which it was last tried without gain, and skip it until one of its blocks changes. Changed blocks get a new stamp: the window's blocks after an anneal, the blocks of every accepted move, and the blocks that differ when a losing try is reverted.

Evidence (paired test, `scripts/greedy_dlb_proto.py`): from the same post-anneal state each try runs both greedies; 25 tries each on designs 99 and 93.

| Design | Greedy time, full | Greedy time, DLB | Speed-up | Wins full / DLB | Per-try score, DLB − full      |
| ------ | ----------------- | ---------------- | -------- | --------------- | ------------------------------ |
| 99     | 205.6 s           | 75.9 s           | 2.7×     | 5 / 6           | median 0, mean +0.00018        |
| 93     | 216.8 s           | 69.7 s           | 3.1×     | 14 / 13         | median +0.00002, mean +0.00017 |

LNS tries per second went from 0.089 to 0.166 (99) and 0.086 to 0.173 (93), with the 3 s anneal. The mean score difference comes from losing tries (DLB ignores HPWL shifts from net neighbours that moved); wins are unaffected. A cheaper variant that keeps the full rescan but restricts it to the window's touch-and-net halo saved only 15–25% (the net halo covers most of the layout), so the bits are what matter.

Also applies to `Refiner.run` (polish/ILS), `fix_violations` and every flow's refine stage, which use the same `_greedy`.

Risk: small quality loss per greedy; a final full greedy pass before returning removes it.

### 2. LNS settings: 1 s anneal, windows of at most 14 blocks

Evidence: the anneal output is never kept as is (3 of 505); the descent does the work. With a 1 s anneal (same seed, 600 s): design 99 gained 0.94% vs 0.40%, design 97 0.92% vs 0.91%. Windows of 15–20 blocks give about half the gain per second of 6–14 (above).

A/B of the combination (DLB + 1 s anneal + kmax 14, "tuned") against the current LNS, 600 s each, seed 5, from `final_1024.json`:

| Design | Weight | Tries base / tuned | Wins base / tuned | HPWL gain base | HPWL gain tuned | Gain in first 150 s, base / tuned |
| ------ | ------ | ------------------ | ----------------- | -------------- | --------------- | --------------------------------- |
| 95     | 5.7%   | 56 / 215           | 14 / 26           | 1.91%          | 2.74%           | 0.60% / 2.13%                     |
| 94     | 5.3%   | 53 / 129           | 12 / 16           | 0.77%          | 1.33%           | 0.31% / 0.97%                     |
| 92     | 4.5%   | 54 / 122           | 4 / 15            | 0.35%          | 0.59%           | 0.21% / 0.43%                     |
| 90     | 3.8%   | 45 / 153           | 8 / 26            | 2.07%          | 2.00%           | 1.29% / 0.71%                     |
| Sum    |        | 208 / 619          | 38 / 83           | 5.10%          | 6.66%           | 2.41% / 4.24%                     |

Tuned won on 3 designs and tied on 1 (90: 2.00% vs 2.07%); 3× the tries, 2.2× the wins, +31% HPWL gain at 600 s and +76% in the first 150 s, which is what a contest budget of minutes would see. All 8 outputs valid with no soft violations.

Single-seed runs are noisy (two seeds of the same LNS on one design differ by up to 2×), so treat a per-design comparison as weak and the sum over designs as the signal.

### 3. One partition flow + LNS instead of six flows + polish (runtime)

Evidence (600 s LNS, same seed, current code):

| Design | Start layout                    | Gap at start | Gap after 600 s LNS             |
| ------ | ------------------------------- | ------------ | ------------------------------- |
| 99     | RL-tree flow only               | −2.79%       | −4.43%                          |
| 99     | best-of-6 + polish (final_1024) | −4.21%       | −4.59%                          |
| 93     | RL-tree flow only               | −1.33%       | −3.84%                          |
| 93     | best-of-6 + polish (final_1024) | −1.83%       | −3.21% (worklist greedy −3.72%) |

The 0.5–1.4% head start that 16 h of flows bought is mostly gone after 600 s of LNS on one core. Producing the start layout costs ~220 s × 6 processes for the tree flow alone vs ~580 s × 6 for all flows plus polish (per design at the 21 h rebuild's rate). A contest-mode pipeline would be: RL-tree partition flow (fewer restarts or a shorter budget) → `fix_violations` → LNS for the rest of the time on all cores, with seeds as parallel chains and the best kept.

What it would take: a full 100-design run of that pipeline with a fixed per-design budget (e.g. 300 s and 900 s wall), compared with `final_1024.json` on weighted gap and count at 1.000.

Risk: the tree flow had soft violations on several big designs (98, 97, 81, 80, 77, 76) that best-of covered with other flows; the repair moves must fix those from the tree layout. A second flow (plain partition with refine) as a fallback start for violated designs is cheap insurance.

### 4. GT-free stopping and allocation (for the contest run and step 8)

Facts:
- `lns_final.py` without `--stop-at-gt` keeps a design while its relative cost drops by more than `--min-gain` 1e-4 per round. In the data, every design still gains 0.3–2% HPWL per 900 s round, so nothing stops: 6 rounds at seeds 1,1,2,4,4,4 is ~10 CPU-h per design-line, ~1,000 CPU-h for 100 designs.
- Before LNS, big designs (n ≥ 100) (final_1023.json) sat at a mean gap of −0.8% with a standard deviation of 2.0%, and 7 of 21 were above GT: any of them can be, and our own HPWL is the only estimate of where GT is. So without GT the right target is "most weighted margin in the time available", not "stop at 1.000".

Rule proposed: total LNS budget B; each design gets B × w_i / Σw (with a floor of one round); a design with a soft violation left keeps going (as now); spare time from finished designs is reassigned by weight.

Estimate on the 10/24 stage (16 CPU-h, GT-steered): the same 16 CPU-h by weight gives 98 1.2 h, 97 1.1 h, 87 0.5 h, 86 0.4 h, 83 0.34 h, 81 0.29 h, 80 0.26 h, 76 0.19 h, 75 0.18 h. Reading each design's cost at that time off the 10/24 round logs, 83, 81 and 80 would end near 1.001, 76 near 1.02–1.03 on its partition line, 48 near 1.02; weighted sum ≈ +0.0004 (1.0004 instead of 1.0000), most of it from 76. All other time goes to margin on the big designs, which is what a contest run needs.

What it would take: change the round loop in `lns_final.py` to a budget split by weight (~40 lines), then measure it on the full rebuild instead of `--stop-at-gt`.

### 5. Robustness fixes

R1. Pocket swap (design 50). `scripts/pocket_trials_proto.py`, 6 seeds over all partners and window sizes (~77 anneals, ~15 min each run):

| Search penalty                  | Anneals | No valid partition | Valid | All pins met |
| ------------------------------- | ------- | ------------------ | ----- | ------------ |
| none (as now)                   | 77      | 11                 | 66    | 3            |
| shaped 0.5 (pin + cluster gaps) | 78      | 19                 | 59    | 1            |

All 4 successes were in the smallest window (factor 2.0: 24–25 blocks + 2 pocket blocks): 4 of 72 there, 0 of 101 in the 30–38-block windows. Fix: spend the seeds on the smallest window (and try factors 1.0–1.5), stop at the first success. At ~5.5% per anneal, 50 anneals give ~94% success. The shaped penalty doesn't help.

R2. Edge pins met through a sliver. Designs 23 (block 7, bottom edge) and 73 (block 22, right edge) in `final_1024.json` touch their edge only through a zero-area spike 1e-6 high (block 22: x 143.79→154 at y 35.267074–35.267075). Prime's `utils.check_boundary_const` and our evaluator both count it (line intersection), so the score holds, but a stricter checker wouldn't. The LNS round on 73 merged the spike away (Refiner.layout's 1e-5 coordinate merge) and its output scored 1.080. Fixes: (a) treat a block as on its edge only if its contact length is ≥ 1e-3 (in `Refiner.soft_viol`, `_edge_gap`, the window costs); (b) re-run the edge-pin repair on 23 and 73; (c) in `lns()`, check each accepted state through `layout()` + the evaluator's soft count (a few calls per run), so a run never ends worse than it started. `merge_layout.py --by-hpwl` rejects such outputs already, so nothing bad reaches a result file today; it costs the run.

R3. LNS acceptance rate (3.4% overall on 10/24; 8–32% here on fresh designs). This is mostly a symptom of how far a design has been pushed, not a defect: 76 went 5 wins in 177 tries after 10 CPU-h. Proposals 1 and 2 raise wins per second; proposal 7 targets the rate itself.

### 6. numba port of the re-cut kernels (runtime)

The greedy's cost is per-call numpy overhead on arrays of a few rectangles (`snap`, `split`, `thin`, `badness_cached`, `hp_local`, `soft_viol` together ~60 of 92 s). `rl/partition_kernels.py` already holds numba kernels for the rectangle-set operations; extending it with one kernel per move type (pair re-cut: 4 candidate cuts with area split, sliver and leaf checks, HPWL delta) should give another 3–5× on the greedy on top of proposal 1. With 1 + 2 + 6, an LNS try would drop from ~11 s to ~1.5–2 s.

Risk: shape checks (`shape_ok` uses shapely) stay in Python; keep them on the accepted candidate only, as now.

### 7. Learned window chooser (RL / contextual bandit)

Idea: score candidate windows (k random windows per step) with a model predicting P(win) × gain / time; pick the best, keep ε-random exploration. Trained from LNS traces, which `scripts/lns_trace_proto.py` already writes (window blocks, size, time, scores before/after, win).

Evidence so far:
- Window size alone explains a 2× difference in gain per second (above): a learnable signal, but one a fixed kmax already captures.
- A hand-made prior (centre block drawn by weighted distance from its net-weighted ideal spot, half the time): design 99 0.85% vs 0.40%, design 97 0.91% vs 0.91% (single seeds; inconclusive).
- The kick policy (the same idea for pair kicks) lost to random kicks on held-out designs (10/22).

What it would take, with an honest protocol:
1. Start layouts on ~60–100 training designs with n ≥ 90 from `../PrimeTensorData` (RL-tree flow, ~200 s × 6 processes each ≈ 3–6 h).
2. LNS traces on them (900 s each ≈ 3–5 h on 6 processes; ~100 tries per design, 5–10k labelled windows). Per-window features: size, blocks' summed displacement from their ideal spots, internal vs crossing net weight, rigid count, thin-block count, tries and wins near it so far.
3. Fit a small gradient-boosted model or MLP; check AUC on held-out training designs first. Go on only if AUC ≥ 0.65.
4. Final test on the 100 test designs never used in training: paired runs, equal wall time, ≥ 3 seeds each, the 20 largest designs; report wins/losses per design.

Expected gain: unknown; if it doubles the win rate it's worth about as much as proposal 1. Risk: high (the kick policy precedent; LNS outcomes depend on the anneal's randomness).

## Negative or neutral results (don't redo)

- N1. Combining independent LNS runs from the same start: 600–900 s of LNS changes 83–104 of design 99's 120 blocks (greedy descents ripple across the layout), so no two runs have disjoint changes to merge (`scripts/merge_runs_proto.py`: 0 pieces transplanted on 99 and 97). Parallel seeds can only be used as best-of.
- N2. Window-local greedy with a touch + net halo: only 15–25% faster than the full greedy (the net halo covers most blocks), quality the same within noise (99: 0.26% vs 0.40%; 97: 1.13% vs 0.91%). Superseded by proposal 1.
- N3. Shaped penalty in the pocket swap: 1 success in 78 anneals vs 3 in 77 without.

## Untested ideas (cheap to try next)

- Late-acceptance or record-to-record acceptance in `lns()` (accept a try within 0.05% of the current score, keep the best seen), since single chains stall after ~300 s while fresh seeds keep finding gains.
- Anneal time 0 (random slicing tree of the window, straight to the descent), given that the anneal never wins on its own.
- A GT-HPWL predictor for the stopping rule: needs σ well under the ~2% spread of our pre-LNS gaps to beat "allocate by weight"; low priority.

## Files

- Prototypes: `scripts/lns_trace_proto.py` (instrumented LNS; options `--greedy full|local|worklist|dlb`, `--anneal-s`, `--kmax`, `--pick random|ideal`), `scripts/greedy_dlb_proto.py` + `scripts/greedy_dlb_lib_proto.py` (paired DLB test), `scripts/greedy_profile_proto.py`, `scripts/pocket_trials_proto.py`, `scripts/merge_runs_proto.py`, `scripts/lns_trace_stats_proto.py` (trace summaries).
- Outputs: `prime_results/proto_lns/` (traces `*.jsonl`, summaries, layouts, `profile_99.txt`, `dlb_99.json`, `dlb_93.json`), `prime_results/proto_pocket/`.

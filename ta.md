# Canvas sizing review: pin-box die boundary

Date: 2026-09-24
Code: `iccad2026contest/rl/env.py` (`GridPlacementEnv._estimate_canvas_size`, `reset`)

## Summary

The proposal was to set the default die boundary (the canvas) to the smallest
box that covers the boundary pins' maximum x/y coordinates. That box is a
correct die boundary, but it doesn't work as the placement canvas. Two ways of
using it were tried and both did worse, so the previous sizing is restored:

- shape from the pin box
- area = total block area / `UTILIZATION_TARGET` (0.5)

## 1. The pin box as a die boundary

Measured on all 100 validation instances. The box runs from (0,0) to
(max pin x, max pin y).

| Measure                                 | Result                     |
|-----------------------------------------|----------------------------|
| Pins start at (0,0)                     | 100/100                    |
| Pin box contains the real die           | 100/100                    |
| Pin box area ÷ real die area            | 1.03–1.41× (median 1.22×)  |
| Per-axis ratio, pin max ÷ real die max  | 1.00–1.20× (median ~1.11×) |
| Block area ÷ pin box area (utilization) | 0.69–0.93 (median 0.79)    |
| Fewest valid pins in any instance       | 50                         |

So the box never undershoots the real die, and it's always the same shape
as the pin bounding box the previous code already used. Because the pins
start at (0,0), max − min equals max.

## 2. Attempt A: the pin box as the hard canvas

`_estimate_canvas_size` returned `(max pin x, max pin y)` directly.

- **Test result:** 101 of 193 tests failed.
  - All 100 validation random-placement stress tests failed with
    `No free position for block N ... canvas too small`, even after 30
    retries with different random seeds.
  - `test_data.py::test_real_training_rollouts_respect_hard_constraints`
    also failed.
  - The other 92 tests passed.
- **Cause:** the pin box's utilization is about 0.79, roughly the size of the
  finished, tightly packed layout. Placing blocks one at a time with no
  lookahead needs more room while the layout is being built, especially at
  48×48 grid resolution. This is the same failure seen earlier at
  utilization 0.7 (99 of 193 tests failed).
- **Impact:** at contest time `greedy_fallback_positions` would widen the
  canvas and recover. During training, most rollouts would fail, the same
  pattern behind the earlier −10 average-reward flatline.

## 3. Attempt B: the pin box as the die, inside a padded canvas

Design:
- The die is the pin box. The working grid extends right and up from it
  until blocks fill half its area (utilization 0.5), keeping the die's shape.
- The die's right/top edges fall exactly on a grid line (`die_cells`,
  typically 38 of the 48 cells). That keeps the existing no-overlap
  guarantees when blocks snap to the edge.
- Blocks pinned to the right or top edge anchor to the die edge instead of
  the canvas edge. This covers the placement mask, the forced coordinates,
  the cluster-touch checks and the final snap to the edge.
- No extra reward term was needed. The contest scores boundary constraints
  against the final layout's bounding box, and the reward is computed on the
  compacted layout, so area outside the die is already penalized.

Test result: 193/193 passed.

### Comparison to the previous version

Seeded random placements, 100 validation instances × 3 seeds. Both canvases
have the same area (median 1.94× the real die area).

| Metric                                                     | Previous (shape/area) | Attempt B |
|------------------------------------------------------------|-----------------------|-----------|
| Pinned blocks off their required edge                      | 1360/7209 (18.9%)     | 3987/7209 (55.3%) |
| Pinned blocks placed without the edge constraint (no room) | 648                   | 816       |
| Final bounding box ÷ real die area (median)                | 1.770                 | 1.825     |
| Average seed retries to finish a placement                 | 1.31                  | 1.75      |

**Cause:** free blocks spill into the extra space past the die, so blocks
pinned to the die edge no longer touch the edge of the finished layout. The
cleanup pass (`rl/compaction.py`) only moves blocks toward the origin, so it
can't pull a right/top-pinned block back out to the layout's final edge.

**Caveat:** random placement is a pessimistic stand-in for a trained policy,
and training would push blocks back inside the die. Still, the policy would
start from almost three times as many edge violations, and it did worse on
every measure.

## 4. Decision and current state

- The previous shape/area sizing is restored. It reproduces the baseline
  numbers above exactly and passes 193/193 tests.
- Existing checkpoints were trained on this canvas, so no retrain is needed.
- **Not committed:** HEAD still has the older square
  `CANVAS_PADDING = 1.5` version of `rl/env.py`. The shape/area sizing only
  exists in the working tree, alongside other uncommitted changes in that
  file. Commit it soon.

## 5. Lessons and possible follow-up

- A size that covers the finished optimal layout isn't enough for
  one-block-at-a-time construction. This is the fourth independent
  confirmation in this project, after `CANVAS_PADDING`, `PIN_SAFETY_MARGIN`
  and `UTILIZATION_TARGET`.
- The pin box already sets the canvas shape and is a trustworthy die. It
  just shouldn't be the hard canvas edge.
- **Possible follow-up:** let the cleanup pass move right/top-pinned blocks
  outward to the finished layout's edge, then retry Attempt B. This touches
  the cleanup pass's no-overlap guarantees and needs its own testing.
- **Possible follow-up:** if the die is ever used again, give the policy's
  CNN a channel showing where the die is. It currently can't see the die
  edge. Adding a channel means migrating the checkpoint.

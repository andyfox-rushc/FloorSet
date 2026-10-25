# Full rebuild, 10/24–10/25

`bash scripts/reproduce_final.sh prime_results/repro_full`, started 10/24 12:38 with the 10/24 code: faster LNS (windows of 6–14 blocks, 1 s anneals, don't-look bits), the 1e-3 edge-contact rule, the refiner output fix and the 12-seed pocket swap.

## Result

All 100 designs at 1.000 (weighted 1.0000), the same as the published `final_1024.json`, with no hand-run steps. Total wall clock 12 h 39 min (steps 1–7: 11 h 33 min; step 8 rerun: 1 h 6 min), against the ~21 h expected.

| Step | Output                       | Rebuild | Published | At 1.000 | Wall clock    |
| ---- | ---------------------------- | ------- | --------- | -------- | ------------- |
| 1    | `packer.json`                | 1.2318  | 1.2247    | 0        | 1 h 41 min    |
| 2    | `packer_refined.json`        | 1.1427  | 1.1425    | 0        | 50 min        |
| 3    | `partition.json`             | 1.1953  | 1.2915    | 0        | 2 h 13 min    |
| 4    | `partition_refined.json`     | 1.0187  | 1.1462    | 50       | 50 min        |
| 5    | `partition_refine.json`      | 1.0199  | 1.1229    | 46       | 2 h 42 min    |
| 6    | `partition_refine_tree.json` | 1.0071  | 1.1077    | 75       | 2 h 43 min    |
| 7    | `final.json`                 | 1.0016  | 1.0054    | 86       | 34 min        |
| 8    | `final_lns.json`             | 1.0000  | 1.0000    | 100      | 1 h 6 min     |

Weighted quality cost (ground truth = 1.000, lower is better). All files 100/100 feasible. Steps 3–6 are 0.10–0.13 better than the published runs, which had 1–2 failures each; the likely cause is the refiner output fix (layouts no longer lose edge or cluster contacts when converted back), though this wasn't isolated.

## Step 8 and design 50

The first step-8 run (`final_lns_v1.json`, 3 h 45 min) ended at 1.0001: design 50 kept its edge-pin violation (1.0571) through all six rounds. Cause: the 10/24 change to `pocket_fix` (rl/window.py) returned the first anneal that met block 9's pin even when its score after the greedy descent was worse than the start; under the rebuild's settings that first one scored 1.068 against 1.057, the caller rejected it, and no further seeds were tried. Fix (10/25): stop early only on a result with fewer violations and a lower score, otherwise keep the best. In 6 of 6 standalone runs it then fixed design 50 with no violations left (0.989–1.002). Step 8 was rerun with the fix: design 50 reached 1.000 in round 1, and only design 76 needed a second round.

## Checks

- `scripts/verify_results.py` re-scores all 8 files (numbers above).
- Strict edge contact (1e-3 rule) and round trip through the refiner: no misses or failures on `final_lns.json`.
- 18 tests pass (test_refine, test_partition, test_window, test_prime_evaluate).

## Against the published files

| File                         | Weighted | At 1.000 | Mean HPWL gap | Within 1% of GT | Closest to GT |
| ---------------------------- | -------- | -------- | ------------- | --------------- | ------------- |
| `final_1024.json`            | 1.0000   | 100      | −3.13%        | 29              | −0.05%        |
| rebuild `final_lns.json`     | 1.0000   | 100      | −3.24%        | 23              | −0.04% (29)   |
| `final_1024_wl.json`         | 1.0000   | 100      | −4.06%        | 13              | −0.16% (76)   |
| `final_1025.json`            | 1.0000   | 100      | −4.17%        | 9               | −0.16% (76)   |

`final_1025.json` is `final_1024_wl.json` with 14 designs taken from the rebuild where its wirelength is shorter (0, 5, 6, 12, 25, 36, 48, 50, 77, 80, 83, 86, 97, 98), each re-scored and with the rebuild's own run time. It passes the strict edge-contact and round-trip checks.

Origins in the rebuild's final file: RL tree flow 77 (67 direct, 10 after LNS), partition + refine 16 (12, 4 after LNS), partition + refine in restarts 7.

Step 8 still stops a design at contest cost 1.000 (`--stop-at-gt`), which uses the ground truth to allocate time only.

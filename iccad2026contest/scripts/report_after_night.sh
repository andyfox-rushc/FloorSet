#!/bin/bash
# 10-21 night, last steps (unattended): after the full refined run, polish the best layout
# per design and build prime_report_1022.pdf.
cd "$(dirname "$0")/.."
until [ -f prime_results/partition_full_refine.done ]; do sleep 60; done
CHOICE=$(cut -d' ' -f3 prime_results/night_full.choice 2>/dev/null)
LBL="Partition + refine in restarts"; [ "$CHOICE" = "tree" ] && LBL="Partition + refine + tree policy"
FLOWS=(--flow "Packer flow=prime_results/contour_pack20.json")
[ -f prime_results/contour_pack20_ils.json ] && FLOWS+=(--flow "Packer + refine=prime_results/contour_pack20_ils.json")
FLOWS+=(--flow "Partition (t80, cache fixed)=prime_results/partition_full_t80b.json")
[ -f prime_results/partition_t80b_refined.json ] && FLOWS+=(--flow "Partition + refine=prime_results/partition_t80b_refined.json")
FLOWS+=(--flow "$LBL=prime_results/partition_full_refine.json")
FLOORSET_PARTITION_MEMO=20000 ../venv/bin/python scripts/final_polish.py "${FLOWS[@]}" --seconds 60 --jobs 3 \
  --out prime_results/final_1022.json > prime_results/final_1022.log 2>&1
../venv/bin/python scripts/generate_prime_report_1022.py --out prime_report_1022.pdf "${FLOWS[@]}" \
  --flow "Final (best of all + polish)=prime_results/final_1022.json" \
  --panels "Partition (t80, cache fixed)" > prime_results/report_1022.log 2>&1
echo finished > prime_results/report_1022.done

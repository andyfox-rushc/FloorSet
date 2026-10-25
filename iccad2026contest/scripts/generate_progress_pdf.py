#!/usr/bin/env python3
"""Progress report PDF: 09-22 full run vs the current pipeline.

Pages: summary, anchor designs (table + layout images), this week's QoR
work, per-case cost chart, full 100-case table, white-space diagnosis and
next steps. matplotlib PdfPages only (no extra dependency).

  python scripts/generate_progress_pdf.py --new full_eval_iter10000.json \
      --anchors results_checkpoints/iter10000 --out progress_report_0927.pdf
"""
import argparse
import json
import statistics as st
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from iccad2026_evaluate import (ContestEvaluator, calculate_bbox_area,  # noqa: E402
                                calculate_hpwl_b2b, calculate_hpwl_p2b)

ANCHORS = [0, 49, 98]
W, H = 11, 8.5            # landscape letter

WORK = [
    ("Finer occupancy grid + shape mask",
     "Free space is tracked on a grid 4x finer than the action grid, so blocks no longer "
     "waste space rounding up to whole cells; the policy only picks shapes that fit somewhere. "
     "Random rollouts that finish: 81% -> 100% on the old canvas."),
    ("Tighter canvas, adaptive schedule",
     "Canvas utilization raised from 0.50 to 0.60, then an adaptive schedule tightens it "
     "during training (now 0.70, probing 0.725) and fine-tuning falls back to a looser canvas "
     "if a design can't finish. Area gap tracks canvas size: 0.42 at 0.65 -> 0.35 at 0.70."),
    ("Fine-tune reward normalization",
     "Per-design fine-tuning rewarded failed rollouts more than feasible ones (absolute "
     "wirelength); rewards are now measured against a reference layout."),
    ("Dead-end lookahead + shape fitting",
     "A cell is rejected if the next 3 largest blocks would no longer fit anywhere; free "
     "blocks without edge pins stretch (same area) to sit flush with neighbours. More "
     "rollouts finish at every canvas size (e.g. 52% -> 67% at 0.70) at equal cost."),
    ("Training data shuffling (bug fix)",
     "Training read the 1M-design set in file order: ~112 same-size designs in a row, and "
     "every run restarted at the first file, so only the same ~20k designs were ever seen. "
     "Now shuffled across files with a mixed buffer."),
    ("Stable training at tight canvases",
     "Failed rollouts are scored by how far they got (-5..-10, not a flat -10), advantages "
     "are normalized, the learning rate is 3e-4 decaying (was fixed 1e-3), and tighter "
     "canvases are probed with 25% of iterations instead of all at once."),
    ("Lookahead inputs to the network",
     "The policy and value network now see six numbers about what is still to come "
     "(remaining area vs free space, largest remaining block, whether it still fits)."),
    ("Re-packing polish (replaces annealer)",
     "Keeps the RL arrangement, re-packs every block as tight as it allows, then anneals "
     "moves/reshapes with a fast copy of the contest cost. 100 designs, same inputs: "
     "cost 1.707 -> 1.644 (old annealer: 1.690)."),
    ("Final 20,000-iteration training run",
     "Training finished 20,000 iterations. Every 1,000-iteration checkpoint was scored on the "
     "100 designs; the top 5 then ran the full pipeline on a quiet machine: iter 12000 1.644, "
     "19000 1.648, 15000/16000 1.654, 20000 1.667. Iter 12000 is installed. Better raw-policy "
     "scores after 12000 did not survive fine-tuning + polish."),
    ("Engineering",
     "Solver pinned to 1 thread (8 threads made it 50x slower under load); every training "
     "checkpoint is now scored automatically on the 100 designs with images for 0/49/98."),
    ("Tried and rejected",
     "Pin box as the hard canvas (rollouts rarely finish); anchoring edge blocks to the pin "
     "box (area -0.07 but edge violations x2.3); shifting time from polish to fine-tuning; "
     "per-design canvas ratchet (cost 1.663 -> 1.687); edge-pinned blocks placed last (cost "
     "1.83 -> 3.99); 3x/6x area weight in the training reward (area gap 0.350 -> 0.350/0.355)."),
]


def text_page(pdf, title, paragraphs, subtitle=None):
    fig = plt.figure(figsize=(W, H))
    fig.text(0.06, 0.93, title, fontsize=18, weight="bold")
    if subtitle:
        fig.text(0.06, 0.895, subtitle, fontsize=10, color="0.35")
    y = 0.85
    for head, body in paragraphs:
        if head:
            fig.text(0.06, y, head, fontsize=11, weight="bold")
            y -= 0.028
        for line in textwrap.wrap(body, 125):
            fig.text(0.08, y, line, fontsize=9.5)
            y -= 0.024
        y -= 0.014
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, title, header, rows, col_widths=None, subtitle=None, fontsize=8.5, highlight=None,
               decorate=None):
    fig, ax = plt.subplots(figsize=(W, H))
    ax.axis("off")
    fig.text(0.06, 0.94, title, fontsize=16, weight="bold")
    top = 0.88
    if subtitle:
        lines = textwrap.wrap(subtitle, 150)
        fig.text(0.06, 0.905, "\n".join(lines), fontsize=9.5, color="0.35", va="top")
        top = 0.89 - 0.022 * len(lines) - 0.02  # keep the table below the subtitle
    tbl = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
                   colWidths=col_widths, bbox=[0.0, 0.0 if len(rows) > 20 else 1 - 0.06 * (len(rows) + 1), 1.0,
                                               1.0 if len(rows) > 20 else 0.06 * (len(rows) + 1)])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#dfe6ee")
            cell.set_text_props(weight="bold")
        elif highlight and highlight(r - 1, c):
            cell.set_facecolor("#e3f2e3")
    fig.subplots_adjust(left=0.05, right=0.95, top=top, bottom=0.05)
    if decorate:        # optional extra drawing on the finished table (fig, table)
        decorate(fig, tbl)
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf, title, path, caption):
    fig = plt.figure(figsize=(W, H))
    fig.text(0.06, 0.94, title, fontsize=16, weight="bold")
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.83])
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    fig.text(0.06, 0.04, caption, fontsize=9.5, color="0.3")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="my_optimizer_results.json")
    ap.add_argument("--new", default="full_eval_iter012000.json")
    ap.add_argument("--new-key", default="20:repack:8")
    ap.add_argument("--anchors", default="results_20260928")
    ap.add_argument("--label", default="iteration 12000")
    ap.add_argument("--out", default="progress_report_0928.pdf")
    a = ap.parse_args()

    old = {r["test_id"]: r for r in json.load(open(ROOT / a.old))["test_results"]}
    new = {int(k): v for k, v in json.load(open(ROOT / a.new))[a.new_key]["per_case"].items()}
    anchors = {r["design"]: r for r in json.load(open(ROOT / a.anchors / "results.json"))}
    ev = ContestEvaluator(str(ROOT.parent) + "/", verbose=False)
    ev._load_dataset()

    ids = sorted(set(old) & set(new))
    avg = lambda d, k: st.mean(d[i][k] for i in ids)
    med = lambda d, k: st.median(d[i][k] for i in ids)
    wins = sum(new[i]["cost"] < old[i]["cost"] for i in ids)
    old_feas = sum(bool(old[i]["is_feasible"]) for i in ids)
    new_feas = sum(bool(new[i]["feasible"]) for i in ids)

    with PdfPages(ROOT / a.out) as pdf:
        # 1. Summary
        rows = [
            ["Average cost", f"{avg(old, 'cost'):.3f}", f"{avg(new, 'cost'):.3f}",
             f"{100 * (avg(new, 'cost') / avg(old, 'cost') - 1):+.1f}%"],
            ["Median cost", f"{med(old, 'cost'):.3f}", f"{med(new, 'cost'):.3f}",
             f"{100 * (med(new, 'cost') / med(old, 'cost') - 1):+.1f}%"],
            ["Avg HPWL gap", f"{avg(old, 'hpwl_gap'):.3f}", f"{avg(new, 'hpwl_gap'):.3f}", ""],
            ["Avg area gap", f"{avg(old, 'area_gap'):.3f}", f"{avg(new, 'area_gap'):.3f}", ""],
            ["Avg soft violations (vrel)", f"{avg(old, 'violations_relative'):.3f}", f"{avg(new, 'vrel'):.3f}", ""],
            ["Feasible", f"{old_feas}/{len(ids)}", f"{new_feas}/{len(ids)}", ""],
            ["Designs improved", "", f"{wins}/{len(ids)}", ""],
            ["Avg runtime per design (s)", f"{avg(old, 'runtime_seconds'):.1f}", f"{avg(new, 'secs'):.1f}", ""],
        ]
        table_page(pdf, "RL Floorplanner progress 9/22 - 9/28",
                   ["Metric (100 validation designs)", "9/22 full run", f"Now ({a.label})", "Change"], rows,
                   col_widths=[0.4, 0.2, 0.2, 0.2], fontsize=11,
                   subtitle="Full contest pipeline on all 100 validation designs. Cost = (1 + 0.5 (HPWL gap + area gap)) "
                            "x exp(2 vrel); 1.0 = matches ground truth. Local runtime factor = 1.0 (neutral). "
                            "The new full run solved 8 designs in parallel, so its runtimes include contention; "
                            "see the anchor page for single-design runtimes.")

        # 2. Anchor designs table
        rows = []
        for t in ANCHORS:
            o, n = old[t], anchors[t]
            sample = ev.dataset[t]
            area_t, b2b, p2b, pins, _ = sample["input"]
            pos = [tuple(p) for p in o["positions"]]
            o_hpwl = calculate_hpwl_b2b(pos, b2b) + calculate_hpwl_p2b(pos, p2b, pins)
            o_area = calculate_bbox_area(pos)
            rows.append([f"{t}", f"{n['blocks']}", "9/22", f"{o_hpwl:.1f}", f"{n['hpwl_gt']:.1f}",
                         f"{o['hpwl_gap']:.3f}", f"{o_area:,.0f}", f"{n['bbox_area_gt']:,.0f}",
                         f"{o['area_gap']:.3f}", f"{o['violations_relative']:.3f}", f"{o['cost']:.3f}",
                         f"{o['runtime_seconds']:.1f}"])
            rows.append(["", "", "now", f"{n['hpwl']:.1f}", f"{n['hpwl_gt']:.1f}", f"{n['hpwl_gap']:.3f}",
                         f"{n['bbox_area']:,.0f}", f"{n['bbox_area_gt']:,.0f}", f"{n['area_gap']:.3f}",
                         f"{n['vrel']:.3f}", f"{n['cost']:.3f}", f"{n['runtime_s']:.1f}"])
        table_page(pdf, "Anchor designs 0 / 49 / 98",
                   ["Design", "Blocks", "Run", "HPWL", "HPWL\n(GT)", "HPWL\ngap", "BBox\narea", "BBox\n(GT)",
                    "Area\ngap", "vrel", "Cost", "Runtime\n(s)"], rows, fontsize=10,
                   subtitle=f"Full pipeline (20 s PPO fine-tune + 8 s re-pack), {a.label}. These come from the "
                            "dedicated anchor run that produced the layout images on the next pages; the full-results "
                            "table is a separate run of all 100 designs, and single runs vary by about +-0.1 in cost.",
                   highlight=lambda r, c: r % 2 == 1 and c == 10)
        for t in ANCHORS:
            n = anchors[t]
            image_page(pdf, f"Design {t}: ground truth (left) vs ours (right)", ROOT / a.anchors / f"design{t}_layout.png",
                       f"Ours: HPWL {n['hpwl']:.1f} (GT {n['hpwl_gt']:.1f}), bbox {n['bbox_area']:,.0f} "
                       f"(GT {n['bbox_area_gt']:,.0f}), cost {n['cost']:.3f} (9/22: {old[t]['cost']:.3f}). "
                       "Red: pins and pin bounding box.")

        # 3. This week's work
        text_page(pdf, "What we did this week to improve QoR", WORK[:6])
        text_page(pdf, "What we did this week to improve QoR (cont.)", WORK[6:])

        # 4. Per-case chart
        fig, ax = plt.subplots(figsize=(W, H))
        fig.text(0.06, 0.94, "Cost per design: 9/22 vs now", fontsize=16, weight="bold")
        xs = range(len(ids))
        ax.bar([x - 0.2 for x in xs], [old[i]["cost"] for i in ids], width=0.4, label="9/22", color="#b0b8c4")
        ax.bar([x + 0.2 for x in xs], [new[i]["cost"] for i in ids], width=0.4, label=f"now ({a.label})",
               color="#2f6db5")
        ax.axhline(1.0, color="0.3", lw=0.8, ls="--")
        ax.text(len(ids) - 1, 1.02, "ground truth = 1.0", ha="right", fontsize=8, color="0.3")
        ax.set_xlabel("validation design id")
        ax.set_ylabel("cost (lower is better)")
        ax.set_xticks(list(xs)[::5])
        ax.set_xticklabels([ids[x] for x in list(xs)[::5]])
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(top=0.88)
        pdf.savefig(fig)
        plt.close(fig)

        # 5. Full table
        allrows = []
        for i in ids:
            o, n = old[i], new[i]
            blocks = int((ev.dataset[i]["input"][0] != -1).sum())
            allrows.append([i, blocks, f"{o['cost']:.3f}", f"{n['cost']:.3f}", f"{n['cost'] - o['cost']:+.3f}",
                            f"{o['hpwl_gap']:.3f}", f"{n['hpwl_gap']:.3f}", f"{o['area_gap']:.3f}",
                            f"{n['area_gap']:.3f}", f"{o['violations_relative']:.3f}", f"{n['vrel']:.3f}",
                            "Y" if n["feasible"] else "N", f"{n['secs']:.1f}"])
        header = ["Design", "Blocks", "Cost\n9/22", "Cost\nnow", "Change", "HPWL gap\n9/22", "HPWL gap\nnow",
                  "Area gap\n9/22", "Area gap\nnow", "vrel\n9/22", "vrel\nnow", "Feasible", "Runtime\nnow (s)"]
        for k in range(0, len(allrows), 34):
            chunk = allrows[k:k + 34]
            table_page(pdf, f"Full results, designs {chunk[0][0]}-{chunk[-1][0]}", header, chunk, fontsize=7.5,
                       highlight=lambda r, c, chunk=chunk: c == 4 and chunk[r][4].startswith("-"))

        # 6. White space and next steps
        text_page(pdf, "Why there is still so much white space, and what's next", [
            ("Diagnosis", "Our layouts fill the canvas they are given: on the median design the final bounding box is "
             "0.96x the canvas, and the canvas at utilization 0.70 is 1.39x the ground-truth box (which has only 3% "
             "white space). Edge-pinned blocks are placed early and flush with the canvas edges, fixing a canvas-sized "
             "frame; compaction and re-packing reclaim only ~4%. White space is set by canvas size, not packing quality."),
            ("Tried this week, none reduced white space", "Per-design canvas ratchet at solve time: tighter "
             "canvases almost never finish, area unchanged, cost 1.663 -> 1.687. Edge-pinned blocks placed last: the "
             "interior still spreads to fill the canvas (area gap 0.350 -> 0.343) and edge violations rise 3.0 -> 13.9 "
             "per design. 3x and 6x area weight in the training reward (2,000 iterations each vs a 1x control): area "
             "gap 0.350 / 0.350 / 0.355, cost 1.819 / 1.863 / 1.885."),
            ("What that tells us", "The spread comes from wirelength pulling blocks toward pins on the pin-box edges; "
             "only the canvas stops it. Reducing white space needs the canvas itself to shrink with the policy still "
             "able to finish, or an area term the policy sees during each placement, not only at the end."),
            ("Also", "Runtime counts in the official score (x max(0.7, (runtime/median)^0.3)); fine-tuning and polish "
             "both level off quickly, so shorter solves are worth testing."),
        ])
    print(f"wrote {a.out}: {len(ids)} designs, avg cost {avg(old, 'cost'):.3f} -> {avg(new, 'cost'):.3f}, "
          f"{wins} improved")


if __name__ == "__main__":
    main()

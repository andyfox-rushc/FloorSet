#!/usr/bin/env python3
"""Generates weekly_progress.pdf: the before/after summary table plus the
full 100-case my_optimizer_results.json breakdown, as a multi-page table
document (matplotlib PdfPages -- no extra dependency beyond what's already
installed)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent.parent))
from iccad2026_evaluate import (ContestEvaluator, calculate_hpwl_b2b,  # noqa: E402
                                 calculate_hpwl_p2b, calculate_bbox_area)

ROOT = Path(__file__).parent.parent
RESULTS_PATH = ROOT / "my_optimizer_results.json"
OUT_PATH = ROOT / "weekly_progress.pdf"

BEFORE = dict(avg_hpwl_gap=1.002, avg_area_gap=1.053, avg_vrel=0.125,
              avg_cost=2.62, feasible="100/100")

ANCHOR_CASES = [0, 49, 98]

TITLE = "RL Floorplanner progress 9/15-9/22"


def anchor_case_detail_page(pdf, tr_by_id):
    """First page: per-case HPWL / bounding-box-area breakdown (ours vs.
    ground truth) for the three tracked anchor cases (0, 49, 98)."""
    ev = ContestEvaluator("../", verbose=False)
    ev._load_dataset()

    rows = []
    for tid in ANCHOR_CASES:
        r = tr_by_id[tid]
        sample = ev.dataset[tid]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        block_count = int((area_target != -1).sum().item())
        _, target_pos = ev._extract_baseline(tid, labels, b2b_conn, p2b_conn, pins_pos, block_count)

        positions = [tuple(p) for p in r["positions"]]
        hpwl = calculate_hpwl_b2b(positions, b2b_conn) + calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        bbox_area = calculate_bbox_area(positions)

        gt_positions = [tuple(float(v) for v in target_pos[i]) for i in range(block_count)]
        gt_hpwl = calculate_hpwl_b2b(gt_positions, b2b_conn) + calculate_hpwl_p2b(gt_positions, p2b_conn, pins_pos)
        gt_area = calculate_bbox_area(gt_positions)

        rows.append([
            tid, f"{hpwl:.1f}", f"{gt_hpwl:.1f}", f"{r['hpwl_gap']:.3f}",
            f"{bbox_area:,.0f}", f"{gt_area:,.0f}", f"{r['area_gap']:.3f}",
            f"{r['violations_relative']:.3f}", f"{r['cost']:.3f}",
            "Yes" if r["is_feasible"] else "No", f"{r['runtime_seconds']:.1f}s",
        ])

    col_labels = ["Test ID", "HPWL\n(ours)", "HPWL\n(GT)", "HPWL\nGap",
                  "Bbox Area\n(ours)", "Bbox Area\n(GT)", "Area\nGap",
                  "Vrel", "Cost", "Feasible", "Runtime"]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.text(0.5, 0.95, TITLE, fontsize=20, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.85, "Tracked anchor cases: HPWL, bounding-box area, and cost detail",
              fontsize=11, ha="center", va="top")
    fig.text(0.5, 0.14, "GT = ground truth", fontsize=8, ha="center", va="top", style="italic")
    ax.set_position([0.04, 0.08, 0.92, 0.62])
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor("#404040")
        else:
            cell.set_facecolor("#f2f2f2" if row % 2 == 0 else "white")
    pdf.savefig(fig)
    plt.close(fig)


def summary_page(pdf, summary_rows):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    # Fixed-position text via fig.text (not suptitle/ax.title, which
    # overlapped in the rendered PDF despite looking fine in a standalone
    # PNG check -- explicit, well-separated y-coordinates are foolproof).
    fig.text(0.5, 0.95, TITLE, fontsize=20, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.85, "Full 100-case evaluation: before vs. after this week's fixes",
              fontsize=11, ha="center", va="top")
    ax.set_position([0.08, 0.08, 0.84, 0.62])
    ax.axis("off")

    col_labels = ["Metric", "Before", "After", "Change"]
    table = ax.table(cellText=summary_rows, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor("#404040")
        else:
            cell.set_facecolor("#f2f2f2" if row % 2 == 0 else "white")
    pdf.savefig(fig)
    plt.close(fig)


def results_table_pages(pdf, rows, rows_per_page=32):
    col_labels = ["Test ID", "Blocks", "HPWL Gap", "Area Gap", "Vrel", "Cost", "Feasible"]
    n_pages = (len(rows) + rows_per_page - 1) // rows_per_page
    for p in range(n_pages):
        chunk = rows[p * rows_per_page: (p + 1) * rows_per_page]
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title(f"Full 100-Case Results (page {p+1}/{n_pages})",
                      fontsize=13, fontweight="bold", pad=12)
        table = ax.table(cellText=chunk, colLabels=col_labels,
                          loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight="bold", color="white")
                cell.set_facecolor("#404040")
            else:
                cell.set_facecolor("#f2f2f2" if row % 2 == 0 else "white")
        pdf.savefig(fig)
        plt.close(fig)


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    summary = data["summary"]
    tr = sorted(data["test_results"], key=lambda r: r["test_id"])

    import statistics as st
    hpwl = [r["hpwl_gap"] for r in tr]
    area = [r["area_gap"] for r in tr]
    vrel = [r["violations_relative"] for r in tr]
    cost = [r["cost"] for r in tr]
    after = dict(avg_hpwl_gap=st.mean(hpwl), avg_area_gap=st.mean(area),
                 avg_vrel=st.mean(vrel), avg_cost=st.mean(cost),
                 feasible=f"{summary['num_feasible']}/{summary['num_tests']}")

    def pct(b, a):
        return f"{100*(a-b)/b:+.0f}%"

    summary_rows = [
        ["avg_hpwl_gap", f"{BEFORE['avg_hpwl_gap']:.3f}", f"{after['avg_hpwl_gap']:.3f}",
         pct(BEFORE['avg_hpwl_gap'], after['avg_hpwl_gap'])],
        ["avg_area_gap", f"{BEFORE['avg_area_gap']:.3f}", f"{after['avg_area_gap']:.3f}",
         pct(BEFORE['avg_area_gap'], after['avg_area_gap'])],
        ["avg_vrel", f"{BEFORE['avg_vrel']:.3f}", f"{after['avg_vrel']:.3f}",
         pct(BEFORE['avg_vrel'], after['avg_vrel'])],
        ["avg_cost", f"{BEFORE['avg_cost']:.3f}", f"{after['avg_cost']:.3f}",
         pct(BEFORE['avg_cost'], after['avg_cost'])],
        ["feasible", BEFORE["feasible"], after["feasible"], "—"],
    ]

    results_rows = [
        [r["test_id"], r["block_count"], f"{r['hpwl_gap']:.3f}", f"{r['area_gap']:.3f}",
         f"{r['violations_relative']:.3f}", f"{r['cost']:.3f}",
         "Yes" if r["is_feasible"] else "No"]
        for r in tr
    ]

    tr_by_id = {r["test_id"]: r for r in tr}

    with PdfPages(OUT_PATH) as pdf:
        anchor_case_detail_page(pdf, tr_by_id)
        summary_page(pdf, summary_rows)
        results_table_pages(pdf, results_rows)

    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()

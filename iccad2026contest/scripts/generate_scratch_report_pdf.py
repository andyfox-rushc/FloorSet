#!/usr/bin/env python3
"""From-scratch retraining report, in weekly_progress.pdf's style (see
scripts/generate_weekly_progress_pdf.py): anchor table, anchor layouts,
100-case before (installed policy) vs after (new checkpoint), training
progress, full 100-case table. matplotlib PdfPages only.

  python scripts/generate_scratch_report_pdf.py --after scratch_final_eval.json \
      --anchors results_scratch/final --label "iteration 20000" --out scratch_report_1003.pdf
"""
import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
ANCHORS = [0, 49, 98]
HEAD, ALT = "#404040", "#f2f2f2"


class Pages(PdfPages):
    """PdfPages that can also write each page as page<N>.png (for review)."""
    png_dir = None

    def savefig(self, fig, **kw):
        super().savefig(fig, **kw)
        if self.png_dir:
            n = self.get_pagecount()
            fig.savefig(Path(self.png_dir) / f"page{n:02d}.png", dpi=70)


def style(table):
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor(HEAD)
        else:
            cell.set_facecolor(ALT if row % 2 == 0 else "white")


def header(fig, title, subtitle):
    fig.text(0.5, 0.95, title, fontsize=20, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.87, subtitle, fontsize=11, ha="center", va="top")


def load_eval(path):
    d = json.load(open(ROOT / path))
    d = d[next(iter(d))]
    return {int(k): v for k, v in d["per_case"].items()}


def anchor_rows(anchor_dir):
    """Median-cost run per anchor design from <dir>/run*/results.json (or
    <dir>/results.json for a single run)."""
    d = ROOT / anchor_dir
    runs = []
    for f in sorted(d.glob("run*/results.json")) or [d / "results.json"]:
        runs.extend(json.load(open(f)))
    out = {}
    for tid in ANCHORS:
        xs = sorted((x for x in runs if x["design"] == tid), key=lambda x: x["cost"])
        if xs:
            out[tid] = xs[len(xs) // 2]
    return out


def anchor_table_page(pdf, title, anchors, label):
    rows = [[tid, f"{r['hpwl']:.1f}", f"{r['hpwl_gt']:.1f}", f"{r['hpwl_gap']:.3f}",
             f"{r['bbox_area']:,.0f}", f"{r['bbox_area_gt']:,.0f}", f"{r['area_gap']:.3f}",
             f"{r['vrel']:.3f}", f"{r['cost']:.3f}", "Yes" if r["feasible"] else "No",
             f"{r['runtime_s']:.1f}s"] for tid, r in anchors.items()]
    cols = ["Test ID", "HPWL\n(ours)", "HPWL\n(GT)", "HPWL\nGap", "Bbox Area\n(ours)",
            "Bbox Area\n(GT)", "Area\nGap", "Vrel", "Cost", "Feasible", "Runtime"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    header(fig, title, f"Tracked anchor cases ({label}): HPWL, bounding-box area, and cost detail")
    fig.text(0.5, 0.14, "GT = ground truth.  Median of 3 runs per design (2 s fine-tune + 1 s polish).",
             fontsize=8, ha="center", va="top", style="italic")
    ax.set_position([0.04, 0.08, 0.92, 0.62])
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 2.4)
    style(t)
    pdf.savefig(fig)
    plt.close(fig)


def anchor_image_pages(pdf, title, anchors, label):
    for tid, r in anchors.items():
        img = ROOT / r["image"]
        if not img.exists():
            continue
        fig = plt.figure(figsize=(11, 6.2))
        header(fig, title, f"Design {tid} layout ({label}): ground truth (left) vs ours (right)")
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.80])
        ax.imshow(mpimg.imread(img))
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def summary_page(pdf, title, before, after, before_label, after_label):
    def avg(d, k):
        return st.mean(v[k] for v in d.values())

    def pct(b, a):
        return f"{100 * (a - b) / b:+.0f}%" if b else "—"

    rows = []
    for name, k in (("avg_hpwl_gap", "hpwl_gap"), ("avg_area_gap", "area_gap"),
                    ("avg_vrel", "vrel"), ("avg_cost", "cost")):
        b, a = avg(before, k), avg(after, k)
        rows.append([name, f"{b:.3f}", f"{a:.3f}", pct(b, a)])
    feas = lambda d: f"{sum(bool(v['feasible']) for v in d.values())}/{len(d)}"
    rows.append(["feasible", feas(before), feas(after), "—"])
    better = sum(after[i]["cost"] < before[i]["cost"] for i in after if i in before)
    rows.append(["designs improved", "—", f"{better}/{len(after)}", "—"])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    header(fig, title, "Full 100-case evaluation: installed policy vs. from-scratch retrain")
    fig.text(0.5, 0.80, f"Before = {before_label}.  After = {after_label}.",
             fontsize=8.5, ha="center", va="top", style="italic")
    ax.set_position([0.08, 0.06, 0.84, 0.64])
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["Metric", "Before", "After", "Change"],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 2.0)
    style(t)
    pdf.savefig(fig)
    plt.close(fig)


def progress_pages(pdf, title, pipe_rows, base_cost, images):
    if pipe_rows:
        cols = ["Iteration", "Cost", "HPWL Gap", "Area Gap", "Vrel", "vs installed"]
        rows = [[it, f"{r['cost']:.3f}", f"{r['hpwl_gap']:.3f}", f"{r['area_gap']:.3f}",
                 f"{r['vrel']:.3f}", f"{r['cost'] - base_cost:+.3f}"] for it, r in pipe_rows]
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        header(fig, title, "Training progress: full pipeline on 100 designs, every 2000 iterations")
        fig.text(0.5, 0.80, f"Installed policy (iter 12000) cost {base_cost:.3f}. "
                 "Evals before the final one ran alongside training.",
                 fontsize=8.5, ha="center", va="top", style="italic")
        ax.set_position([0.08, 0.06, 0.84, 0.66])
        ax.axis("off")
        t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        t.scale(1, 1.7)
        style(t)
        pdf.savefig(fig)
        plt.close(fig)
    for img, sub in images:
        p = ROOT / img
        if not p.exists():
            continue
        fig = plt.figure(figsize=(11, 4.8))
        header(fig, title, sub)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.78])
        ax.imshow(mpimg.imread(p))
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def results_table_pages(pdf, after, blocks, rows_per_page=32):
    rows = [[i, blocks.get(i, "—"), f"{r['hpwl_gap']:.3f}", f"{r['area_gap']:.3f}",
             f"{r['vrel']:.3f}", f"{r['cost']:.3f}", "Yes" if r["feasible"] else "No"]
            for i, r in sorted(after.items())]
    cols = ["Test ID", "Blocks", "HPWL Gap", "Area Gap", "Vrel", "Cost", "Feasible"]
    n = (len(rows) + rows_per_page - 1) // rows_per_page
    for p in range(n):
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title(f"Full 100-Case Results (page {p + 1}/{n})", fontsize=13, fontweight="bold", pad=12)
        t = ax.table(cellText=rows[p * rows_per_page:(p + 1) * rows_per_page], colLabels=cols,
                     loc="upper center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.5)
        style(t)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="exp_hw_ref_base12000.json")
    ap.add_argument("--before-label", default="installed iter 12000")
    ap.add_argument("--after", required=True)
    ap.add_argument("--anchors", required=True, help="dir with run*/results.json or results.json")
    ap.add_argument("--label", required=True, help='e.g. "iteration 20000"')
    ap.add_argument("--pipe-glob", default="scratch_pipe_iter*.json")
    ap.add_argument("--title", default="RL Floorplanner progress: from-scratch retrain")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png-dir", default=None, help="also save each page as a PNG here")
    a = ap.parse_args()

    from iccad2026_evaluate import ContestEvaluator
    ev = ContestEvaluator("../", verbose=False)
    ev._load_dataset()
    blocks = {i: int((ev.dataset[i]["input"][0] != -1).sum().item()) for i in range(len(ev.dataset))}

    before, after = load_eval(a.before), load_eval(a.after)
    pipe_rows = []
    for f in sorted(ROOT.glob(a.pipe_glob)):
        d = json.load(open(f))
        d = d[next(iter(d))]
        pipe_rows.append((int(f.stem.split("iter")[-1]), d))
    base_cost = st.mean(v["cost"] for v in before.values())
    anchors = anchor_rows(a.anchors)
    Pages.png_dir = a.png_dir
    if a.png_dir:
        Path(a.png_dir).mkdir(parents=True, exist_ok=True)
    with Pages(ROOT / a.out) as pdf:
        anchor_table_page(pdf, a.title, anchors, a.label)
        anchor_image_pages(pdf, a.title, anchors, a.label)
        summary_page(pdf, a.title, before, after, a.before_label, f"from-scratch retrain, {a.label}")
        progress_pages(pdf, a.title, pipe_rows, base_cost, [
            ("results_scratch/greedy_progress.png",
             "Training progress: 100 designs, greedy rollout (no fine-tune), every 1000 iterations"),
            ("results_scratch/progress.png",
             "Training progress: anchor designs, full pipeline (median of 3 runs), every 500 iterations")])
        results_table_pages(pdf, after, blocks)
    print(f"saved {ROOT / a.out}")


if __name__ == "__main__":
    main()

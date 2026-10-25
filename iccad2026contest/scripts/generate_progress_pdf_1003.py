#!/usr/bin/env python3
"""Progress report PDF 9/28 - 10/3, same format as progress_report_0928.pdf
(scripts/generate_progress_pdf.py): summary, anchor table, anchor layouts,
this week's QoR work, per-case cost chart, full 100-case table, training
progress, white space / next steps. "Last week" = the 9/28 report's numbers
(full_eval_iter012000.json, results_20260928).

  python scripts/generate_progress_pdf_1003.py --new <eval.json> --new-key 2:slice+repack:1 \
      --anchors results_scratch/final --label "..." --out progress_report_1003.pdf
"""
import argparse
import json
import math
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
sys.path.insert(0, str(ROOT / "scripts"))
from generate_progress_pdf import W, H, text_page, table_page, image_page  # noqa: E402

ANCHORS = [0, 49, 98]
WORK_FILE = ROOT / "scripts" / "progress_1003_text.json"


def load_eval(path, key=None):
    d = json.load(open(ROOT / path))
    d = d[key] if key else d[next(iter(d))]
    return {int(k): v for k, v in d["per_case"].items()}


def load_anchors(d):
    """Median-cost run per design from <d>/run*/results.json, else <d>/results.json."""
    d = ROOT / d
    runs = []
    for f in sorted(d.glob("run*/results.json")) or [d / "results.json"]:
        runs.extend(json.load(open(f)))
    out = {}
    for t in ANCHORS:
        xs = sorted((x for x in runs if x["design"] == t), key=lambda x: x["cost"])
        if xs:
            out[t] = xs[len(xs) // 2]
    return out


def _flag_china(ax):
    """PRC flag (3:2): red field, one large and four small yellow five-pointed
    stars, each small star with one point aimed at the large star's centre."""
    import matplotlib.patches as mpatches

    def star(cx, cy, r, angle):
        pts = []
        for k in range(10):     # alternate outer / inner vertices
            rr = r if k % 2 == 0 else r * 0.382
            a = angle + k * math.pi / 5
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        ax.add_patch(mpatches.Polygon(pts, closed=True, color="#ffde00", lw=0))

    ax.add_patch(mpatches.Rectangle((0, 0), 30, 20, color="#de2910"))
    star(5, 15, 3, math.pi / 2)
    for cx, cy in ((10, 18), (12, 16), (12, 13), (10, 11)):
        star(cx, cy, 1, math.atan2(15 - cy, 5 - cx))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 20)


def _flag_uk(ax):
    """Union Jack (2:1, height 30): blue field; St Andrew's white saltire (6 wide);
    St Patrick's red saltire (2 wide), offset to one side of the white in each
    arm (pinwheel / counterchanged); St George's red cross (6) with white border (10)."""
    import matplotlib.patches as mpatches
    import numpy as np
    W, H = 60.0, 30.0
    ax.add_patch(mpatches.Rectangle((0, 0), W, H, color="#012169"))

    def band(p0, p1, n, off, width, color):
        p0, p1, n = np.array(p0, float), np.array(p1, float), np.array(n, float)
        a, b = n * (off - width / 2), n * (off + width / 2)
        ax.add_patch(mpatches.Polygon([p0 + a, p1 + a, p1 + b, p0 + b], closed=True, color=color, lw=0))

    c = np.array([W / 2, H / 2])
    for corner in ((0, 0), (W, H), (0, H), (W, 0)):
        u = np.array(corner) - c
        u = u / np.linalg.norm(u)
        far = c + u * 40                         # past the corner; clipped by the axes
        n_ccw = np.array([-u[1], u[0]])
        band(c, far, n_ccw, 0.0, 6.0, "white")   # St Andrew
        band(c, far, n_ccw, 1.0, 2.0, "#c8102e")  # St Patrick, offset (pinwheel)
    ax.add_patch(mpatches.Rectangle((W / 2 - 5, 0), 10, H, color="white"))
    ax.add_patch(mpatches.Rectangle((0, H / 2 - 5), W, 10, color="white"))
    ax.add_patch(mpatches.Rectangle((W / 2 - 3, 0), 6, H, color="#c8102e"))
    ax.add_patch(mpatches.Rectangle((0, H / 2 - 3), W, 6, color="#c8102e"))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)


def _flags_in_cell(fig, cell):
    """Small PRC and UK flags at the right end of a table cell."""
    fig.canvas.draw()
    bb = cell.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    h = bb.height * 0.62
    figw, figh = fig.get_size_inches()
    wcn = h * 1.5 * figh / figw               # 3:2 in physical units
    wuk = h * 2.0 * figh / figw               # 2:1
    y = bb.y0 + (bb.height - h) / 2
    x_uk = bb.x1 - wuk - bb.width * 0.03
    x_cn = x_uk - wcn - bb.width * 0.02
    for x, w, draw in ((x_cn, wcn, _flag_china), (x_uk, wuk, _flag_uk)):
        ax = fig.add_axes([x, y, w, h])
        draw(ax)
        ax.set_axis_off()


def leaderboard_page(pdf, path, new, ids, wts, text, label="now"):
    """Leaderboard sorted by raw score (weighted cost before the runtime
    factor), our 100-design result placed among the entries."""
    import csv
    lb = list(csv.DictReader(open(ROOT / path)))
    tw = sum(wts.values())
    raw = sum(new[i]["cost"] * wts[i] for i in ids) / tw
    ent = [(float(r["raw_score"]), f"Leaderboard #{r['rank']}", f"{float(r['n_feasible']):.0f}/100") for r in lb]
    ent.append((raw, f"Team ICB ({label})", "100/100"))
    ent.sort(key=lambda e: e[0])
    best = ent[0][0]
    rows = [[str(k + 1), name, f"{c:.3f}", "" if k == 0 else f"+{c - best:.3f}", feas]
            for k, (c, name, feas) in enumerate(ent)]
    us = next(k for k, e in enumerate(ent) if e[1].startswith("Team ICB"))
    table_page(pdf, text.get("leaderboard_title", "ICCAD 2026 Problem C leaderboard, sorted by cost"),
               ["Place by cost", "Entry", "Cost (raw score)", "Behind best", "Feasible"], rows,
               col_widths=[0.15, 0.33, 0.18, 0.18, 0.16], fontsize=10,
               highlight=lambda r, c: r == us,
               subtitle=text.get("leaderboard_note", ""),
               decorate=lambda fig, tbl: _flags_in_cell(fig, tbl[us + 1, 1]))

class Pages(PdfPages):
    png_dir = None

    def savefig(self, fig, **kw):
        super().savefig(fig, **kw)
        if self.png_dir:
            fig.savefig(Path(self.png_dir) / f"page{self.get_pagecount():02d}.png", dpi=70)


def continuation_page(pdf, d, text, logs=("logs/train_ab_A.log", "logs/train_ab_A_full.log"), installed=4000):
    """Training reward, 100-design greedy cost and anchor average vs iteration for one run."""
    import csv
    import re
    rew = {}
    for log in logs:
        for line in open(ROOT / log):
            m = re.match(r"iter (\d+)/\d+ .*avg_reward=(\S+)", line)
            if m:
                rew.setdefault(int(m[1]) // 1000 * 1000 + 500, []).append(float(m[2]))
    rew = sorted((k, st.mean(v)) for k, v in rew.items() if len(v) >= 200)
    greedy = [(int(r["iter"]), float(r["cost"])) for r in csv.DictReader(open(d / "greedy.csv"))]
    anch = {}
    for r in csv.DictReader(open(d / "summary.csv")):
        anch.setdefault(int(r["iter"]), []).append(float(r["cost"]))
    anch = sorted((k, st.mean(v)) for k, v in anch.items() if len(v) == len(ANCHORS))

    fig = plt.figure(figsize=(W, H))
    fig.text(0.06, 0.94, text.get("continuation_title", "Does more training help?"), fontsize=16, weight="bold")
    fig.text(0.06, 0.905, textwrap.fill(text["continuation_note"], 140), fontsize=9.5, color="0.35", va="top")
    panels = [(rew, "training reward (avg per 1,000 iters)", "higher is better"),
              (greedy, "greedy cost, 100 designs", "lower is better"),
              (anch, "avg cost, designs 0/49/98 (no best-of)", "lower is better")]
    for k, (pts, title, note) in enumerate(panels):
        ax = fig.add_axes([0.07 + k * 0.32, 0.12, 0.26, 0.56])
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="#2f6db5", marker="o", ms=3.5, lw=1.4)
        ax.axvline(installed, color="#b05a2f", ls="--", lw=1.1)
        ax.text(installed, 0.02, " installed (4,000)", transform=ax.get_xaxis_transform(), fontsize=8,
                color="#b05a2f", va="bottom")
        ax.set_title(f"{title}\n({note})", fontsize=10)
        ax.set_xlabel("training iteration")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="0.9")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="full_eval_iter012000.json")
    ap.add_argument("--old-anchors", default="results_20260928")
    ap.add_argument("--old-label", default="9/28 report")
    ap.add_argument("--new", required=True)
    ap.add_argument("--new-key", default=None)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--label", required=True, help="what 'now' is, e.g. 'installed iter 12000'")
    ap.add_argument("--pipeline", default="2 s PPO fine-tune + 1 s slicing/re-pack polish")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="label=eval.json pairs shown as extra summary columns")
    ap.add_argument("--progress-glob", default="scratch_pipe_iter*.json")
    ap.add_argument("--title", default="RL Floorplanner progress 9/28 - 10/3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png-dir", default=None)
    ap.add_argument("--text", default=str(WORK_FILE), help="report text JSON")
    ap.add_argument("--leaderboard", default=None, help="contest leaderboard CSV (rank,total_score,raw_score,...)")
    ap.add_argument("--alt-image", nargs="*", default=[],
                    help='design=results_dir=label: show that run\'s layout on the design page instead, '
                         'captioned with its own source and numbers')
    a = ap.parse_args()

    from iccad2026_evaluate import ContestEvaluator
    ev = ContestEvaluator(str(ROOT.parent) + "/", verbose=False)
    ev._load_dataset()
    blocks = lambda i: int((ev.dataset[i]["input"][0] != -1).sum())

    old, new = load_eval(a.old), load_eval(a.new, a.new_key)
    old_a, new_a = load_anchors(a.old_anchors), load_anchors(a.anchors)
    extras = [(lab, load_eval(p)) for lab, p in (e.split("=", 1) for e in a.extra)]
    text = json.load(open(a.text))
    ids = sorted(set(old) & set(new))
    avg = lambda d, k: st.mean(d[i][k] for i in ids)
    med = lambda d, k: st.median(d[i][k] for i in ids)
    feas = lambda d: f"{sum(bool(d[i]['feasible']) for i in ids)}/{len(ids)}"
    wins = sum(new[i]["cost"] < old[i]["cost"] for i in ids)

    Pages.png_dir = a.png_dir
    if a.png_dir:
        Path(a.png_dir).mkdir(parents=True, exist_ok=True)
    with Pages(ROOT / a.out) as pdf:
        # 1. Summary
        cols = [a.old_label, f"Now ({a.label})"] + [lab for lab, _ in extras]
        data = [old, new] + [d for _, d in extras]
        pct = lambda k, f: f"{100 * (f(new, k) / f(old, k) - 1):+.1f}%"
        # official total: designs weighted by exp(blocks / 12) (README, Scoring)
        wts = {i: math.exp(blocks(i) / 12) for i in ids}
        wavg = lambda d, k: sum(d[i][k] * wts[i] for i in ids) / sum(wts.values())
        rows = [["Weighted score (official)"] + [f"{wavg(d, 'cost'):.3f}" for d in data] + [pct("cost", wavg)],
                ["Average cost"] + [f"{avg(d, 'cost'):.3f}" for d in data] + [pct("cost", avg)],
                ["Median cost"] + [f"{med(d, 'cost'):.3f}" for d in data] + [pct("cost", med)],
                ["Avg HPWL gap"] + [f"{avg(d, 'hpwl_gap'):.3f}" for d in data] + [pct("hpwl_gap", avg)],
                ["Avg area gap"] + [f"{avg(d, 'area_gap'):.3f}" for d in data] + [pct("area_gap", avg)],
                ["Avg soft violations (vrel)"] + [f"{avg(d, 'vrel'):.3f}" for d in data] + [""],
                ["Feasible"] + [feas(d) for d in data] + [""],
                ["Designs improved vs 9/28", ""] + [f"{sum(d[i]['cost'] < old[i]['cost'] for i in ids)}/{len(ids)}"
                                                    for d in data[1:]] + [""],
                ["Avg runtime per design (s)"] + [f"{avg(d, 'secs'):.1f}" for d in data] + [pct("secs", avg)]]
        n = len(cols)
        table_page(pdf, a.title, ["Metric (100 validation designs)"] + cols + ["Change vs 9/28"], rows,
                   col_widths=[0.3] + [0.55 / n] * n + [0.15], fontsize=10,
                   subtitle="Full contest pipeline on all 100 validation designs. Cost = (1 + 0.5 (HPWL gap + area gap)) "
                            "x exp(2 vrel); 1.0 = matches ground truth. Local runtime factor = 1.0 (neutral); the "
                            "official score also multiplies by max(0.7, (runtime/median)^0.3) per design (see the "
                            "leaderboard page). " + text.get("summary_note", ""))

        # 1b. Contest leaderboard (10-06): raw quality vs the runtime factor
        if a.leaderboard:
            leaderboard_page(pdf, a.leaderboard, new, ids, wts, text, a.label)

        # 2. Anchor table
        rows = []
        for t in ANCHORS:
            o, m = old_a[t], new_a[t]
            for tag, r in (("9/28", o), ("now", m)):
                rows.append([f"{t}" if tag == "9/28" else "", f"{r['blocks']}" if tag == "9/28" else "", tag,
                             f"{r['hpwl']:.1f}", f"{r['hpwl_gt']:.1f}", f"{r['hpwl_gap']:.3f}",
                             f"{r['bbox_area']:,.0f}", f"{r['bbox_area_gt']:,.0f}", f"{r['area_gap']:.3f}",
                             f"{r['vrel']:.3f}", f"{r['cost']:.3f}", f"{r['runtime_s']:.1f}"])
        table_page(pdf, "Anchor designs 0 / 49 / 98",
                   ["Design", "Blocks", "Run", "HPWL", "HPWL\n(GT)", "HPWL\ngap", "BBox\narea", "BBox\n(GT)",
                    "Area\ngap", "vrel", "Cost", "Runtime\n(s)"], rows, fontsize=10,
                   subtitle=f"Now: {a.label}, {a.pipeline}, median of 3 runs per design (single runs vary by up to "
                            "~0.4 in cost at this budget). 9/28: iteration 12000, 20 s fine-tune + 8 s re-pack, single runs. "
                            "Three designs are a noisy sample; the 100-design average is the reliable comparison.",
                   highlight=lambda r, c: r % 2 == 1 and c == 10 and float(rows[r][10]) < float(rows[r - 1][10]))
        alt = {int(t): (d, lab) for t, d, lab in (e.split("=", 2) for e in a.alt_image)}
        for t in ANCHORS:
            m = new_a[t]
            if t in alt:
                r = load_anchors(alt[t][0])[t]
                image_page(pdf, f"Design {t}: ground truth (left) vs ours (right), {alt[t][1]}", ROOT / r["image"],
                           textwrap.fill(f"Shown: {alt[t][1]} (not the installed policy): HPWL {r['hpwl']:.1f} (GT {r['hpwl_gt']:.1f}), "
                           f"bbox {r['bbox_area']:,.0f} (GT {r['bbox_area_gt']:,.0f}), cost {r['cost']:.3f}. The installed "
                           f"policy's run in the table (cost {m['cost']:.3f}) pushes right-pinned blocks past the pin box. "
                           "Red: pins and pin bounding box.", 150))
                continue
            image_page(pdf, f"Design {t}: ground truth (left) vs ours (right)", ROOT / m["image"],
                       f"Ours: HPWL {m['hpwl']:.1f} (GT {m['hpwl_gt']:.1f}), bbox {m['bbox_area']:,.0f} "
                       f"(GT {m['bbox_area_gt']:,.0f}), cost {m['cost']:.3f} (9/28: {old_a[t]['cost']:.3f}). "
                       "Red: pins and pin bounding box.")

        # 3. This week's work
        work = [tuple(p) for p in text["work"]]
        for k in range(0, len(work), 3):     # 10-18: 6 (then 4) per page overflowed with the longer entries
            text_page(pdf, "What we did this week to improve QoR" + (" (cont.)" if k else ""), work[k:k + 3])

        # 4. Per-case chart
        fig, ax = plt.subplots(figsize=(W, H))
        fig.text(0.06, 0.94, "Cost per design: 9/28 vs now", fontsize=16, weight="bold")
        xs = range(len(ids))
        ax.bar([x - 0.2 for x in xs], [old[i]["cost"] for i in ids], width=0.4, label=a.old_label, color="#b0b8c4")
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
            o, m = old[i], new[i]
            allrows.append([i, blocks(i), f"{o['cost']:.3f}", f"{m['cost']:.3f}", f"{m['cost'] - o['cost']:+.3f}",
                            f"{o['hpwl_gap']:.3f}", f"{m['hpwl_gap']:.3f}", f"{o['area_gap']:.3f}",
                            f"{m['area_gap']:.3f}", f"{o['vrel']:.3f}", f"{m['vrel']:.3f}",
                            "Y" if m["feasible"] else "N", f"{m['secs']:.1f}"])
        header = ["Design", "Blocks", "Cost\n9/28", "Cost\nnow", "Change", "HPWL gap\n9/28", "HPWL gap\nnow",
                  "Area gap\n9/28", "Area gap\nnow", "vrel\n9/28", "vrel\nnow", "Feasible", "Runtime\nnow (s)"]
        for k in range(0, len(allrows), 34):
            chunk = allrows[k:k + 34]
            table_page(pdf, f"Full results, designs {chunk[0][0]}-{chunk[-1][0]}", header, chunk, fontsize=7.5,
                       highlight=lambda r, c, chunk=chunk: c == 4 and chunk[r][4].startswith("-"))

        # 6. Training progress (from-scratch run)
        prog = []
        for f in sorted(ROOT.glob(a.progress_glob)):
            d = json.load(open(f))
            d = d[next(iter(d))]
            prog.append([int(f.stem.split("iter")[-1]), f"{d['cost']:.3f}", f"{d['hpwl_gap']:.3f}",
                         f"{d['area_gap']:.3f}", f"{d['vrel']:.3f}"])
        if prog:
            table_page(pdf, "From-scratch retrain: full pipeline every 2,000 iterations",
                       ["Iteration", "Cost", "HPWL gap", "Area gap", "vrel"], prog, fontsize=11,
                       subtitle=text.get("progress_note", ""))
        greedy = ROOT / "results_scratch" / "greedy_progress.png"
        if greedy.exists():
            image_page(pdf, "From-scratch retrain: 100 designs, greedy rollout, every 1,000 iterations", greedy,
                       "Greedy rollout without fine-tune or polish: depends only on the checkpoint (no run-to-run "
                       "noise). Dotted: installed iter 12000 (no frame).")

        # 6b. Policy pick: every from-scratch checkpoint, repeats of the top ones
        scan = []
        for f in sorted((ROOT / "allck").glob("scratch_iter*.json")):
            d = json.load(open(f))
            scan.append((int(f.stem.split("iter")[-1]), d[next(iter(d))]["cost"]))
        if scan:
            reps = {}
            for f in sorted((ROOT / "allck").glob("rep*_*.json")):
                name = f.stem.split("_", 1)[1]
                d = json.load(open(f))
                reps.setdefault(name, []).append(d[next(iter(d))]["cost"])
            base = st.mean(v["cost"] for v in new.values())
            fig = plt.figure(figsize=(W, H))
            fig.text(0.06, 0.94, "Picking the policy: every from-scratch checkpoint on 100 designs", fontsize=16,
                     weight="bold")
            fig.text(0.06, 0.905, textwrap.fill(text.get("pick_note", ""), 150), fontsize=9.5, color="0.35", va="top")
            ax = fig.add_axes([0.08, 0.42, 0.86, 0.38])
            xs, ys = zip(*scan)
            ax.plot(xs, ys, color="#2f6db5", marker="o", ms=4, lw=1.5, label="from-scratch checkpoint")
            ax.axhline(base, color="#b05a2f", ls="--", lw=1.2, label=f"installed iter 12000 ({base:.3f})")
            ax.set_xlabel("training iteration")
            ax.set_ylabel("average cost, 100 designs")
            ax.legend(frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, color="0.9")
            rows = []
            for name, cs in sorted(reps.items(), key=lambda kv: st.mean(kv[1])):
                first = None
                if name.startswith("scratch_iter"):
                    it = int(name.split("iter")[-1])
                    first = dict(scan).get(it)
                elif name.startswith("installed"):
                    first = base
                allc = ([first] if first is not None else []) + cs
                label = (f"from-scratch iter {int(name.split('iter')[-1])}" if name.startswith("scratch")
                         else "installed iter 12000")
                rows.append([label] + [f"{c:.3f}" for c in allc] + [f"{st.mean(allc):.3f}"])
            if rows:
                n = max(len(r) for r in rows) - 2
                hdr = ["Policy"] + [f"Run {k + 1}" for k in range(n)] + ["Average"]
                rows = [r[:-1] + [""] * (n + 2 - len(r)) + [r[-1]] for r in rows]   # pad short rows
                tax = fig.add_axes([0.08, 0.06, 0.86, 0.06 * (len(rows) + 1)])
                tax.axis("off")
                tbl = tax.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="center",
                                colWidths=[0.3] + [0.7 / (n + 1)] * (n + 1))
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9.5)
                for (r, c), cell in tbl.get_celld().items():
                    if r == 0:
                        cell.set_facecolor("#dfe6ee")
                        cell.set_text_props(weight="bold")
            pdf.savefig(fig)
            plt.close(fig)

        # 6b'. Arm A continued past 4000 (10-05): does more training help?
        cont = ROOT / "results_ab_A"
        if text.get("continuation_note") and (cont / "greedy.csv").exists():
            continuation_page(pdf, cont, text)

        # 6c. Policy review and proposed improvements (10-04)
        if text.get("review"):
            text_page(pdf, text["review_title"], [tuple(x) for x in text["review"]])
        if text.get("proposals"):
            text_page(pdf, text["proposals_title"], [tuple(x) for x in text["proposals"]])

        # 7. White space and next steps
        text_page(pdf, text.get("next_title", "White space, and what's next"), [tuple(p) for p in text["next"]])
    print(f"wrote {a.out}: {len(ids)} designs, avg cost {avg(old, 'cost'):.3f} -> {avg(new, 'cost'):.3f}, "
          f"{wins} improved")


if __name__ == "__main__":
    main()

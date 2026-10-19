"""Ground truth vs our layout for the largest validation designs (10-17).

One summary page (costs, gaps, violations, who wins) and one page per design
with the ground truth and our official layout side by side on the same axes.
Preplaced blocks are hatched; blocks that miss their required edge (boundary
violation) are outlined in red; pins and the pin box are drawn as in the
anchor images (scripts/anchor_results.py).

  python scripts/big_designs_pdf.py --results official_eval_1017_repack.json --top 10 \
      --out big_designs_1017.pdf
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution


def boundary_misses(pos, cons):
    """Indices of blocks with a boundary code whose required edge is off the bounding box."""
    x0 = min(p[0] for p in pos)
    y0 = min(p[1] for p in pos)
    x1 = max(p[0] + p[2] for p in pos)
    y1 = max(p[1] + p[3] for p in pos)
    out = set()
    for i, (x, y, w, h) in enumerate(pos):
        code = int(cons[i, 4]) if cons.shape[1] > 4 else 0
        if not code:
            continue
        ok = {1: abs(x - x0) < 1e-6, 2: abs(x + w - x1) < 1e-6, 4: abs(y + h - y1) < 1e-6, 8: abs(y - y0) < 1e-6}
        if not all(ok[b] for b in (1, 2, 4, 8) if code & b):
            out.add(i)
    return out


def panel(ax, pos, cons, pins, title, colors):
    miss = boundary_misses(pos, cons)
    pre = cons[:, 1] != 0
    for i, (x, y, w, h) in enumerate(pos):
        ax.add_patch(mpatches.Rectangle((x, y), w, h, facecolor=colors[i], alpha=0.7,
                                        edgecolor="red" if i in miss else "black",
                                        linewidth=2.2 if i in miss else 0.6,
                                        hatch="///" if pre[i] else None))
        ax.text(x + w / 2, y + h / 2, str(i), ha="center", va="center", fontsize=5)
    valid = pins[(pins[:, 0] != -1) | (pins[:, 1] != -1)]
    if len(valid):
        ax.scatter(valid[:, 0], valid[:, 1], s=5, c="red", marker="x")
        ax.add_patch(mpatches.Rectangle((0, 0), valid[:, 0].max(), valid[:, 1].max(),
                                        fill=False, edgecolor="red", linestyle="--"))
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.autoscale()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="official_eval_1017_repack.json")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--label", default="10/17 official run")
    ap.add_argument("--out", default="big_designs_1017.pdf")
    ap.add_argument("--png-dir", default=None, help="also write each page as a PNG (for checking)")
    a = ap.parse_args()
    ev = ContestEvaluator("../", verbose=False)
    ev._load_dataset()
    res = {x["test_id"]: x for x in json.load(open(a.results))["test_results"]}
    sizes = {}
    for t in res:
        area = ev.dataset[t]["input"][0]
        sizes[t] = int((area != -1).sum())
    ids = sorted(res, key=lambda t: (-sizes[t], t))[:a.top]
    ids.sort(key=lambda t: sizes[t])
    rows = []
    for t in ids:
        area, b2b, p2b, pins, cons = ev.dataset[t]["input"]
        n = sizes[t]
        base, gt = ev._extract_baseline(t, ev.dataset[t]["label"], b2b, p2b, pins, n)
        gt = [tuple(map(float, q)) for q in gt[:n]]
        ours = [tuple(map(float, q)) for q in res[t]["positions"]]
        score = lambda pos: evaluate_solution({"positions": pos, "runtime": 1.0}, base, cons, b2b, p2b, pins,
                                              area, gt, median_runtime=1.0)
        rows.append(dict(t=t, n=n, gt=gt, ours=ours, mg=score(gt), mo=score(ours), cons=cons[:n].numpy(),
                         pins=pins.numpy(), secs=res[t]["runtime_seconds"]))

    def save(pdf, fig):
        pdf.savefig(fig)
        if a.png_dir:
            Path(a.png_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(a.png_dir) / f"page{pdf.get_pagecount():02d}.png", dpi=70)
        plt.close(fig)

    with PdfPages(a.out) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.94, f"The {len(rows)} largest designs: ground truth vs ours", ha="center", fontsize=18)
        fig.text(0.5, 0.905, f"Our layouts from {a.results} ({a.label}). Contest cost, lower is better; "
                 "the gaps are relative to the ground truth.", ha="center", fontsize=9.5)
        hdr = ["Design", "Blocks", "GT cost", "Our cost", "Result", "HPWL gap", "Area gap",
               "Boundary viol.\nGT / ours", "Grouping viol.\nGT / ours", "Solve time"]
        cells = []
        for r in rows:
            g, o = r["mg"], r["mo"]
            res_txt = "win" if o.cost < g.cost - 1e-9 else ("tie" if abs(o.cost - g.cost) <= 1e-9 else "loss")
            cells.append([str(r["t"]), str(r["n"]), f"{g.cost:.3f}", f"{o.cost:.3f}", res_txt,
                          f"{o.hpwl_gap:+.3f}", f"{o.area_gap:+.3f}",
                          f"{g.boundary_violations} / {o.boundary_violations}",
                          f"{g.grouping_violations} / {o.grouping_violations}", f"{r['secs']:.0f} s"])
        ax = fig.add_axes([0.04, 0.33, 0.92, 0.54])
        ax.axis("off")
        tb = ax.table(cellText=cells, colLabels=hdr, loc="upper center", cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(9)
        tb.scale(1, 1.9)
        for (i, j), c in tb.get_celld().items():
            if i == 0:
                c.set_facecolor("#dfe6ee")
                c.set_height(c.get_height() * 1.4)
            elif j == 4:
                c.set_facecolor({"win": "#d8f0d8", "loss": "#f6dada"}.get(cells[i - 1][4], "white"))
        w = [math.exp(r["n"] / 12) for r in rows]
        wins = sum(r["mo"].cost < r["mg"].cost - 1e-9 for r in rows)
        sg = sum(wi * r["mg"].cost for wi, r in zip(w, rows)) / sum(w)
        so = sum(wi * r["mo"].cost for wi, r in zip(w, rows)) / sum(w)
        share = sum(w) / sum(math.exp(sizes[t] / 12) for t in sizes)
        fig.text(0.06, 0.25,
                 f"We beat the ground truth on {wins} of {len(rows)}. Weighted cost over these designs "
                 f"(contest weights e^(blocks/12)): ground truth {sg:.4f}, ours {so:.4f}.\n"
                 f"These {len(rows)} designs carry {100 * share:.0f}% of the total contest weight.\n\n"
                 "Images: same axes for both layouts. Hatched = preplaced block; red outline = block that misses "
                 "its required edge\n(boundary violation); red x = pins; dashed red = pin box. The ground truth "
                 "also has boundary violations -\nthe evaluator scores it the same way as ours.",
                 fontsize=9.5, va="top")
        save(pdf, fig)

        for r in rows:
            g, o = r["mg"], r["mo"]
            colors = plt.cm.tab20(np.linspace(0, 1, r["n"]))
            fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), sharex=True, sharey=True)
            verdict = "we win" if o.cost < g.cost - 1e-9 else "ground truth wins"
            fig.suptitle(f"Design {r['t']} ({r['n']} blocks): ours {o.cost:.3f} vs ground truth {g.cost:.3f} "
                         f"- {verdict}", fontsize=13)
            panel(axes[0], r["gt"], r["cons"], r["pins"],
                  f"Ground truth - cost {g.cost:.3f}\nHPWL {g.hpwl_total:.1f}   bbox area {g.bbox_area:.0f}\n"
                  f"violations: boundary {g.boundary_violations}, grouping {g.grouping_violations}, "
                  f"MIB {g.mib_violations}", colors)
            panel(axes[1], r["ours"], r["cons"], r["pins"],
                  f"Ours - cost {o.cost:.3f}\nHPWL {o.hpwl_total:.1f} ({o.hpwl_gap:+.3f})   bbox area "
                  f"{o.bbox_area:.0f} ({o.area_gap:+.3f})\nviolations: boundary {o.boundary_violations}, "
                  f"grouping {o.grouping_violations}, MIB {o.mib_violations}", colors)
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            save(pdf, fig)
    print(f"wrote {a.out}: {len(rows)} designs {ids}")


if __name__ == "__main__":
    main()

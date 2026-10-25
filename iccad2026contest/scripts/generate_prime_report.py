"""Build prime_results_report.pdf: Team ICB on FloorSet-Prime -- the
zero-white-space partition flow, the packer flow and their per-design best
of both, with scores and layouts drawn next to the ground truth.

  python scripts/generate_prime_report.py --out prime_results_report.pdf
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon as MplPolygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from generate_code_doc_pdf import LEFT, Doc  # noqa: E402

COLOURS = {"soft": "#9ec5e8", "fixed": "#f2b36b", "preplaced": "#9a9a9a"}


def q(r):
    return r.get("quality_cost", r["cost"])


def weighted(rows):
    n_max = max(r["block_count"] for r in rows)
    w = [math.exp((r["block_count"] - n_max) / 12) for r in rows]
    return sum(q(r) * x for r, x in zip(rows, w)) / sum(w)


def role(c):
    return "preplaced" if c[1] else "fixed" if c[0] else "soft"


class ReportDoc(Doc):
    def render(self, block):
        if block[0] == "panels":
            return self.panels(*block[1:])
        if block[0] == "legend":
            return self.legend()
        return super().render(block)

    def legend(self):
        self.need(0.03)
        for k, (name, c) in enumerate((("soft block", COLOURS["soft"]), ("fixed shape", COLOURS["fixed"]),
                                       ("preplaced", COLOURS["preplaced"]))):
            x = LEFT + k * 0.2
            self.fig.patches.append(plt.Rectangle((x, self.y - 0.014), 0.02, 0.012, fc=c, ec="0.3", lw=0.5,
                                                  transform=self.fig.transFigure))
            self.fig.text(x + 0.027, self.y - 0.008, name, fontsize=8.5, va="center")
        self.fig.text(LEFT + 0.6, self.y - 0.008, "red dashes: W x H frame",
                      fontsize=8.5, va="center", color="#c03030")
        self.y -= 0.03

    def panels(self, design, title, layouts):
        """Ground truth and the flows' layouts side by side, one scale."""
        h = 0.255
        self.need(h + 0.03)
        self.text(title, size=9.5, weight="bold", dy=0.02, width=110)
        top = self.y
        W = max(max(x for v in L for x, _ in v) for _, L, _ in layouts if L)
        H = max(max(y for v in L for _, y in v) for _, L, _ in layouts if L)
        k = len(layouts)
        span = (1 - 2 * LEFT) / k
        for j, (label, verts, picked) in enumerate(layouts):
            ax = self.fig.add_axes([LEFT + j * span, top - h, span - 0.012, h - 0.03])
            ax.set_xlim(-0.02 * W, W * 1.02)
            ax.set_ylim(-0.02 * H, H * 1.02)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#2a7a2a" if picked else "0.6")
                sp.set_linewidth(2.0 if picked else 0.6)
            if not verts:
                ax.text(0.5, 0.5, "no valid layout", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_title(label, fontsize=7.5)
                continue
            for i, v in enumerate(verts):
                ax.add_patch(MplPolygon(v, closed=True, fc=COLOURS[role(design["constraints"][i])], ec="0.25", lw=0.35))
            lw = max(x for v in verts for x, _ in v)
            lh = max(y for v in verts for _, y in v)
            ax.add_patch(plt.Rectangle((0, 0), lw, lh, fill=False, ec="#c03030", lw=0.8, ls="--"))
            ax.set_title(label + ("  [picked]" if picked else ""), fontsize=7.5,
                         color="#2a7a2a" if picked else "black")
        self.y = top - h - 0.01


def build(a):
    from prime_evaluate import load_designs
    load = lambda f: {r["id"]: r for r in json.loads(Path(f).read_text())}
    P, K, B = load(a.partition), load(a.packer), load(a.best)
    A0 = load(a.adapter) if a.adapter and Path(a.adapter).exists() else None
    ids = sorted(B)
    feas = [i for i in ids if P[i]["feasible"]]
    gain = sorted(feas, key=lambda i: q(P[i]) - q(K[i]))
    show = []
    for i in gain[:3] + gain[-2:][::-1] + sorted(ids, key=lambda i: -P[i]["block_count"])[:2] \
            + [i for i in ids if not P[i]["feasible"]][:1]:
        if i not in show:
            show.append(i)
    designs = {d["id"]: d for d in load_designs(ids=show)}

    rows = [["Adapter only (preliminary run)", f"{weighted(list(A0.values())):.3f}",
             f"{sum(r['feasible'] for r in A0.values())}",
             f"{sum(r['hpwl_gap'] for r in A0.values()) / 100:+.3f}",
             f"{sum(r['area_gap'] for r in A0.values()) / 100:+.3f}",
             f"{sum(q(r) <= 1.05 for r in A0.values())}"]] if A0 else []
    for name, D in (("Packer flow (adapter + contour)", K), ("Partition flow (zero white space)", P),
                    ("Best of both, per design", B)):
        ok = [r for r in D.values() if "hpwl_gap" in r]
        rows.append([name, f"{weighted(list(D.values())):.3f}", f"{sum(r['feasible'] for r in D.values())}",
                     f"{sum(r['hpwl_gap'] for r in ok) / len(ok):+.3f}",
                     f"{sum(r['area_gap'] for r in ok) / len(ok):+.3f}",
                     f"{sum(q(r) <= 1.05 for r in D.values())}"])
    bands = []
    for lo, hi in ((21, 40), (41, 60), (61, 80), (81, 100), (101, 120)):
        s = [i for i in ids if lo <= B[i]["block_count"] <= hi]
        m = lambda D: sum(q(D[i]) for i in s) / len(s)
        bands.append([f"{lo}-{hi}", str(len(s)), f"{m(K):.3f}", f"{m(P):.3f}", f"{m(B):.3f}",
                      str(sum(B[i]["picked"] == "first" for i in s))])
    picked_p = sum(B[i]["picked"] == "first" for i in ids)

    doc = [
        ("title", "Team ICB floorplanner on FloorSet-Prime: zero-white-space results",
         "All 100 test designs  |  10/21/2026"),
        ("h1", "Summary"),
        ("table", ["Flow", "Weighted cost", "Feasible", "HPWL gap", "Area gap", "Within 5%"], rows,
         [0.38, 0.13, 0.1, 0.13, 0.13, 0.13]),
        ("p", "Costs leave out the contest's runtime term (reduced budgets, shared machine). The ground "
              "truth scores exactly 1.000, which is also the lowest possible cost: gaps are floored at 0, "
              "so the aim is to match it. 'Within 5%' counts designs with cost at most 1.05."),
        ("p", f"The best-of-both flow picks, per design and without the ground truth, the cheaper of the "
              f"two layouts on the contest formula; it chose the partition layout on {picked_p} designs. "
              "Picking with the ground truth instead would give "
              f"{weighted([min(P[i], K[i], key=q) for i in ids]):.3f}, so the solve-time choice is close "
              "to ideal."),
        ("h1", "The two flows"),
        ("p", "Packer flow: the Lite contest pipeline on a rectangle version of each design (soft blocks "
              "as rectangles, fixed polygons as their bounding box, preplaced polygons split into pinned "
              "pieces), then the sequence-pair packer. It leaves white space, so its area is about 10% above "
              "the ground truth."),
        ("p", "Partition flow: the frame is taken from the pins (exact on all 100 designs, and its area "
              "equals the total block area), then divided by a slicing tree whose cuts are placed at exact "
              "areas. White space is zero by construction and the area gap is exactly 0 on every valid "
              "layout. Soft blocks take whatever rectilinear shape the cuts leave (L, T, notched and "
              "interlocking shapes, at most 16 corners); fixed blocks are leaves placed in a corner of their "
              "region, so their neighbours wrap around them; enclosed pockets between preplaced blocks go to "
              "a block of matching area. Six annealing searches run in parallel per design, seeded half by "
              "a centroid predictor being trained on the Prime training set and half by the analytic placer."),
        ("h2", "By design size (mean cost)"),
        ("table", ["Blocks", "Designs", "Packer", "Partition", "Best of both", "Partition picked"], bands,
         [0.14, 0.12, 0.16, 0.16, 0.2, 0.22]),
        ("p", "The partition flow's two invalid designs (63 and 75, cost 10 each) are both in the 81-100 "
              "band, hence its mean of 2.09 there; best of both falls back to the packer layout for them. Where both are valid the flows score the same on "
              "average, but each wins on about half the designs, which is why combining them helps."),
        ("h1", "Layouts"),
        ("p", "Each row: the ground truth, the partition flow and the packer flow at one scale; the green "
              "frame marks the layout the best-of-both flow picked. First the designs where the partition "
              "flow gains most, then where the packer flow is better, then the largest designs and a "
              "partition failure."),
        ("legend",),
    ]
    for i in show:
        d = designs[i]
        title = (f"Design {i} ({d['block_count']} blocks): partition {q(P[i]):.3f}, packer {q(K[i]):.3f}"
                 + (f"  |  partition HPWL gap {P[i]['hpwl_gap']:+.3f}, area gap {P[i]['area_gap']:+.3f}"
                    if P[i]["feasible"] else "  |  partition: no valid layout"))
        doc.append(("panels", d, title, [
            ("Ground truth (1.000)", d["gt"], False),
            (f"Partition flow ({q(P[i]):.3f})" if P[i]["feasible"] else "Partition flow",
             P[i]["layout"] if P[i]["feasible"] else None, B[i]["picked"] == "first"),
            (f"Packer flow ({q(K[i]):.3f})", K[i]["layout"], B[i]["picked"] == "second"),
        ]))
    doc += [
        ("h1", "Next steps"),
        ("p", "1. Wirelength in the partition flow (+0.15 vs the ground truth): tree moves aimed at net "
              "partners. 2. The two partition failures: leaves pinched around interior preplaced blocks. "
              "3. Rerun with the finished centroid predictor (training ends about 7 am). 4. Contest "
              "budgets (packer 200, 3 restarts) instead of the reduced ones used here."),
    ]
    with PdfPages(Path(a.out)) as pdf:
        d = ReportDoc(pdf)
        for b in doc:
            d.render(b)
        d.close()
    print(f"wrote {a.out}: {d.page - 1} pages; designs shown {show}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prime_results_report.pdf")
    ap.add_argument("--partition", default="prime_results/partition_full.json")
    ap.add_argument("--packer", default="prime_results/contour_pack20.json")
    ap.add_argument("--best", default="prime_results/best_of_both.json")
    ap.add_argument("--adapter", default="prime_results/prelim_pack20.json")
    build(ap.parse_args())


if __name__ == "__main__":
    main()

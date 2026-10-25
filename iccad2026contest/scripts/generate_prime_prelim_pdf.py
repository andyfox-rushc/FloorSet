"""Build prime_prelim_results.pdf: Team ICB's first FloorSet-Prime (rectilinear
polygon) results -- method, scores per design and layouts next to the ground
truth. Reads result files written by prime_evaluate.py --output; a full
100-design run is summarized too when its file exists.

  python scripts/generate_prime_prelim_pdf.py --out prime_prelim_results.pdf
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

# Adapter-only smoke run (10-20, packer 10, one solve): the printed costs included the
# runtime term over a 5-design median, divided out by hand to quality costs.
ADAPTER_ONLY = {0: 1.054, 1: 1.048, 2: 1.165, 50: 1.349, 99: 1.452}
ROLE_COLOURS = {"soft": "#9ec5e8", "fixed": "#f2b36b", "preplaced": "#9a9a9a"}


def _weighted(rows, key):
    n_max = max(r["block_count"] for r in rows)
    w = [math.exp((r["block_count"] - n_max) / 12) for r in rows]
    return sum(r[key] * x for r, x in zip(rows, w)) / sum(w)


def total_runtime_weighted(rows):
    return _weighted(rows, "cost")


def _role(c):
    return "preplaced" if c[1] else "fixed" if c[0] else "soft"


class ResultsDoc(Doc):
    def render(self, block):
        if block[0] == "layouts":
            return self.layouts(*block[1:])
        return super().render(block)

    def layouts(self, design, layout, title):
        """Ground truth and our layout side by side, same scale."""
        h = 0.33
        self.need(h + 0.03)
        self.text(title, size=10, weight="bold", dy=0.022)
        top = self.y
        polys = [d for d in (design["gt"], layout)]
        W = max(max(x for v in p for x, _ in v) for p in polys)
        H = max(max(y for v in p for _, y in v) for p in polys)
        for k, (verts, label) in enumerate(((design["gt"], "Ground truth"), (layout, "Team ICB"))):
            ax = self.fig.add_axes([LEFT + k * 0.43, top - h, 0.41, h - 0.02])
            for i, v in enumerate(verts):
                ax.add_patch(MplPolygon(v, closed=True, fc=ROLE_COLOURS[_role(design["constraints"][i])],
                                        ec="0.25", lw=0.4))
            lw, lh = max(x for q in verts for x, _ in q), max(y for q in verts for _, y in q)
            ax.add_patch(plt.Rectangle((0, 0), lw, lh, fill=False, ec="#c03030", lw=0.8, ls="--"))
            ax.set_xlim(-0.02 * W, W * 1.02)
            ax.set_ylim(-0.02 * H, H * 1.02)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{label}: bbox {lw:.0f} x {lh:.0f}", fontsize=8)
        self.y = top - h - 0.012


def build(args):
    from prime_evaluate import load_designs
    smoke = json.loads(Path(args.smoke).read_text())
    full = json.loads(Path(args.full).read_text()) if args.full and Path(args.full).exists() else None
    designs = {d["id"]: d for d in load_designs(ids=[r["id"] for r in smoke])}
    rows = []
    for r in smoke:
        rows.append([str(r["id"]), str(r["block_count"]), f"{ADAPTER_ONLY.get(r['id'], float('nan')):.3f}",
                     f"{r['quality_cost']:.3f}", f"{r['hpwl_gap']:+.3f}", f"{r['area_gap']:+.3f}",
                     f"{r['v_rel']:.3f}", "yes" if r["feasible"] else "no", f"{r['runtime']:.0f} s"])
    wq = _weighted(smoke, "quality_cost")
    wa = _weighted([dict(r, a=ADAPTER_ONLY[r["id"]]) for r in smoke], "a")

    doc = [
        ("title", "Team ICB floorplanner on FloorSet-Prime: preliminary results",
         "Rectilinear polygon blocks  |  10/20/2026"),
        ("h1", "Summary"),
        ("table", ["Measure", "Value"], [
            ["Designs solved with the contour packer", f"{len(smoke)} of 100 test designs"],
            ["Feasible layouts", f"{sum(r['feasible'] for r in smoke)} / {len(smoke)}"],
            ["Weighted cost, these designs (ground truth = 1.000)", f"{wq:.3f}"],
            ["Same designs, before the contour packer", f"{wa:.3f}"],
            ["Mean wirelength / area gap vs ground truth",
             f"{sum(r['hpwl_gap'] for r in smoke) / len(smoke):+.3f} / {sum(r['area_gap'] for r in smoke) / len(smoke):+.3f}"],
            ["Shelf-packing baseline, all 100 designs", "7.515"],
        ] + ([["Full 100-design run (adapter only, packer 20)",
               f"{_weighted(full, 'quality_cost'):.3f}, feasible {sum(r['feasible'] for r in full)}/100"]]
             if full else []), [0.62, 0.38]),
        ("p", "Costs leave out the contest's runtime term: these runs used reduced budgets and shared "
              "the machine, so runtimes are not representative. Lower is better; 1.0 is the ground "
              "truth's score and an infeasible layout scores 10. Each number is a single solve, and single "
              "designs vary by up to about +-0.05 between runs."),
        ("h1", "What was run"),
        ("p", "FloorSet-Prime has the same inputs as FloorSet-Lite, but every block is a rectilinear "
              "polygon (rectangles, L, T, U and step shapes; 40% of blocks are not rectangles) and the "
              "ground-truth layouts tile their bounding box with no white space at all. Prime defines "
              "constraint checks but no score, so the scorer (prime_evaluate.py) applies the contest "
              "formula with Prime's geometry: wirelength between polygon centroids, area of the "
              "[0, W] x [0, H] frame, overlap and +-1% area as hard constraints, and fixed shapes equal up "
              "to rotation. Scored this way the ground truth is exactly 1.000 on all 100 designs."),
        ("p", "The solver is the Lite contest pipeline (RL layout, best-of candidates, sequence-pair "
              "packer, finishing steps) run on a rectangle version of each design "
              "(rl/prime_adapter.py): soft blocks stay rectangles; preplaced polygons are split into "
              "their rectangle pieces, each pinned in place, so other blocks can fill their notches; "
              "fixed polygons are placed as their bounding box. The contour packer "
              "(rl/packer.py) then lets a block pack against the actual pieces of its left neighbours "
              "rather than their boxes, which can place it inside a fixed polygon's notch; every "
              "keep-if-better check uses exact polygon geometry."),
        ("table", ["Setting", "Value"], [
            ["Packer budget (FLOORSET_PACK)", "10 s x (n / 77)^2 per job (contest default 200)"],
            ["Whole-pipeline restarts (FLOORSET_SOLVE_RUNS)", "1 (contest default 3)"],
            ["Worker processes", "2 (shared machine)"],
            ["Policy / learned placers", "Lite checkpoints, not retrained on Prime"],
        ], [0.5, 0.5]),
        ("h1", "Contour packer: results per design"),
        ("table", ["Design", "Blocks", "Before", "Contour", "HPWL gap", "Area gap", "V_rel", "Feasible", "Time"],
         rows, [0.09, 0.09, 0.1, 0.1, 0.12, 0.12, 0.1, 0.12, 0.16]),
        ("p", "'Before' is the adapter alone, without the contour packer (same budget, one solve). "
              "Area is the larger gap on 4 of the 5 designs: the ground truth fills its frame "
              "completely, while our soft blocks are still rectangles and fixed polygons only partly "
              "use their notches."),
    ]
    if full:
        bands = [(21, 40), (41, 60), (61, 80), (81, 100), (101, 120)]
        brow = []
        for lo, hi in bands:
            b = [r for r in full if lo <= r["block_count"] <= hi]
            m = lambda k: sum(r[k] for r in b) / len(b)
            brow.append([f"{lo}-{hi}", str(len(b)), f"{m('quality_cost'):.3f}", f"{m('hpwl_gap'):+.3f}",
                         f"{m('area_gap'):+.3f}", f"{m('v_rel'):.3f}", f"{m('runtime'):.0f} s"])
        q = sorted(r["quality_cost"] for r in full)
        doc += [
            ("h1", "Full 100-design run (adapter only)"),
            ("p", "All 100 test designs, before the contour packer existed: packer budget 20, one solve "
                  "per design, 6 worker processes on an otherwise idle 8-core machine. Weighted cost "
                  f"{_weighted(full, 'quality_cost'):.3f} without the runtime term "
                  f"({total_runtime_weighted(full):.3f} with it), feasible "
                  f"{sum(r['feasible'] for r in full)} / 100, median cost {q[50]:.3f}, best {q[0]:.3f}, "
                  f"worst {q[-1]:.3f}; {sum(c <= 1.05 for c in q)} designs within 5% of the ground truth, "
                  "none below it yet."),
            ("table", ["Blocks", "Designs", "Mean cost", "HPWL gap", "Area gap", "V_rel", "Mean time"],
             brow, [0.14, 0.12, 0.14, 0.15, 0.15, 0.13, 0.17]),
            ("p", "The largest designs cost the most, and they dominate the weighted score (weight "
                  "exp(n / 12)). Wirelength and area gaps are about equal overall; soft-constraint "
                  "violations are small."),
        ]
    doc += [
        ("page",),
        ("h1", "Layouts"),
        ("p", "Blue: soft blocks; orange: fixed shapes; grey: preplaced. The dashed red line is the "
              "[0, W] x [0, H] frame the area is measured on. Both panels of a design share one scale."),
    ]
    for r in smoke:
        d = designs[r["id"]]
        doc.append(("layouts", d, r["layout"],
                    f"Design {r['id']} ({r['block_count']} blocks): cost {r['quality_cost']:.3f}, "
                    f"HPWL gap {r['hpwl_gap']:+.3f}, area gap {r['area_gap']:+.3f}"))
    doc += [
        ("h1", "Next steps"),
        ("p", "1. Score all 100 designs with the contour packer at the same budget, then at "
              "contest settings. 2. Let fixed polygons rotate (Prime allows it). 3. Make the vertical "
              "packing contour-aware too: today only x packs against pieces, so some notch orientations "
              "fill only by chance. 4. Allow rectilinear soft-block shapes, the main lever on the area "
              "gap. 5. Retrain the policy and learned placers on FloorSet-Prime."),
    ]
    with PdfPages(Path(args.out)) as pdf:
        d = ResultsDoc(pdf)
        for b in doc:
            d.render(b)
        d.close()
    print(f"wrote {args.out}: {d.page - 1} pages")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prime_prelim_results.pdf")
    ap.add_argument("--smoke", default="prime_results/smoke5_contour.json")
    ap.add_argument("--full", default="prime_results/prelim_pack20.json")
    build(ap.parse_args())


if __name__ == "__main__":
    main()

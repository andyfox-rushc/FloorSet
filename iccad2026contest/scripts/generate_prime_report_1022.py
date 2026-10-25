"""Build prime_report_1022.pdf: Team ICB on FloorSet-Prime after the 10-21 night --
the approach (RPolyPlan / JigsawPlanner plan with reinforcement learning, what
was built, what worked), a summary table, all 100 designs, and the 5 anchor
designs drawn next to the ground truth.

  python scripts/generate_prime_report_1022.py --out prime_report_1022.pdf

Flows are given as label=file pairs (prime_evaluate.py result files); the
best-of-all column picks per design without the ground truth, as
prime_combined.py does (prime_evaluate.relative_cost against the first flow).
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from generate_prime_report import ReportDoc, q, weighted  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ANCHORS = [0, 24, 49, 74, 98]


def summary_row(name, D):
    ok = [r for r in D.values() if "hpwl_gap" in r]
    n = len(D)
    return [name, f"{weighted(list(D.values())):.4f}", f"{sum(r['feasible'] for r in D.values())}/{n}",
            f"{sum(r['hpwl_gap'] for r in ok) / max(len(ok), 1):+.3f}",
            f"{sum(r['area_gap'] for r in ok) / max(len(ok), 1):+.3f}",
            f"{sum(r['v_rel'] for r in ok) / max(len(ok), 1):.3f}",
            f"{sum(q(r) <= 1.05 for r in D.values())}",
            f"{sum(r['feasible'] and q(r) <= 1.0 + 1e-6 for r in D.values())}",
            f"{sum(r.get('hpwl_gap', 1) < 0 for r in D.values() if r['feasible'])}"]


def best_of(flows, designs):
    """Per design, the flow whose layout is cheapest on the contest formula
    relative to the first flow's layout (no ground truth)."""
    import prime_evaluate as E
    names = list(flows)
    out = {}
    for i, d in designs.items():
        args = (d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"], d["constraints"],
                d["target_polygons"])
        ref = next((flows[n][i]["layout"] for n in names if flows[n].get(i, {}).get("layout")), None)
        best = None
        for n in names:
            r = flows[n].get(i)
            if not r or not r.get("feasible") or not r.get("layout"):
                continue
            c, ok = E.relative_cost(args, r["layout"], ref)
            if ok and (best is None or c < best[0]):
                best = (c, n)
        if best is None:
            out[i] = dict(flows[names[0]][i], picked=None)
        else:
            out[i] = dict(flows[best[1]][i], picked=best[1])
    return out


def auto_summary(flows, B, a):
    """Headline sentences from the numbers (used when the notes say AUTO)."""
    vals = list(B.values())
    w = weighted(vals)
    n1 = sum(r["feasible"] and q(r) <= 1.0 + 1e-6 for r in vals)
    nh = sum(r["feasible"] and r.get("hpwl_gap", 1) < 0 for r in vals)
    n5 = sum(q(r) <= 1.05 for r in vals)
    out = [f"Best layout per design over all flows: weighted cost {w:.4f} on the 100 test designs, "
           f"{sum(r['feasible'] for r in vals)} valid; {n1} designs at 1.000 (they match or beat the ground "
           f"truth), {nh} with shorter wirelength than the ground truth, {n5} within 5%. The 10/20 report's best "
           f"of both was 1.160."]
    base = [k for k in flows if k.startswith("Partition (t80")]
    if base:
        b = flows[base[0]]
        out.append(f"The partition flow alone (cache fixed, before refinement) scores {weighted(list(b.values())):.4f}; "
                   "the pair re-cut refinement accounts for most of the gain over it.")
    return out


def build(a):
    from prime_evaluate import load_designs
    load = lambda f: {r["id"]: r for r in json.loads(Path(f).read_text())}
    flows = {}
    for spec in a.flow:
        label, f = spec.split("=", 1)
        if Path(f).exists():
            flows[label] = load(f)
    designs = {d["id"]: d for d in load_designs()}
    pick_from = {k: v for k, v in flows.items() if k in a.pick} if a.pick else flows
    B = best_of(pick_from, designs)
    ids = sorted(designs)
    notes = json.loads(Path(a.notes).read_text()) if a.notes and Path(a.notes).exists() else {}

    rows = [summary_row(k, v) for k, v in flows.items()] + [summary_row("Best of all, picked per design", B)]
    head = ["Flow", "Weighted", "Valid", "HPWL", "Area", "V_rel", "<=1.05", "=1.000", "HPWL<GT"]
    widths = [0.34, 0.09, 0.08, 0.08, 0.08, 0.07, 0.08, 0.08, 0.10]

    bands = []
    labels = list(flows)
    for lo, hi in ((21, 40), (41, 60), (61, 80), (81, 100), (101, 120)):
        s = [i for i in ids if lo <= designs[i]["block_count"] <= hi]
        bands.append([f"{lo}-{hi}", str(len(s))] + [f"{sum(q(flows[k][i]) for i in s) / len(s):.3f}" for k in labels]
                     + [f"{sum(q(B[i]) for i in s) / len(s):.3f}"])

    short = notes.get("short", {})
    def origin(i):
        """The flow a design's picked layout came from (through the final polish)."""
        lab = B[i].get("picked")
        for _ in range(3):                         # through both final steps
            if lab and lab.startswith("Final") and lab in flows and flows[lab][i].get("picked"):
                lab = flows[lab][i]["picked"]
        return lab

    per = []
    for i in ids:
        r = dict(B[i], picked=origin(i))
        per.append([str(i), str(designs[i]["block_count"])]
                   + [f"{q(flows[k][i]):.3f}" if i in flows[k] else "-" for k in labels]
                   + [f"{q(r):.3f}", f"{r.get('hpwl_gap', float('nan')):+.3f}" if r.get("feasible") else "-",
                      short.get(r.get("picked"), r.get("picked") or "-")])

    doc = [("title", notes.get("title", "Team ICB on FloorSet-Prime"), notes.get("subtitle", ""))]
    doc += [("h1", "Summary"), ("table", head, rows, widths)]
    for p in notes.get("summary", []):
        if p == "AUTO":
            for line in auto_summary(flows, B, a):
                doc.append(("p", line))
        else:
            doc.append(("p", p))
    for sec in notes.get("sections", []):
        doc.append(("h1", sec["h"]))
        for p in sec["p"]:
            doc.append(("table",) + tuple(p[1:]) if isinstance(p, list) and p and p[0] == "table" else ("p", p))
    doc += [("h2", "Mean cost by design size"),
            ("table", ["Blocks", "Designs"] + [short.get(k, k) for k in labels] + ["Best"], bands,
             [0.1, 0.09] + [0.81 / (len(labels) + 1)] * (len(labels) + 1))]
    doc += [("h1", "All 100 designs"),
            ("p", notes.get("table_note", "Quality cost per design (no runtime term); 1.000 is the ground "
                                          "truth's own score and the floor.")),
            ("table", ["ID", "Blocks"] + [short.get(k, k) for k in labels] + ["Best", "HPWL gap", "Picked"], per,
             [0.06, 0.07] + [0.6 / len(labels)] * len(labels) + [0.09, 0.1, 0.08])]
    doc += [("h1", "Anchor designs"),
            ("p", notes.get("anchor_note", "")), ("legend",)]
    show = [k for k in a.panels if k in flows]
    for i in ANCHORS:
        d = designs[i]
        title = f"Design {i} ({d['block_count']} blocks): " + ", ".join(
            f"{short.get(k, k)} {q(flows[k][i]):.3f}" for k in show) + f"  |  best {q(B[i]):.3f}"
        doc.append(("panels", d, title, [("Ground truth (1.000)", d["gt"], False)] + [
            (f"{short.get(k, k)} ({q(flows[k][i]):.3f})", flows[k][i]["layout"] if flows[k][i].get("feasible") else None,
             False) for k in show] + [(f"Final, from {short.get(origin(i), origin(i))} ({q(B[i]):.3f})",
                                       B[i].get("layout"), True)]))
    for sec in notes.get("after", []):
        doc.append(("h1", sec["h"]))
        for p in sec["p"]:
            doc.append(("p", p))
    with PdfPages(Path(a.out)) as pdf:
        d = ReportDoc(pdf)
        for b in doc:
            d.render(b)
        d.close()
    Path(a.out).with_suffix(".best.json").write_text(json.dumps(list(B.values())))
    print(f"wrote {a.out}: {d.page - 1} pages")
    for r in rows:
        print("  " + " | ".join(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prime_report_1022.pdf")
    ap.add_argument("--flow", action="append", default=[], help="label=results.json (repeatable, in column order)")
    ap.add_argument("--pick", action="append", default=[], help="labels the best-of may pick from (default all)")
    ap.add_argument("--panels", action="append", default=[], help="labels drawn next to the ground truth")
    ap.add_argument("--notes", default="scripts/prime_report_1022_text.json")
    build(ap.parse_args())


if __name__ == "__main__":
    main()

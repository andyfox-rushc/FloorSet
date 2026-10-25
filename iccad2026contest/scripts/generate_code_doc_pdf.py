"""Build floorplan_code.pdf: the files needed to build and run Team ICB's
floorplanner, how to get, install, run and verify it, and how each model
checkpoint was trained. matplotlib PdfPages only (no extra dependency).

  python scripts/generate_code_doc_pdf.py --out floorplan_code.pdf
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

W, H = 8.5, 11            # portrait letter
LEFT, TOP, BOTTOM = 0.08, 0.95, 0.055
REPO = "git@github.com:andyfox-rushc/FloorSet.git"

# Document content: (kind, ...) blocks laid out top to bottom with page breaks.
#   ("h1", text) ("h2", text) ("p", text) ("code", [lines]) ("table", header, rows, widths) ("page",)
DOC = [
    ("title", "Team ICB floorplanner: code, build and reproduction guide",
     "ICCAD 2026 CAD Contest, Problem C (FloorSet-Lite)  |  10/18/2026"),
    ("h1", "Latest results"),
    ("table", ["Measure (contest evaluator, 100 validation designs)", "Team ICB 10/18"], [
        ["Official weighted score (exp(blocks/12) weights)", "1.0343"],
        ["Ground-truth layouts, scored by the same evaluator", "1.108"],
        ["Published leader (10/6 leaderboard, raw cost)", "1.084"],
        ["Designs where we beat the ground truth's own score", "76 / 100"],
        ["Feasible layouts", "100 / 100"],
        ["Mean HPWL gap / area gap / relative violations", "0.019 / 0.025 / 0.004"],
        ["Mean runtime per design (8-core laptop, see 6)", "641 s"],
    ], [0.72, 0.28]),
    ("p", "Reference result file: iccad2026contest/official_eval_1018_restart.json (every design's layout, "
          "cost and runtime). 1.0 means the ground truth's wirelength and area; the ground truth itself "
          "scores 1.108 because it breaks some of its own edge and cluster constraints."),
    ("p", "This document is for someone who wants to reproduce and verify these results: what to "
          "check out, how to install and run it, how to compare their numbers with ours, and how "
          "every trained model was produced."),

    ("h1", "1. Get the code"),
    ("code", [f"git clone {REPO}",
              "cd FloorSet",
              "# all solver commands below run from FloorSet/iccad2026contest"]),
    ("p", "The repository is a fork of Intel's FloorSet repository. The top level holds Intel's "
          "dataset loaders and the 100 validation designs (LiteTensorDataTest/); our solver lives "
          "in iccad2026contest/ next to the contest evaluator."),

    ("h1", "2. Install"),
    ("p", "No compilation step: everything is Python. Tested with Python 3.14.4 and the versions "
          "below (requirements.txt gives the minimum versions):"),
    ("table", ["Package", "Tested version"], [
        ["torch (CPU build is enough)", "2.14.0"], ["numpy", "2.5.2"], ["shapely", "2.1.2"],
        ["matplotlib", "3.11.1"], ["tqdm", "4.70.0"], ["requests", "2.34.2"],
    ], [0.6, 0.4]),
    ("code", ["cd FloorSet",
              "python3 -m venv venv",
              "source venv/bin/activate",
              "pip install -r iccad2026contest/requirements.txt"]),
    ("p", "The scripts call ../venv/bin/python from iccad2026contest/, i.e. the venv inside FloorSet/. "
          "The validation data is in the repository; the 1M-design training set (about 15 GB "
          "download, 24 GB unpacked, into FloorSet/floorset_lite/) is only needed to retrain models "
          "(section 7) and downloads automatically on first use."),

    ("h1", "3. Run and verify"),
    ("h2", "Full run (the official number)"),
    ("code", ["cd iccad2026contest",
              "../venv/bin/python iccad2026_evaluate.py --evaluate my_optimizer.py \\",
              "    --output my_run.json            # ~18 h on 8 cores",
              "../venv/bin/python scripts/compare_with_gt.py my_run.json official_eval_1018_restart.json"]),
    ("p", "The evaluator prints the total score; compare_with_gt.py scores the ground-truth layouts "
          "with the same evaluator and prints, for each result file, the weighted score, the mean "
          "gaps, how many designs beat the ground truth and the largest losses. All defaults are "
          "the configuration that produced 1.0343; no environment variables are needed. The count of "
          "designs that beat the ground truth moves more than the score between runs (76 in this run, "
          "77 in the 10/17 run at 1.0404, 72 in the 10/16 run at 1.0432)."),
    ("h2", "Quick check (about 15 + 35 minutes)"),
    ("code", ["../venv/bin/python scripts/solver_ab.py --ids 0 98 --out quick.json",
              "../venv/bin/python scripts/anchor_results.py --test-ids 0 49 98 --out-dir anchors"]),
    ("p", "solver_ab.py solves chosen designs one at a time exactly as the evaluator does. Ours: "
          "design 0 1.000 and design 98 1.000 in the official run; 1.000-1.012 and 1.000 over three "
          "further runs (ground truth 1.091 and 1.080). Design 98 takes about 16 minutes, design 49 "
          "about 17 (it keeps a violation, so it always gets 3 solves). "
          "anchor_results.py also draws each layout next to the ground truth (PNG files)."),
    ("h2", "What to expect"),
    ("p", "Results vary from run to run: the packer searches for a fixed wall-clock time, so "
          "machine speed and load change how far it gets. Our repeated runs of one configuration "
          "differ by about +-0.01 on the weighted score, and single designs much more (design 49: "
          "1.127-1.274 over our runs). Run on an otherwise idle machine; a slower or busy machine "
          "scores worse. Six worker processes are used (FLOORSET_PROCS, default min(6, cpus))."),

    ("page",),
    ("h1", "4. Files needed"),
    ("p", "Paths relative to FloorSet/. 'Intel' = unchanged from the upstream repository."),
    ("h2", "Solver (run time)"),
    ("table", ["File", "Role"], [
        ["iccad2026contest/my_optimizer.py", "Entry point (MyOptimizer.solve), the whole pipeline"],
        ["iccad2026contest/iccad2026_evaluate.py", "Contest evaluator (Intel, unchanged)"],
        ["iccad2026contest/rl/__init__.py", "Package marker"],
        ["iccad2026contest/rl/data.py", "Contest inputs -> FloorplanInstance"],
        ["iccad2026contest/rl/ordering.py", "Placement order for the RL policy"],
        ["iccad2026contest/rl/env.py", "Grid placement environment (hard constraints)"],
        ["iccad2026contest/rl/encoder.py", "Graph network over the netlist"],
        ["iccad2026contest/rl/networks.py", "RL policy / value networks"],
        ["iccad2026contest/rl/ppo.py", "PPO rollouts and updates"],
        ["iccad2026contest/rl/reward.py", "Contest cost as the RL reward"],
        ["iccad2026contest/rl/finetune.py", "Per-design PPO fine-tuning of the policy"],
        ["iccad2026contest/rl/checkpoint_meta.py", "Checkpoint metadata (canvas utilization)"],
        ["iccad2026contest/rl/compaction.py", "Gravity compaction"],
        ["iccad2026contest/rl/anneal.py", "Annealing polish"],
        ["iccad2026contest/rl/repack.py", "Re-packing polish, exact cost helpers"],
        ["iccad2026contest/rl/slicing.py", "Slicing polish"],
        ["iccad2026contest/rl/legalize.py", "Wall repair / legalization"],
        ["iccad2026contest/rl/analytic.py", "Analytic placers, learned candidates, best-of"],
        ["iccad2026contest/rl/learned_placer.py", "Learned placer network (centres, relations)"],
        ["iccad2026contest/rl/sizing.py", "Differentiable shape fitting"],
        ["iccad2026contest/rl/packer.py", "Sequence-pair packer, squeeze, wirelength relaxation"],
        ["cost.py utils.py visualize.py", "Intel helpers imported by the evaluator"],
        ["liteLoader.py lite_dataset.py litetestLoader.py", "Intel data loaders"],
        ["lite_dataset_test.py prime_dataset.py", "Intel data loaders"],
        ["LiteTensorDataTest/ (200 files)", "Intel: the 100 validation designs + labels"],
    ], [0.55, 0.45]),
    ("h2", "Trained models (iccad2026contest/checkpoints/)"),
    ("table", ["File", "Size", "Role"], [
        ["policy.pt + policy.pt.meta.json", "304 KB", "RL placement policy (PPO), canvas utilization 0.70"],
        ["placer_a.pt", "1.3 MB", "Learned placer A: block centres + aspect ratios"],
        ["placer_r.pt", "7.4 MB", "Relation network 1 (pairwise relations + centres)"],
        ["placer_relB.pt", "7.4 MB", "Relation network 2 (independent training run)"],
    ], [0.4, 0.12, 0.48]),
    ("h2", "Training, verification and documentation"),
    ("table", ["File", "Role"], [
        ["iccad2026contest/rl/train.py, rl/pretrain.py", "RL policy training (python -m rl.train)"],
        ["iccad2026contest/scripts/train_placer.py", "Learned placer / relation network training"],
        ["iccad2026contest/scripts/polish_bench.py", "Shared evaluator setup used by the scripts"],
        ["iccad2026contest/scripts/__init__.py", "Package marker"],
        ["iccad2026contest/scripts/compare_with_gt.py", "Verify a result file against the ground truth"],
        ["iccad2026contest/scripts/solver_ab.py", "Solve chosen designs one at a time"],
        ["iccad2026contest/scripts/anchor_results.py", "Solve + draw layouts vs ground truth"],
        ["iccad2026contest/scripts/big_designs_pdf.py", "Ground truth vs ours for the largest designs (PDF)"],
        ["iccad2026contest/scripts/generate_code_doc_pdf.py", "Builds this document"],
        ["iccad2026contest/official_eval_1018_restart.json", "Reference result (1.0343)"],
        ["iccad2026contest/algorithm.md", "Design notes and every experiment, dated"],
        ["iccad2026contest/requirements.txt", "Python packages"],
    ], [0.55, 0.45]),
    ("p", "We checked this list by copying only these files into an empty directory and running "
          "the solver, the placer training and the policy training from there."),

    ("page",),
    ("h1", "5. How the solver works"),
    ("p", "For each design my_optimizer.py builds candidate layouts, packs the most promising ones "
          "with a long search, and returns the cheapest by the exact contest cost:"),
    ("table", ["Stage", "What it does"], [
        ["1. RL policy", "policy.pt places blocks one at a time on a grid (GNN encoder + CNN position "
                         "head), fine-tuned on the design with PPO for a few seconds ('base')"],
        ["2. Analytic placers", "Five wirelength-driven global placements from scratch at canvas "
                                "utilizations 0.75-1.0 ('scratch...' candidates)"],
        ["3. Learned placers", "Networks trained on 1M ground-truth layouts predict block positions "
                               "and pairwise left/right/below/above relations ('learned1-4')"],
        ["4. Shape fitting", "Gradient descent on soft-block aspect ratios for a fixed arrangement"],
        ["5. Best-of", "All candidates scored; the best plus the four learned ones go on"],
        ["6. Packer", "Sequence-pair annealing on the exact contest cost, 200 s at 77 blocks "
                      "(scaled by (blocks/77)^2), one process per layout, shape refit halfway; "
                      "split clusters penalized in the search and repaired exactly afterwards"],
        ["7. Second round", "The winner is packed once more from where it ended, with a search penalty "
                            "for running past the outline revealed by preplaced edge blocks; kept only if "
                            "cheaper (new 10/17)"],
        ["8. Finish", "Squeeze, wirelength relaxation, then the cheapest layout inside the box "
                      "where the pins sit (preferred within 5% of the best cost); soft blocks up to 16:1 throughout"],
        ["9. Restarts", "While the cheapest layout still breaks a soft constraint, stages 1-8 run "
                        "again with new seeds (at most 3 runs), the reruns also aiming for the outline; "
                        "cheapest kept (new 10/18)"],
    ], [0.22, 0.78]),
    ("h2", "The policy and models used"),
    ("table", ["Candidate", "Checkpoint", "Arrangement from"], [
        ["base", "policy.pt", "RL policy"],
        ["scratch0.75 ... scratch1.0", "(none)", "analytic placement"],
        ["learned", "placer_a.pt", "predicted centres"],
        ["learned2", "placer_r.pt", "predicted pairwise relations"],
        ["learned3", "placer_r.pt", "predicted centres"],
        ["learned4", "placer_relB.pt", "predicted pairwise relations"],
    ], [0.3, 0.3, 0.4]),
    ("h2", "Settings (defaults = the 1.0343 run)"),
    ("table", ["Environment variable", "Default", "Meaning"], [
        ["FLOORSET_PACK", "200", "Packer seconds per layout at 77 blocks"],
        ["FLOORSET_PROCS", "min(6, cpus)", "Worker processes"],
        ["FLOORSET_MAX_ASPECT", "16", "Longest soft-block side / shortest (was 8 until 10/15)"],
        ["FLOORSET_REATTACH", "1", "Exact repair of split clusters after packing (new 10/16)"],
        ["FLOORSET_PACK_GAP", "1", "Search penalty on cluster-piece distance (new 10/16)"],
        ["FLOORSET_REPACK_WINNER", "1", "Second packer round from the winner, kept if cheaper (new 10/17)"],
        ["FLOORSET_WALL_PASS", "100", "Search penalty past preplaced walls in that round (new 10/17)"],
        ["FLOORSET_SOLVE_RUNS", "3", "Most whole solves per design (new 10/18)"],
        ["FLOORSET_RESTART_VIOL", "1", "Solve again only while soft violations remain (new 10/18)"],
        ["FLOORSET_CANDIDATES", "base,scratch0.75..1.0,learned..learned4", "Candidate layouts"],
        ["FLOORSET_CHECKPOINT", "checkpoints/policy.pt", "RL policy"],
        ["FLOORSET_PLACER, ..._PLACER4", "see table above", "Learned placer checkpoints"],
    ], [0.34, 0.33, 0.33]),
    ("p", "algorithm.md documents every other switch, with the measurement that set its default."),

    ("h1", "6. Machine used for the results"),
    ("p", "Intel Core Ultra 7 258V (8 cores), 16 GB RAM, Windows 11 + WSL2 (Linux 6.18), CPU only. "
          "One official run of the 100 designs took about 18 hours (641 s per design on average, "
          "most of it the packer's fixed time budget)."),

    ("page",),
    ("h1", "7. Retraining the models (optional)"),
    ("p", "The checkpoints in the repository are the ones that produced the results. To retrain "
          "them, run the commands below from iccad2026contest/. Training stops on a wall-clock "
          "limit and runs on several threads, so a retrained model will not be bit-identical; "
          "expect the solver's score to land within the run-to-run noise rather than exactly. Judge "
          "a retrained model by a full evaluation run (section 3), not by its validation error: "
          "training network 1 on to 358k designs lowered that error but scored worse (1.0709)."),
    ("h2", "Training data"),
    ("code", ["../venv/bin/python -c \"from iccad2026_evaluate import get_training_dataloader; \\",
              "    get_training_dataloader(data_path='../', batch_size=1)\"   # -> ../floorset_lite/"]),
    ("h2", "RL policy: checkpoints/policy.pt"),
    ("code", ["../venv/bin/python -m rl.train --iterations 4000 --episodes-per-iter 4 \\",
              "    --ppo-epochs 4 --checkpoint checkpoints/policy.pt --checkpoint-every 500 \\",
              "    --threads 3 --data-seed 12345"]),
    ("p", "From scratch: reward-prediction pretraining of the encoder, then PPO over the training "
          "designs with an adaptive canvas utilization (settles at 0.70). We use the iteration-4000 "
          "checkpoint (13.4 hours on 3 threads); longer training did not improve the pipeline."),
    ("h2", "Learned placer A: checkpoints/placer_a.pt (128 wide, 6 layers)"),
    ("code", ["../venv/bin/python scripts/train_placer.py --hours 7 --threads 2 --val-every 10000 \\",
              "    --out checkpoints/placer.pt"]),
    ("p", "placer_a.pt is the snapshot after about 290k training designs (validation centre "
          "error 0.067), taken while the run continued; to match it, stop near 290k designs."),
    ("h2", "Base for the relation networks: placer_d.pt (256 wide, 8 layers)"),
    ("code", ["../venv/bin/python scripts/train_placer.py --hours 6 --threads 4 --seed 4 --lr 5e-4 \\",
              "    --hidden 256 --layers 8 --norm --val-every 5000 --out checkpoints/placer_d.pt"]),
    ("h2", "Relation network 1: checkpoints/placer_r.pt (two stages)"),
    ("code", ["../venv/bin/python scripts/train_placer.py --hours 5 --threads 4 --seed 5 --lr 3e-4 \\",
              "    --hidden 256 --layers 8 --norm --rel --resume checkpoints/placer_d.pt \\",
              "    --val-every 5000 --out checkpoints/placer_rel.pt",
              "../venv/bin/python scripts/train_placer.py --hours 6 --threads 6 --seed 6 --lr 2e-4 \\",
              "    --hidden 256 --layers 8 --norm --rel --resume checkpoints/placer_rel.pt \\",
              "    --val-every 5000 --out checkpoints/placer_r.pt"]),
    ("p", "About 312k designs in total; validation centre error 0.050, relation accuracy 0.918."),
    ("h2", "Relation network 2: checkpoints/placer_relB.pt (independent of network 1)"),
    ("code", ["../venv/bin/python scripts/train_placer.py --hours 8 --threads 4 --seed 31 --lr 2e-4 \\",
              "    --hidden 256 --layers 8 --norm --rel --resume checkpoints/placer_d.pt \\",
              "    --val-every 5000 --out checkpoints/placer_relB.pt"]),
    ("p", "About 357k designs; validation centre error 0.050, relation accuracy 0.917."),
]


class Doc:
    def __init__(self, pdf):
        self.pdf, self.fig, self.y, self.page = pdf, None, TOP, 0

    def new_page(self):
        if self.fig is not None:
            self.fig.text(0.5, 0.03, f"Team ICB floorplanner -- page {self.page}", ha="center",
                          fontsize=7.5, color="0.5")
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(W, H))
        self.page += 1
        self.y = TOP

    def need(self, h):
        if self.fig is None or self.y - h < BOTTOM:
            self.new_page()

    def text(self, s, size=9.5, x=LEFT, dy=0.0185, width=98, **kw):
        lines = textwrap.wrap(s, width) or [""]
        self.need(dy * len(lines))
        for ln in lines:
            self.fig.text(x, self.y, ln, fontsize=size, va="top", **kw)
            self.y -= dy

    def table(self, header, rows, widths, size=8):
        # wrap each cell to its column width; rows grow with their tallest cell
        chars = [max(8, int(w * 108)) for w in widths]
        span = 1 - 2 * LEFT
        xs = [LEFT + span * sum(widths[:i]) for i in range(len(widths))]
        dy = 0.0165

        def draw(cells, bold=False, shade=None):
            wrapped = [textwrap.wrap(str(c), n) or [""] for c, n in zip(cells, chars)]
            h = dy * max(len(w) for w in wrapped) + 0.006
            self.need(h)
            if shade:
                self.fig.patches.append(plt.Rectangle((LEFT, self.y - h + 0.004), span, h, color=shade,
                                                      transform=self.fig.transFigure, zorder=0))
            for x, w in zip(xs, wrapped):
                for k, ln in enumerate(w):
                    self.fig.text(x + 0.005, self.y - k * dy, ln, fontsize=size, va="top",
                                  weight="bold" if bold else "normal")
            self.y -= h
            self.fig.add_artist(plt.Line2D([LEFT, LEFT + span], [self.y + 0.004] * 2, color="0.8", lw=0.5))

        draw(header, bold=True, shade="#dfe6ee")
        for i, r in enumerate(rows):
            draw(r, shade="#f6f8fa" if i % 2 else None)
        self.y -= 0.012

    def render(self, block):
        kind = block[0]
        if kind == "title":
            self.need(0.1)
            self.text(block[1], size=15, weight="bold", dy=0.03, width=62)
            self.text(block[2], size=10, color="0.35", dy=0.03)
            self.y -= 0.01
        elif kind == "page":
            self.new_page()
        elif kind == "h1":
            self.need(0.08)
            self.y -= 0.008
            self.text(block[1], size=13.5, weight="bold", dy=0.03)
        elif kind == "h2":
            self.need(0.06)
            self.text(block[1], size=10.5, weight="bold", dy=0.024)
        elif kind == "p":
            self.text(block[1])
            self.y -= 0.008
        elif kind == "code":
            dy = 0.0165
            h = dy * len(block[1]) + 0.012
            self.need(h)
            self.fig.patches.append(plt.Rectangle((LEFT, self.y - h + 0.006), 1 - 2 * LEFT, h,
                                                  color="#f1f1f1", transform=self.fig.transFigure, zorder=0))
            self.y -= 0.004
            for ln in block[1]:
                self.fig.text(LEFT + 0.012, self.y, ln, fontsize=8, family="monospace", va="top")
                self.y -= dy
            self.y -= 0.016
        elif kind == "table":
            self.table(block[1], block[2], block[3])

    def close(self):
        self.new_page()       # flushes the last page (and leaves an empty figure)
        plt.close(self.fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="floorplan_code.pdf")
    a = ap.parse_args()
    with PdfPages(Path(a.out)) as pdf:
        d = Doc(pdf)
        for b in DOC:
            d.render(b)
        d.close()
    print(f"wrote {a.out}: {d.page - 1} pages")


if __name__ == "__main__":
    main()

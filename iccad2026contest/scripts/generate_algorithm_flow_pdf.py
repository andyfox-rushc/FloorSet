"""Build algorithm_flow.pdf: Team ICB's solve-time algorithm flow, from the
contest entry point (MyOptimizer.solve) through the RL stage and every stage
after it. Reuses the page layout of generate_code_doc_pdf.py.

  python scripts/generate_algorithm_flow_pdf.py --out algorithm_flow.pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from generate_code_doc_pdf import LEFT, Doc  # noqa: E402

# Flow chart boxes: (title, detail, colour). The restart loop wraps all but the first and last.
FLOW = [
    ("MyOptimizer.solve()", "my_optimizer.py:180  -  contest entry point", "#dfe6ee"),
    ("from_solve_args", "rl/data.py:123  -  raw tensors -> FloorplanInstance", "#eef2f6"),
    ("1. RL layout: finetune_and_solve", "rl/finetune.py:112  -  ~2 s per-design PPO from checkpoints/policy.pt", "#e3f1e3"),
    ("2. Best-of candidates: best_of", "rl/analytic.py:349  -  10 candidates in parallel, ranked by relative cost", "#fdf1dc"),
    ("3. Compact packer: compact_pack x top 4", "rl/packer.py:446  -  sequence-pair annealing, 200 s x (n/77)^2 each", "#fbe3e3"),
    ("4. Winner selection + wall repack", "my_optimizer.py:258-284  -  origin / pin-box rules, repack the winner", "#efe3f6"),
    ("5. Finish: squeeze, relax_hpwl, reattach", "rl/packer.py:693 / 794 / 919", "#e3eef6"),
    ("return [(x, y, w, h)] x block_count", "cheapest of up to 3 runs", "#dfe6ee"),
]

DOC = [
    ("title", "Team ICB floorplanner: algorithm flow",
     "ICCAD 2026 CAD Contest, Problem C (FloorSet-Lite)  |  10/20/2026"),
    ("p", "This document follows one design through the solver, from the contest's call into "
          "MyOptimizer.solve() to the list of rectangles it returns. Section 1 covers the entry point "
          "and the restart loop around the whole pipeline; sections 2-7 detail each stage in the order "
          "it runs. Line numbers refer to the current tree; defaults are the configuration behind the "
          "10/18 official result (weighted score 1.0343, 76 of 100 designs beating the ground truth, "
          "641 s mean per design)."),
    ("flow",),

    ("h1", "0. What the solver optimizes"),
    ("p", "The contest scores each design with cost = (1 + 0.5 (HPWL_gap + Area_gap)) x exp(2 V_rel) x "
          "max(0.7, (runtime / median)^0.3), or 10 if any hard constraint is broken (overlap, soft-block "
          "area outside 1%, fixed or preplaced dimensions/position changed). The gaps are measured "
          "against the hidden ground truth and clamped at 0; V_rel is the share of soft constraints "
          "(boundary edge, cluster grouping, MIB shape sharing) that are violated."),
    ("p", "The ground truth is not available at solve time, so every stage ranks layouts with "
          "rl/repack.py contest_relative_cost (line 491): the contest evaluator run with the policy's "
          "own layout as the baseline. Ranking is unchanged by that choice of baseline, because the "
          "baseline divides every candidate's HPWL and area by the same numbers. Inner search loops "
          "use a fast numpy copy of the same formula (repack.py _Cost and packer.py _Batch.cost)."),

    ("h1", "1. Entry point: MyOptimizer.solve()"),
    ("p", "The evaluator (iccad2026_evaluate.py --evaluate my_optimizer.py) imports my_optimizer.py, "
          "finds the class that subclasses FloorplanOptimizer and calls solve() once per design with: "
          "block_count, area_targets [n], b2b_connectivity [E,3] (block i, block j, weight), "
          "p2b_connectivity [E,3] (pin, block, weight), pins_pos [P,2], constraints [n,5] (fixed, "
          "preplaced, mib_id, cluster_id, boundary bits 1 left / 2 right / 4 top / 8 bottom) and "
          "target_positions [n,4] (x, y, w, h, -1 where free). It must return n tuples (x, y, w, h)."),
    ("h2", "Setup"),
    ("p", "The constructor sets torch to one thread (rollout tensors are tiny; more threads oversubscribe "
          "the cores) and reads the stage switches from environment variables. solve() converts the raw "
          "tensors into a FloorplanInstance (rl/data.py:123) with missing inputs replaced by empty "
          "tensors. Its reference_metrics field (the reward scale) is filled in later by the RL stage."),
    ("h2", "Restart loop (my_optimizer.py:209)"),
    ("p", "The whole pipeline (_solve_once, sections 2-6) runs up to FLOORSET_SOLVE_RUNS = 3 times. "
          "Run r uses packer seeds offset by 1000 r, so each run searches differently. After each run "
          "its cost is computed relative to run 0's layout, and the loop stops early as soon as the "
          "cheapest layout so far has no soft violation (FLOORSET_RESTART_VIOL = 1). Runs 2 and 3 also "
          "switch on the packer's wall start (WALL_START = 100, section 4.4) for designs with right or "
          "top preplaced walls. The cheapest run is returned. Single designs vary by up to about 0.15 "
          "between runs, so this took the official score from 1.0404 to 1.0343."),
    ("table", ["Switch (environment variable)", "Default", "Effect"], [
        ["FLOORSET_SOLVE_RUNS", "3", "Whole-pipeline restarts"],
        ["FLOORSET_RESTART_VIOL", "1", "Stop restarting once there are no soft violations"],
        ["FLOORSET_CANDIDATES", "10 candidates", "Best-of list (section 3); empty = RL + polish only"],
        ["FLOORSET_PACK", "200", "Packer seconds per candidate at n = 77; 0 = no packer"],
        ["FLOORSET_PACK_TOPK", "4", "How many candidates are packed"],
        ["FLOORSET_PACK_JOBS", "0", "Packer jobs; 0 = one per packed candidate"],
        ["FLOORSET_SQUEEZE / FLOORSET_RELAX", "1 / 1", "Finishing steps (section 7)"],
        ["FLOORSET_PROCS", "min(6, cpus)", "Worker processes (forkserver pool)"],
        ["FLOORSET_CHECKPOINT", "checkpoints/policy.pt", "RL policy weights"],
    ], [0.36, 0.2, 0.44]),
    ("p", "Parallel work (candidates and packer jobs) goes through _pmap, which uses a forkserver pool "
          "of FLOORSET_PROCS workers and falls back to running in-process if the pool can't start."),

    ("page",),
    ("h1", "2. Stage 1 - RL layout (rl/finetune.py finetune_and_solve)"),
    ("p", "This is the AlphaChip-style part: a graph network reads the netlist and a policy places one "
          "block at a time on a grid. The policy is pretrained offline (rl/train.py, checkpoint at "
          "iteration 4000, trained up to canvas utilization 0.7) and fine-tuned with PPO on the design "
          "being solved for a 2 s budget (at most 150 iterations of 6 episodes, 3 PPO epochs). Its "
          "layout is not the final answer: it seeds the 'base' candidate and is the reference every "
          "later cost is measured against."),
    ("h2", "2.1 Placement order (rl/ordering.py compute_order)"),
    ("p", "The order is fixed per design and never depends on the policy. Preplaced blocks go first and "
          "take no action. MIB groups become one unit led by their first fixed-shape member (else the "
          "largest); followers copy the leader's shape. Units are then ordered by a greedy "
          "connectivity walk: a unit's score starts at 1000 x boundary priority (2 corner, 1 edge), "
          "and each placed unit adds its b2b weight to its neighbours' scores, plus 1e6 for "
          "clustermates, so clusters are placed together. The highest score goes next; with no "
          "connected unit left, the largest remaining unit."),
    ("h2", "2.2 Environment (rl/env.py GridPlacementEnv)"),
    ("p", "Canvas: area = total block area / utilization, aspect ratio = the pin bounding box's "
          "(fallback sqrt(area) x 1.4), stretched to cover preplaced blocks. The action grid is "
          "choose_grid_dim(n) = clamp(8 sqrt(n), 32, 96) cells per side; occupancy is tracked on a grid "
          "4x finer. Utilization comes from the checkpoint's ladder: 0.70, then 0.65, then 0.60, moving "
          "looser only when a whole batch fails before anything has finished. (A ratchet that tightens "
          "the canvas after successes exists but is off: FLOORSET_RATCHET = 0.)"),
    ("p", "Each free block takes two actions: an aspect ratio from [0.2, 0.3, 0.45, 0.67, 1, 1.5, "
          "2.22, 3.33, 5] (w = sqrt(A r), h = sqrt(A / r), so area is exact) and a grid cell. Fixed, "
          "preplaced and MIB-follower shapes are copied, never chosen. Masks make every allowed action "
          "safe:"),
    ("table", ["Mask / rule", "What it does"], [
        ["Overlap", "A cell is legal only if the footprint (rounded up) is free on the fine grid (summed-area table)."],
        ["Aspect", "Only ratios whose footprint fits somewhere are offered."],
        ["Boundary", "Edge-pinned blocks may only use cells flush with their edge(s); falls back to any free cell if none."],
        ["Cluster touch", "If a clustermate is placed, every exact edge-touching spot is searched first and the block is auto-placed there (no action)."],
        ["Preference", "Otherwise cells touching the cluster, then cells touching any placed block, then any free cell."],
        ["Lookahead", "After a trial placement the 3 largest unplaced blocks must still fit; else the cell is masked and redrawn (8 tries)."],
        ["Shape fit", "After placement a free, unpinned block may take a same-area shape (within x/÷1.4) that ends exactly on a neighbour's edge."],
    ], [0.2, 0.8]),
    ("p", "A rollout fails only if some block has no legal shape or cell left; the env then raises and "
          "the episode is dropped. Boundary, grouping and MIB are best-effort here; overlap, area and "
          "fixed/preplaced dimensions are guaranteed by construction."),
    ("h2", "2.3 Networks (rl/encoder.py, rl/networks.py ActorCritic)"),
    ("p", "Encoder: 9 block features (log area, fixed, preplaced, MIB, cluster, 4 boundary bits) and 2 "
          "pin features (position / sqrt(total area)), each projected to 64; 3 residual message-passing "
          "layers sum weight-scaled messages over b2b and p2b edges. It runs once per episode; the "
          "global embedding is the mean block embedding. Per step the context vector is [block "
          "embedding, global embedding, progress, 6 step features] (135 values): remaining area / free "
          "area, largest future block's size, share of its shapes that still fit, steps left."),
    ("p", "Aspect head: MLP 135 -> 64 -> 9. Position head (PositionCNN): three spatial maps plus the "
          "context broadcast as 16 channels, then conv 3x3 -> 32, 3x3 -> 32, 1x1 -> 1 logit per cell. "
          "The maps are occupancy (fraction of each cell filled), the block's cluster footprint, and "
          "the wiremask: each cell's weighted Manhattan distance from the block centre to its placed "
          "neighbours and pins, the exact contest HPWL formula, normalized to [0, 1]. Value and "
          "reward-approximation heads are small MLPs on the global embedding, progress, reward so far "
          "and step features. Invalid actions get logit -1e9."),
    ("h2", "2.4 Reward and PPO (rl/reward.py, rl/ppo.py)"),
    ("p", "The terminal reward is minus the contest cost of the compacted layout, normalized against a "
          "deterministic greedy layout (square shapes, first free cell) instead of the ground truth: "
          "-(1 + 0.5 (HPWL / HPWL_greedy + area / area_greedy)) x exp(2 V_rel). An infeasible layout "
          "scores -(10 + hard violations) e^2; an unfinished one -10 e^2. Because HPWL is a sum over "
          "edges that are fully known once both ends are placed, each position step is credited with "
          "its own exact HPWL and bounding-box increase (x the violation factor), and the remainder "
          "goes on the last step. Returns are undiscounted."),
    ("p", "PPO: advantage = return - value, normalized; clipped surrogate (epsilon 0.2); loss = policy + "
          "0.5 value MSE + 0.5 reward-approx MSE - 0.01 entropy; Adam lr 1e-3, grad clip 1.0. Sampling "
          "temperature anneals from 1.0 to 0.3 over the iteration budget, and updates are skipped when "
          "no further batch would fit in the budget. After the loop one greedy (argmax) rollout at the "
          "tightest working utilization is added; the best feasible layout by reward wins, with the "
          "greedy layout as a fallback that always succeeds."),
    ("h2", "2.5 Compaction (rl/compaction.py compact)"),
    ("p", "The chosen layout is slid toward the lower-left: three passes of x then y, each shifting "
          "whole clusters rigidly, sweeping every movable block down to its nearest obstacle, and "
          "pulling right/top-pinned groups inward. Preplaced blocks never move, and pinned blocks keep "
          "their pinned axis. Finally right/top-pinned blocks are pushed back out to the far edge where "
          "the path is clear. Compaction is applied to the answer and before scoring the reward."),

    ("page",),
    ("h1", "3. Stage 2 - Best-of candidates (rl/analytic.py best_of)"),
    ("p", "Ten candidate layouts are built in parallel (_candidate_job, line 340), each scored with "
          "contest_relative_cost against the RL layout. A name is kind[:itN][:poX]: itN optimizer "
          "iterations (default 500), poX seconds of polish (default 1.0)."),
    ("table", ["Candidate", "How it is built", "Polished"], [
        ["base", "The RL layout from stage 1", "Yes, 1 s"],
        ["scratch0.75 ... scratch1.0 (5)", "Analytic global placement from scratch at utilization 0.75, 0.85, 0.90, 0.95, 1.0; 600 iterations", "Yes, 1 s"],
        ["learned", "Learned placer placer_a.pt; order from predicted centres", "No"],
        ["learned2", "placer_r.pt; order from its relation head", "No"],
        ["learned3", "placer_r.pt; order from predicted centres", "No"],
        ["learned4", "placer_relB.pt (independently trained); relation head", "No"],
    ], [0.26, 0.6, 0.14]),
    ("h2", "3.1 Analytic placement (scratch<util>: global_place, line 58)"),
    ("p", "Soft blocks start square, fixed/preplaced keep their size and MIB members copy their "
          "leader. The outline has area total / util and the pin box's aspect ratio. Each block starts "
          "at the weighted mean of its pins. Adam then moves the centres to minimize smoothed HPWL "
          "(normalized by its starting value) + an overlap penalty whose weight ramps geometrically "
          "from 0.02 to 200 + 10 x out-of-outline + an edge pull for pinned blocks + a weak "
          "cluster attraction. Preplaced blocks are held fixed. The result is legalized "
          "(slicing._legalize_grow: repack._legalize up to 4 times, lowering the floor until no "
          "preplaced block is displaced)."),
    ("h2", "3.2 Learned placers (learned_candidate, line 255; rl/learned_placer.py)"),
    ("p", "PlacerNet is a graph network (15 block features: the base ones plus size relative to the "
          "pin box, preplaced centres, known w/h). It predicts every block's centre in pin-box "
          "coordinates and a log aspect ratio; the relation networks also predict, for block pairs, "
          "which is left of / below the other. These were trained by supervision on ground-truth "
          "layouts from the training set (L1 on centres and aspect, cross-entropy on relations)."),
    ("p", "To build a layout, soft blocks take the predicted aspect at exact area, a sequence pair "
          "(two block orderings that fix every pair's left/right or above/below relation) is read "
          "from the relations or from the centres (cx - cy and cx + cy), and the pair is packed "
          "from the origin with the packer's evaluator. If a preplaced block collides, shapes are "
          "refitted (600 steps) or blocks moved around it. Finally sizing.fit_shapes (1000 steps) "
          "tunes each soft block's aspect by gradient descent through a differentiable packing of the "
          "fixed sequence pair, keeping the best exactly-scored snapshot only if it is cheaper."),
    ("h2", "3.3 Per-candidate polish (base and scratch only)"),
    ("p", "Half the polish budget goes to slicing_polish (rl/slicing.py:368): it rebuilds the layout "
          "as a recursive guillotine partition of a root rectangle (several root shapes and fill "
          "targets, cuts that never cross preplaced blocks or split clusters), gives each soft block "
          "its leaf's shape, legalizes and keeps the cheapest. The other half goes to repack_polish "
          "(rl/repack.py:443): simulated annealing over reshape, swap, move-beside, nudge and "
          "cluster-shift moves, each followed by a longest-path legalization, then boundary snapping "
          "and legalize.repair (moves blocks off walls and back to their clusters). Each polish keeps "
          "its result only if the real evaluator says it is feasible and cheaper."),
    ("h2", "3.4 Ranking"),
    ("p", "Infeasible candidates are dropped and the rest sorted by relative cost. The learned "
          "candidates are moved up to just behind the cheapest one, so the top-4 cut always packs "
          "them: they rank low before packing but often win after it. If nothing is feasible, the RL "
          "layout is used."),

    ("page",),
    ("h1", "4. Stage 3 - Compact packer (rl/packer.py compact_pack)"),
    ("p", "Most of the final quality comes from this stage. Each of the top 4 candidates becomes one "
          "packer job (seed = run offset + job index), all running in parallel. Each gets "
          "pack_budget(n) = max(0.25, 200 x (n / 77)^2) seconds, 77 being the RMS block count of the "
          "validation set (about 200 s at n = 77, 4 s at n = 11, 340 s at n = 100)."),
    ("h2", "4.1 Representation"),
    ("p", "A layout is a sequence pair (G+, G-): a before b in both means a is left of b; a after b "
          "in G+ but before it in G- means a is below b. The start pair is read off the candidate "
          "layout (topological orders of 'left/below' and 'left/above' relations). If the start sits "
          "below or left of 0, colliding blocks are first moved around the preplaced blocks."),
    ("h2", "4.2 Packing and cost"),
    ("p", "Batches of 32 proposals are packed at once: blocks in G- order go bottom-left against the "
          "blocks before them; preplaced blocks are forced to their spot and any push is recorded as "
          "overshoot; right/top-pinned blocks then slide out to the bounding-box edge. The cost is the "
          "contest formula relative to the start layout, with violations counted as off-edge pinned "
          "blocks, MIB shape mismatches and split cluster pieces. Overshoot multiplies it by "
          "10 + overshoot. Search-only factors are added for running past the pin box (weight 3), "
          "for the distance between split cluster pieces (weight 1), and for running past preplaced "
          "walls when a wall penalty is set."),
    ("h2", "4.3 Moves and schedule"),
    ("table", ["Share", "Move"], [
        ["25%", "Reshape a soft block (or MIB group): width x exp(N(0, 0.3)), same area, aspect within 1/16..16"],
        ["35%", "Swap two blocks in G+, G- or both; 15% of the time swap a block with the one nearest its net centroid"],
        ["40%", "Move a block next to another on a random side; half the time a net or cluster partner"],
    ], [0.12, 0.88]),
    ("p", "Each step proposes 32 moves from the current state and takes the cheapest. It is accepted if "
          "downhill, otherwise with probability exp(-delta / T), T decaying geometrically to 0.001 of "
          "its start over the budget. The best feasible layout and the best one inside the pin box are "
          "tracked. With 2 s or more, the search is split in two halves with a gradient shape refit "
          "(sizing.fit_shapes, 300 steps) between them."),
    ("p", "At the end the in-box best is preferred if within 2.5% of the overall best, and the "
          "result is rechecked with the real evaluator; it is returned only if feasible and cheaper "
          "than the start (or newly inside the box). Then reattach_clusters (section 6) runs on it."),
    ("h2", "4.4 Walls"),
    ("p", "Preplaced blocks pinned to the right or top edge fix where that edge of the layout must "
          "be: a 'wall'. A layout that grows past one leaves the pinned block off its edge. On runs 2 "
          "and 3 of the restart loop, for designs with such walls, the idle worker slots (2 of 6 with "
          "4 packer jobs) run extra packer jobs with a wall penalty of 100 from the start."),

    ("h1", "5. Stage 4 - Winner selection and wall repack (my_optimizer.py:258-284)"),
    ("p", "Packer outputs that extend below x = 0 or y = 0 are dropped if any output doesn't. Among "
          "the rest, the cheapest layout inside the pin box wins if it is within 2.5% of the cheapest "
          "overall; otherwise the cheapest wins. The winner is then repacked again (REPACK_WINNER = 1): "
          "4 more packer jobs start from it with wall penalty 100, and the cheapest is kept if it beats "
          "the winner. (A pin-box escalation, FLOORSET_PINBOX_ESCALATE, exists but is off: it fixed "
          "no designs and tripled runtime.)"),

    ("h1", "6. Stage 5 - Finish"),
    ("p", "squeeze (packer.py:693): the sequence pair is fixed and only soft shapes change. Each round "
          "computes every block's slack (how far it could move without growing the outline); a block "
          "with no slack on one axis but some on the other is widened or heightened into that slack "
          "at constant area (100%, 60%, 30% of it), one block or all critical blocks at a time. The "
          "best change is kept; up to 200 rounds, stopping when nothing helps."),
    ("p", "relax_hpwl (packer.py:794): up to 30 sweeps keep the relations and the outline fixed and "
          "slide each block along one axis toward the weighted median of its connected blocks and "
          "pins, within the gap between its neighbours. Preplaced blocks and pinned axes stay put; "
          "cluster members may move only if their cluster doesn't split further."),
    ("p", "reattach_clusters (packer.py:919): for each cluster still in several pieces, members try "
          "spots sliding along every side of the other pieces' blocks (9 offsets), soft blocks also "
          "with 13 aspect ratios and edge-matching stretches; spots that overlap or go below 0 are "
          "dropped and the cheapest is kept. Up to 4 rounds. Each finishing step is kept only if the "
          "real evaluator says the result is feasible and cheaper."),

    ("h1", "7. Output"),
    ("p", "_solve_once returns the finished layout; solve() keeps the cheapest over the restart runs "
          "and returns it as block_count tuples (x, y, w, h). Overlap, area and fixed/preplaced "
          "dimensions hold by construction in stage 1 and are rechecked with the real evaluator at "
          "every 'keep if feasible and cheaper' step (100 of 100 validation designs feasible in the 10/18 run)."),
    ("h2", "Where the time goes (typical 77-block design, one run)"),
    ("table", ["Stage", "Wall time", "Notes"], [
        ["1. RL fine-tuning", "~2 s", "Fixed budget"],
        ["2. Candidates", "a few s", "10 jobs over 6 workers; 1 s polish each, learned ones faster"],
        ["3. Packer", "~200 s", "4 parallel jobs, budget x (n/77)^2"],
        ["4. Wall repack", "~200 s", "4 more parallel jobs from the winner"],
        ["5. Finish", "seconds", "Exact packing per change"],
        ["Restart loop", "x 1-3", "Stops early once there are no soft violations"],
    ], [0.24, 0.16, 0.6]),
]


class FlowDoc(Doc):
    def render(self, block):
        if block[0] != "flow":
            return super().render(block)
        bh, gap = 0.048, 0.018
        h = len(FLOW) * (bh + gap) + 0.02
        self.need(h)
        x0, bw = LEFT + 0.1, 0.62
        tops = []
        for i, (title, detail, colour) in enumerate(FLOW):
            top = self.y - i * (bh + gap)
            tops.append(top)
            self.fig.patches.append(FancyBboxPatch((x0, top - bh), bw, bh, boxstyle="round,pad=0.004",
                                                   fc=colour, ec="0.45", lw=0.7,
                                                   transform=self.fig.transFigure))
            self.fig.text(x0 + bw / 2, top - 0.016, title, ha="center", va="center", fontsize=9.5,
                          weight="bold")
            self.fig.text(x0 + bw / 2, top - 0.034, detail, ha="center", va="center", fontsize=7.5,
                          color="0.3")
            if i:
                self.fig.add_artist(matplotlib.patches.FancyArrowPatch(
                    (x0 + bw / 2, top + gap), (x0 + bw / 2, top + 0.003), arrowstyle="-|>",
                    mutation_scale=9, color="0.35", transform=self.fig.transFigure))
        # restart loop bracket around stages 1-5 (boxes 2..6)
        xl = x0 + bw + 0.025
        ya, yb = tops[2] - bh / 2, tops[6] - bh / 2
        red = "#b04040"
        self.fig.add_artist(plt.Line2D([x0 + bw + 0.006, xl, xl], [yb, yb, ya], color=red, lw=1.0))
        self.fig.add_artist(matplotlib.patches.FancyArrowPatch(
            (xl, ya), (x0 + bw + 0.006, ya), arrowstyle="-|>", mutation_scale=10, color=red, lw=1.0,
            transform=self.fig.transFigure))
        self.fig.text(xl + 0.012, (ya + yb) / 2, "restart loop\nup to 3 runs,\nnew packer seeds;\n"
                      "stop when no\nsoft violation", fontsize=7.5, color="#b04040", va="center")
        self.y -= h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="algorithm_flow.pdf")
    a = ap.parse_args()
    with PdfPages(Path(a.out)) as pdf:
        d = FlowDoc(pdf)
        for b in DOC:
            d.render(b)
        d.close()
    print(f"wrote {a.out}: {d.page - 1} pages")


if __name__ == "__main__":
    main()

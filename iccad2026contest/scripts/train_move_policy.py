"""Train the packer's move scorer (rl/move_policy.py) on logged packer moves.

Each log (scripts/collect_moves.py) holds, per packer step, the features of
the BATCH moves packed and the cost change each caused, divided by the
search's start temperature. The network regresses that change (Huber, clipped
at +-CLIP); held-out designs measure how often the step's best move survives
when only the top quarter by score would be packed (random: 25%).

  python scripts/train_move_policy.py --logs movelogs/r0 --out checkpoints/move_policy.npz
"""
import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import torch

CLIP = 5.0


def load_logs(dirs):
    F, D, S, G = [], [], [], []
    step0 = 0
    for k, path in enumerate(sorted(p for d in dirs for p in glob.glob(os.path.join(d, "*.npz")))):
        z = np.load(path)
        F.append(z["f"]); D.append(z["d"]); S.append(z["step"] + step0); G.append(np.full(len(z["d"]), k))
        step0 += int(z["step"].max()) + 1 if len(z["step"]) else 0
    return np.concatenate(F), np.concatenate(D), np.concatenate(S), np.concatenate(G)


def top_quarter_recall(score, d, step):
    """Share of steps whose best (lowest d) move is in the top quarter by score."""
    hit = tot = 0
    order = np.argsort(step, kind="stable")
    s, dd, st = score[order], d[order], step[order]
    cuts = np.flatnonzero(np.diff(st)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(st)]):
        if b - a < 4:
            continue
        k = max((b - a) // 4, 1)
        hit += int(np.argmin(dd[a:b]) in np.argsort(s[a:b])[:k])
        tot += 1
    return hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    torch.manual_seed(0)
    F, D, S, G = load_logs(a.logs)
    D = np.clip(D, -CLIP, CLIP)
    val = G % 10 == 0           # every 10th design held out
    print(f"{len(D)} moves from {G.max() + 1} designs; improving share {np.mean(D < 0):.3f}", flush=True)
    mu, sd = F[~val].mean(0), F[~val].std(0) + 1e-6
    X = torch.tensor((F - mu) / sd, dtype=torch.float32)
    Y = torch.tensor(D, dtype=torch.float32)
    H = a.hidden
    net = torch.nn.Sequential(torch.nn.Linear(F.shape[1], H), torch.nn.ReLU(), torch.nn.Linear(H, H),
                              torch.nn.ReLU(), torch.nn.Linear(H, 1))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    tr = np.flatnonzero(~val)
    best = -1.0
    for ep in range(a.epochs):
        np.random.shuffle(tr)
        for i in range(0, len(tr), 4096):
            idx = torch.tensor(tr[i:i + 4096])
            loss = torch.nn.functional.huber_loss(net(X[idx])[:, 0], Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            sv = net(X[torch.tensor(np.flatnonzero(val))])[:, 0].numpy()
        rec = top_quarter_recall(sv, D[val], S[val])
        tag = ""
        if rec > best:
            best = rec
            L = [m for m in net if isinstance(m, torch.nn.Linear)]
            np.savez(a.out, mu=mu, sd=sd, **{f"W{i}": L[i].weight.detach().numpy().T for i in range(3)},
                     **{f"b{i}": L[i].bias.detach().numpy() for i in range(3)})
            tag = " (saved)"
        print(f"epoch {ep}: train loss {float(loss):.4f}, held-out top-quarter recall of the best move {rec:.3f}{tag}",
              flush=True)


if __name__ == "__main__":
    main()

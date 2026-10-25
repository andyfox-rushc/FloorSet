"""Migrate a checkpoint to the step-feature network (rl/networks.py
NUM_STEP_FEATS): each head's first layer gets NUM_STEP_FEATS zero input
columns appended, so the migrated network behaves exactly like the old one
until training learns to use the new inputs. A <ckpt>.meta.json is copied.

  python scripts/migrate_checkpoint_step_features.py IN.pt OUT.pt
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch

from rl.networks import NUM_STEP_FEATS, ActorCritic

KEYS = ["aspect_head.mlp.0.weight", "position_cnn.embed_proj.weight",
        "value_head.mlp.0.weight", "reward_approx_head.mlp.0.weight"]


def migrate(src: str, dst: str) -> None:
    sd = torch.load(src, map_location="cpu")
    target = ActorCritic().state_dict()
    for k in KEYS:
        w = sd[k]
        want = target[k].shape[1]
        if w.shape[1] == want:
            continue
        assert w.shape[1] + NUM_STEP_FEATS == want, (k, w.shape, want)
        sd[k] = torch.cat([w, torch.zeros(w.shape[0], NUM_STEP_FEATS)], dim=1)
    ActorCritic().load_state_dict(sd)    # validates every shape
    torch.save(sd, dst)
    meta = Path(src + ".meta.json")
    if meta.exists():
        shutil.copy(meta, dst + ".meta.json")
    print(f"migrated {src} -> {dst}")


if __name__ == "__main__":
    migrate(sys.argv[1], sys.argv[2])

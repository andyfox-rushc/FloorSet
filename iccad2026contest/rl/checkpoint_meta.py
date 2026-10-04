"""Sidecar JSON next to a checkpoint (<path>.meta.json) recording training
state that inference must match -- currently the canvas utilization the
adaptive curriculum in rl/train.py had reached."""

import json
import os
from typing import List

from .env import UTILIZATION_TARGET


def meta_path(checkpoint_path: str) -> str:
    return checkpoint_path + ".meta.json"


def read_checkpoint_meta(checkpoint_path: str) -> dict:
    try:
        with open(meta_path(checkpoint_path)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_checkpoint_meta(checkpoint_path: str, meta: dict) -> None:
    tmp = meta_path(checkpoint_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, meta_path(checkpoint_path))


def utilization_ladder(checkpoint_path: str, step: float = 0.05) -> List[float]:
    """Utilizations for finetune_and_solve, tightest first: the trained
    level, then looser in `step`s down to UTILIZATION_TARGET."""
    top = read_checkpoint_meta(checkpoint_path).get("utilization", UTILIZATION_TARGET)
    ladder = [round(top, 4)]
    while ladder[-1] - step > UTILIZATION_TARGET + 1e-9:
        ladder.append(round(ladder[-1] - step, 4))
    if ladder[-1] > UTILIZATION_TARGET + 1e-9:
        ladder.append(UTILIZATION_TARGET)
    return ladder

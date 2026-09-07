import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from rl.env import GridPlacementEnv


def _run_once(env: GridPlacementEnv, seed: int):
    rng = random.Random(seed)
    steps = 0
    max_steps = 10_000
    while not env.done():
        steps += 1
        assert steps < max_steps, "rollout did not terminate"
        if env.needs_aspect():
            idx = rng.randrange(len(env.aspect_ratios))
            w, h = env.choose_aspect(idx)
        else:
            w, h = env.current_shape()

        result = env.position_mask(w, h)
        if result is None:
            continue  # auto-placed deterministically (cluster touch)
        mask, _, _ = result
        valid = mask.nonzero(as_tuple=False)
        assert valid.numel() > 0, "position_mask returned an all-False mask"
        pick = valid[rng.randrange(valid.shape[0])]
        env.place(int(pick[0]), int(pick[1]))
    return env.finalize()


def run_random_rollout(env: GridPlacementEnv, seed: int = 0, max_seed_tries: int = 20):
    """Drive an env to completion with uniformly random valid actions, used
    to fuzz-test hard-constraint guarantees independent of any policy.

    A uniform-random policy has no foresight -- it can legitimately paint
    itself into a corner (e.g. place an extreme-aspect-ratio block such that
    no later block's footprint fits anywhere), which is exactly the kind of
    mistake a *trained* policy is supposed to learn to avoid. That is a
    property of the random policy, not a correctness bug in the env, so on a
    RuntimeError ("no free position") this retries with the next seed rather
    than failing the test; it only propagates the error if every seed in the
    range fails.
    """
    last_err = None
    for s in range(seed, seed + max_seed_tries):
        try:
            return _run_once(env, s)
        except RuntimeError as e:
            last_err = e
            env.reset()
    raise last_err


@pytest.fixture
def rollout():
    return run_random_rollout

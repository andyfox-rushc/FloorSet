from rl.checkpoint_meta import read_checkpoint_meta, utilization_ladder, write_checkpoint_meta
from rl.env import UTILIZATION_TARGET


def test_ladder_defaults_to_env_target_without_meta(tmp_path):
    assert utilization_ladder(str(tmp_path / "missing.pt")) == [UTILIZATION_TARGET]


def test_ladder_steps_down_from_trained_level(tmp_path):
    ckpt = str(tmp_path / "policy.pt")
    write_checkpoint_meta(ckpt, {"utilization": 0.725, "iteration": 10})
    assert read_checkpoint_meta(ckpt)["iteration"] == 10
    ladder = utilization_ladder(ckpt)
    assert ladder[0] == 0.725 and ladder[-1] == UTILIZATION_TARGET
    assert all(a > b for a, b in zip(ladder, ladder[1:]))

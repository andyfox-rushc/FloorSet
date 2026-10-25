"""Summarize solver_ab outputs (10-07): mean cost, gaps, and how many layouts
sit inside [0, pin box].  python scripts/box_report.py a.json b.json ..."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import polish_bench as PB
from rl.data import from_validation_sample
from rl.packer import in_box, pin_box
PB._init()
for f in sys.argv[1:]:
    d = {int(k): v for k, v in json.load(open(f)).items()}
    inb = sum(in_box(v['positions'], pin_box(from_validation_sample(PB.EV, k))) for k, v in d.items())
    m = lambda k: sum(v[k] for v in d.values()) / len(d)
    print(f"{f}: cost {m('cost'):.4f} hpwl {m('hpwl_gap'):.3f} area {m('area_gap'):.3f} vrel {m('vrel'):.3f} "
          f"in box {inb}/{len(d)} secs {m('secs'):.0f}")

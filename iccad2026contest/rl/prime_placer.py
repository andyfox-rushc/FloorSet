"""
Learned centroid predictor for FloorSet-Prime (10-20): the Lite learned
placer's network (rl/learned_placer.py PlacerNet) trained on Prime ground
truth to predict every block's polygon centroid as a fraction of the frame.

The frame is known exactly at solve time (pin box == ground-truth frame on
all 100 test designs), so the prediction maps straight to frame coordinates
and seeds rl/partition.py: the first slicing tree and the rigid blocks'
corners. Inputs are the Lite features of the rectangle version of the design
(rl/prime_adapter.py to_lite: preplaced polygons as pinned pieces).
"""

from typing import Optional, Sequence

import numpy as np
import torch
from shapely.geometry import Polygon

from . import polygon as G
from .data import from_solve_args
from .learned_placer import PlacerNet, features
from .prime_adapter import to_lite


def lite_instance(block_count, area_targets, b2b, p2b, pins_pos, constraints, target_polygons):
    prob = to_lite(block_count, area_targets, b2b, p2b, pins_pos, constraints, target_polygons)
    return from_solve_args(prob.block_count, prob.area_targets, prob.b2b, prob.p2b, prob.pins_pos,
                           prob.constraints, prob.target_positions)


def frame_of(pins_pos):
    p = pins_pos[(pins_pos[:, 0] >= 0) & (pins_pos[:, 1] >= 0)]
    return float(p[:, 0].max()), float(p[:, 1].max())


def targets(polygons, W, H) -> torch.Tensor:
    """Ground-truth centroids [n, 2] as fractions of the frame."""
    c = [Polygon(G.clean_vertices(v)).centroid.coords[0] for v in polygons]
    return torch.tensor([(x / W, y / H) for x, y in c], dtype=torch.float32)


def loss_fn(net, inst, n, tgt, movable):
    feats, pin_feats, b2b, p2b, _ = features(inst)
    pred, _ = net(feats, pin_feats, b2b, p2b)
    err = (pred[:n] - tgt).abs().sum(dim=1)
    return err[movable].mean() if movable.any() else err.mean()


def load(path: str) -> Optional[PlacerNet]:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except OSError:
        return None
    net = PlacerNet(**ck["cfg"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net


def predict(net, block_count, area_targets, b2b, p2b, pins_pos, constraints, target_polygons) -> np.ndarray:
    """Predicted centroids [n, 2] in frame coordinates."""
    inst = lite_instance(block_count, area_targets, b2b, p2b, pins_pos, constraints, target_polygons)
    W, H = frame_of(pins_pos)
    with torch.no_grad():
        pred, _ = net(*features(inst)[:4])
    return pred[:block_count].double().numpy() * np.array([W, H])

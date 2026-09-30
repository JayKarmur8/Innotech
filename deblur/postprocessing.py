"""Post-processing: turn network outputs back into a full-size BGR image.

* NCHW float -> HWC uint8 BGR (clipped, rounded)
* seamless blending of overlapping tiles with a smooth weight window
* remove padding and resize back to the ORIGINAL input dimensions
"""

import cv2
import numpy as np


def from_tensor(tensor):
    """(1, 3, H, W) or (3, H, W) RGB float [0,1] -> (H, W, 3) float32 BGR [0,1]."""
    arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    arr = arr.transpose(1, 2, 0)[:, :, ::-1]
    return arr


def to_uint8(img_float):
    img = np.nan_to_num(img_float)  # NaN -> 0 (guards against FP16 overflow)
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def blend_window(tile, overlap):
    """2-D weight window: 1 in the centre, linear ramps over ``overlap`` px."""
    if overlap <= 0:
        return np.ones((tile, tile), np.float32)
    ramp = np.ones(tile, np.float32)
    edge = (np.arange(overlap, dtype=np.float32) + 1.0) / (overlap + 1.0)
    ramp[:overlap] = edge
    ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp).astype(np.float32)


class TileAccumulator(object):
    """Accumulates weighted tile outputs into a full canvas."""

    def __init__(self, prep):
        ph, pw = prep.padded_shape
        self.prep = prep
        self.canvas = np.zeros((ph, pw, 3), np.float32)
        self.weights = np.zeros((ph, pw, 1), np.float32)
        self.window = blend_window(prep.tile_size, prep.overlap)[:, :, None]

    def add(self, pos, tile_out_hwc):
        y, x = pos
        t = self.prep.tile_size
        self.canvas[y:y + t, x:x + t] += tile_out_hwc * self.window
        self.weights[y:y + t, x:x + t] += self.window

    def result(self):
        img = self.canvas / np.maximum(self.weights, 1e-8)
        h, w = self.prep.work_shape
        return img[:h, :w]


def finalize(prep, img_float):
    """Convert to uint8 and restore the original image dimensions."""
    out = to_uint8(img_float)
    oh, ow = prep.original_shape
    if out.shape[:2] != (oh, ow):
        out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_CUBIC)
    return out

"""Preprocessing: prepare an OpenCV BGR image for the deblurring network.

Steps
    1. optional downscale of very large images (``max_input_side``)
    2. reflect-pad so the image is at least one tile large
    3. split into overlapping fixed-size tiles (fixed shapes = fastest TensorRT)
    4. per tile: BGR->RGB, uint8 -> float [0,1], HWC -> NCHW
"""

import cv2
import numpy as np

from .image_io import resize_max_side


class PreparedImage(object):
    """Holds everything post-processing needs to rebuild the full image."""

    def __init__(self, image, original_shape, padded_shape, tiles, tile_size, overlap):
        self.image = image                    # (possibly downscaled) BGR uint8
        self.original_shape = original_shape  # (H, W) of the user's image
        self.padded_shape = padded_shape      # (H, W) after padding
        self.tiles = tiles                    # list of (y, x) top-left corners
        self.tile_size = tile_size
        self.overlap = overlap

    @property
    def work_shape(self):
        return self.image.shape[:2]


def tile_positions(length, tile, overlap):
    """Start offsets covering [0, length) with tiles of size ``tile``."""
    if length <= tile:
        return [0]
    stride = tile - overlap
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] + tile < length:
        pos.append(length - tile)   # last tile flush with the border
    return pos


def prepare_image(img, tile_size, overlap, max_input_side=0):
    original_shape = img.shape[:2]
    work = resize_max_side(img, max_input_side)
    h, w = work.shape[:2]
    pad_h, pad_w = max(0, tile_size - h), max(0, tile_size - w)
    padded = work
    if pad_h or pad_w:
        mode = cv2.BORDER_REFLECT_101 if (h > 1 and w > 1) else cv2.BORDER_REPLICATE
        padded = cv2.copyMakeBorder(work, 0, pad_h, 0, pad_w, mode)
    ph, pw = padded.shape[:2]
    tiles = [(y, x) for y in tile_positions(ph, tile_size, overlap)
             for x in tile_positions(pw, tile_size, overlap)]
    prep = PreparedImage(work, original_shape, (ph, pw), tiles, tile_size, overlap)
    prep.padded = padded
    return prep


def to_tensor(img_bgr, dtype=np.float32):
    """BGR uint8 (H, W, 3) -> RGB float (1, 3, H, W) in [0, 1]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    return np.ascontiguousarray(chw[None].astype(dtype, copy=False))


def iter_tile_tensors(prep, dtype=np.float32):
    t = prep.tile_size
    for (y, x) in prep.tiles:
        yield (y, x), to_tensor(prep.padded[y:y + t, x:x + t], dtype)

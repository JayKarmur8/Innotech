"""Inference engine: glues preprocessing -> model -> post-processing together.

    engine = DeblurEngine(cfg)
    result = engine.deblur(bgr_image)       # full resolution, tiled
    result.image, result.inference_ms, result.total_ms
"""

import threading
import time

import numpy as np

from . import postprocessing as post
from . import preprocessing as pre
from .errors import DeblurError, InsufficientResourcesError
from .model_loader import is_oom_error, load_backend


class DeblurResult(object):
    def __init__(self, image, inference_ms, total_ms, tiles, work_shape):
        self.image = image                # restored BGR uint8, same size as input
        self.inference_ms = inference_ms  # time spent inside the network
        self.total_ms = total_ms          # pre + inference + post
        self.tiles = tiles                # number of network calls
        self.work_shape = work_shape      # resolution the network actually saw

    @property
    def fps(self):
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0


class DeblurEngine(object):
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self.backend = load_backend(cfg, log=log)
        self.lock = threading.Lock()   # one inference at a time (GPU memory)

        n, c, h, w = (tuple(self.backend.input_shape) + (None,) * 4)[:4]
        cfg_tile = int(cfg["processing"]["tile_size"])
        if h and w:
            if h != w:
                raise DeblurError("Model input must be square, got %sx%s" % (h, w))
            if h != cfg_tile:
                log("Note: using the model's tile size %d (config says %d)" % (h, cfg_tile))
            self.tile_size = int(h)
        else:
            self.tile_size = cfg_tile
        self.overlap = min(int(cfg["processing"]["tile_overlap"]), self.tile_size // 2 - 1)
        self.max_input_side = int(cfg["processing"]["max_input_side"])
        self.model_name = cfg["model"].get("name", "model")
        log("Loaded %s on %s (tile %d, overlap %d)"
            % (self.model_name, self.backend.description, self.tile_size, self.overlap))

    @property
    def description(self):
        return self.backend.description

    def warmup(self, runs=2):
        """First TensorRT/CUDA calls are slow (lazy init) - run them upfront."""
        dummy = np.zeros((1, 3, self.tile_size, self.tile_size), self.backend.input_dtype)
        for _ in range(runs):
            self.backend.infer(dummy)

    def deblur(self, image, max_input_side=None, overlap=None):
        """Deblur a BGR uint8 image of any size. Output has the same size."""
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise DeblurError("Expected a colour (H, W, 3) image")
        t0 = time.time()
        prep = pre.prepare_image(image, self.tile_size,
                                 self.overlap if overlap is None else overlap,
                                 self.max_input_side if max_input_side is None else max_input_side)
        acc = post.TileAccumulator(prep)
        infer_s = 0.0
        with self.lock:
            try:
                for pos, tensor in pre.iter_tile_tensors(prep, self.backend.input_dtype):
                    t = time.time()
                    out = self.backend.infer(tensor)
                    infer_s += time.time() - t
                    acc.add(pos, post.from_tensor(out))
            except MemoryError as exc:
                raise InsufficientResourcesError("Out of RAM while processing: %s" % exc)
            except DeblurError:
                raise
            except Exception as exc:
                if is_oom_error(exc):
                    raise InsufficientResourcesError("Out of GPU memory: %s" % exc)
                raise DeblurError("Inference failed: %s" % exc)
        restored = acc.result()
        if not np.isfinite(restored).all():
            self.log("WARNING: model produced NaN/Inf values (FP16 overflow?) - "
                     "rebuild the engine with FP32 LayerNorm (default) or precision fp32.")
        result_img = post.finalize(prep, restored)
        total = (time.time() - t0) * 1000.0
        return DeblurResult(result_img, infer_s * 1000.0, total, len(prep.tiles), prep.work_shape)

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None

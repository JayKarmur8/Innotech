"""Build a TensorRT engine from an ONNX model (run this ON the Jetson Nano).

TensorRT engines are specific to the GPU + TensorRT version they were built
with, so they must be built on the target device. Supports the TensorRT 8.x
API shipped with JetPack 4.6 as well as the newer TensorRT 10 API.
"""

import os
import time

from .errors import ModelNotFoundError, ModelLoadError, InsufficientResourcesError

NORM_LAYER_HINTS = ("norm",)


def _keep_norm_layers_fp32(trt, network, config, log):
    """Force LayerNorm sub-layers (mean/var/sqrt/div) to FP32.

    NAFNet computes LayerNorm with ReduceMean/Pow/Sqrt. In pure FP16 the variance
    can overflow and produce NaNs/black patches; keeping only these small layers
    in FP32 costs a few % speed and keeps FP16 everywhere else.
    """
    wanted = {trt.LayerType.REDUCE, trt.LayerType.ELEMENTWISE, trt.LayerType.UNARY}
    count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in wanted and any(h in layer.name.lower() for h in NORM_LAYER_HINTS):
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            count += 1
    if count:
        if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        else:  # TensorRT 8.2 (JetPack 4.6)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    log("  kept %d normalisation layers in FP32" % count)


def build_engine(onnx_path, engine_path, fp16=True, workspace_mb=1024,
                 tile_size=256, fp32_norm=True, log=print):
    """Parse ``onnx_path`` and serialize an optimized engine to ``engine_path``."""
    try:
        import tensorrt as trt
    except ImportError:
        raise ModelLoadError("TensorRT Python bindings are not installed",
                             hint="On Jetson they come with JetPack (python3-libnvinfer). "
                                  "If you use a virtualenv create it with --system-site-packages.")
    if not os.path.isfile(onnx_path):
        raise ModelNotFoundError("ONNX model not found: %s" % onnx_path)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    log("Parsing %s ..." % onnx_path)
    with open(onnx_path, "rb") as fh:
        if not parser.parse(fh.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise ModelLoadError("TensorRT could not parse the ONNX model:\n" + "\n".join(errors),
                                 hint="Re-export with: python3 tools/export_onnx.py --opset 11")

    config = builder.create_builder_config()
    workspace = int(workspace_mb) << 20
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    else:
        config.max_workspace_size = workspace

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            log("  FP16 mode enabled")
            if fp32_norm:
                _keep_norm_layers_fp32(trt, network, config, log)
        else:
            log("  WARNING: this GPU has no fast FP16, building FP32 engine")

    # Dynamic-shape models need an optimization profile. We fix it to one tile.
    inp = network.get_input(0)
    shape = list(inp.shape)
    if any(d < 0 for d in shape):
        fixed = (1, 3, tile_size, tile_size)
        profile = builder.create_optimization_profile()
        profile.set_shape(inp.name, fixed, fixed, fixed)
        config.add_optimization_profile(profile)
        log("  dynamic input -> optimization profile %s" % (fixed,))

    log("Building TensorRT engine (this takes a few minutes on Jetson Nano) ...")
    t0 = time.time()
    try:
        if hasattr(builder, "build_serialized_network"):
            serialized = builder.build_serialized_network(network, config)
        else:  # TensorRT 7
            engine = builder.build_engine(network, config)
            serialized = engine.serialize() if engine is not None else None
    except MemoryError:
        raise InsufficientResourcesError("Out of memory while building the TensorRT engine")
    if serialized is None:
        raise InsufficientResourcesError(
            "TensorRT failed to build the engine (often: not enough memory)",
            hint="Use --workspace 512, enable swap, close other apps and retry.")

    folder = os.path.dirname(os.path.abspath(engine_path))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(engine_path, "wb") as fh:
        fh.write(serialized)
    log("Engine saved to %s (%.1f MB) in %.0f s"
        % (engine_path, os.path.getsize(engine_path) / 1e6, time.time() - t0))
    return engine_path

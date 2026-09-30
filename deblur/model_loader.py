"""Model loading: picks the fastest available inference backend.

Backends (fastest first on Jetson Nano):
    TensorRTBackend    - serialized .engine, FP16, PyCUDA buffers (GPU)
    OnnxRuntimeBackend - onnxruntime-gpu with TensorRT / CUDA execution providers
    OpenCVDNNBackend   - cv2.dnn with CUDA target if available, else CPU

Every backend exposes the same interface:
    backend.infer(nchw_array) -> nchw_float32_array
    backend.input_shape  (N, C, H, W) - H/W may be None for dynamic models
    backend.input_dtype  numpy dtype expected by the model
    backend.description  human readable string for the UI
"""

import os

import numpy as np

from .config import resolve_path
from .errors import (BackendUnavailableError, InsufficientResourcesError,
                     ModelLoadError, ModelNotFoundError)

OOM_MARKERS = ("out of memory", "outofmemory", "cudaerrormemoryallocation",
               "failed to allocate", "bad_alloc", "cuda_error_out_of_memory")


def is_oom_error(exc):
    text = str(exc).lower()
    return isinstance(exc, MemoryError) or any(m in text for m in OOM_MARKERS)


def _static(dim):
    return int(dim) if isinstance(dim, (int, np.integer)) and int(dim) > 0 else None


# ============================================================ TensorRT backend
class TensorRTBackend(object):
    """Runs a serialized TensorRT engine with pre-allocated pinned buffers.

    A private CUDA context is created and pushed/popped around each call so the
    backend can be used from any thread (the GUI runs inference in a worker).
    """

    def __init__(self, engine_path):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
        except ImportError as exc:
            raise BackendUnavailableError("TensorRT backend unavailable: %s" % exc,
                                          hint="sudo apt install python3-libnvinfer && pip3 install pycuda")
        self.trt, self.cuda = trt, cuda
        cuda.init()
        if cuda.Device.count() == 0:
            raise BackendUnavailableError("No CUDA GPU found")
        self.ctx = cuda.Device(0).make_context()
        try:
            self._load(engine_path)
        except Exception as exc:
            self.ctx.pop()
            self.ctx.detach()
            if is_oom_error(exc):
                raise InsufficientResourcesError("Not enough GPU memory to load the engine: %s" % exc)
            if isinstance(exc, (ModelLoadError, InsufficientResourcesError)):
                raise
            raise ModelLoadError("Could not load TensorRT engine %s: %s" % (engine_path, exc))
        self.ctx.pop()

    def _load(self, engine_path):
        trt, cuda = self.trt, self.cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        try:
            trt.init_libnvinfer_plugins(self.logger, "")
        except Exception:
            pass
        with open(engine_path, "rb") as fh, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise ModelLoadError("TensorRT failed to deserialize %s" % engine_path)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise InsufficientResourcesError("Could not create TensorRT execution context")
        self.stream = cuda.Stream()
        self.new_api = hasattr(self.engine, "num_io_tensors")   # TensorRT >= 8.5

        self.inputs, self.outputs, self.bindings = [], [], []
        names = ([self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
                 if self.new_api else [self.engine.get_binding_name(i)
                                       for i in range(self.engine.num_bindings)])
        for idx, name in enumerate(names):
            if self.new_api:
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            else:
                shape = tuple(self.engine.get_binding_shape(idx))
                dtype = trt.nptype(self.engine.get_binding_dtype(idx))
                is_input = self.engine.binding_is_input(idx)
            if any(d < 0 for d in shape):
                raise ModelLoadError("Engine has dynamic shape %s; rebuild with tools/build_engine.py" % (shape,))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            entry = {"name": name, "shape": shape, "dtype": dtype, "host": host, "dev": dev}
            (self.inputs if is_input else self.outputs).append(entry)
            self.bindings.append(int(dev))
            if self.new_api:
                self.context.set_tensor_address(name, int(dev))
        if len(self.inputs) != 1 or len(self.outputs) < 1:
            raise ModelLoadError("Expected 1 input and >=1 output, got %d/%d"
                                 % (len(self.inputs), len(self.outputs)))
        inp = self.inputs[0]
        self.input_shape = inp["shape"]
        self.input_dtype = np.dtype(inp["dtype"])
        self.description = "TensorRT %s (GPU, engine %s)" % (trt.__version__, os.path.basename(engine_path))

    def infer(self, x):
        cuda = self.cuda
        inp, out = self.inputs[0], self.outputs[0]
        np.copyto(inp["host"], x.astype(inp["dtype"], copy=False).ravel())
        self.ctx.push()
        try:
            cuda.memcpy_htod_async(inp["dev"], inp["host"], self.stream)
            if self.new_api:
                ok = self.context.execute_async_v3(self.stream.handle)
            else:
                ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(out["host"], out["dev"], self.stream)
            self.stream.synchronize()
        except Exception as exc:
            if is_oom_error(exc):
                raise InsufficientResourcesError("GPU out of memory during inference: %s" % exc)
            raise
        finally:
            self.ctx.pop()
        if ok is False:
            raise ModelLoadError("TensorRT execution failed")
        return out["host"].reshape(out["shape"]).astype(np.float32)

    def close(self):
        try:
            self.ctx.push()
            for e in self.inputs + self.outputs:
                e["dev"].free()
            del self.context, self.engine
            self.ctx.pop()
            self.ctx.detach()
        except Exception:
            pass


# ======================================================== ONNX Runtime backend
class OnnxRuntimeBackend(object):
    def __init__(self, onnx_path, precision="fp16", cache_dir=None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendUnavailableError("ONNX Runtime unavailable: %s" % exc,
                                          hint="Install onnxruntime-gpu for Jetson (see README).")
        available = ort.get_available_providers()
        providers = []
        if "TensorrtExecutionProvider" in available:
            trt_opts = {"trt_fp16_enable": precision == "fp16",
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": cache_dir or os.path.dirname(onnx_path) or ".",
                        "trt_max_workspace_size": 1 << 30}
            providers.append(("TensorrtExecutionProvider", trt_opts))
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        except Exception as exc:
            if is_oom_error(exc):
                raise InsufficientResourcesError("Not enough memory to load the ONNX model: %s" % exc)
            raise ModelLoadError("ONNX Runtime could not load %s: %s" % (onnx_path, exc))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = tuple(_static(d) for d in inp.shape)
        self.input_dtype = np.dtype(np.float16 if "float16" in inp.type else np.float32)
        used = self.session.get_providers()[0].replace("ExecutionProvider", "")
        device = "CPU" if used == "CPU" else "GPU"
        self.description = "ONNX Runtime %s (%s)" % (ort.__version__, used if device == "CPU"
                                                         else used + ", GPU")
        if device == "CPU":
            self.description += " - slow, install TensorRT for GPU speed"

    def infer(self, x):
        try:
            out = self.session.run([self.output_name], {self.input_name: x.astype(self.input_dtype, copy=False)})
        except Exception as exc:
            if is_oom_error(exc):
                raise InsufficientResourcesError("Out of memory during inference: %s" % exc)
            raise
        return np.asarray(out[0], dtype=np.float32)

    def close(self):
        self.session = None


# ========================================================== OpenCV DNN backend
class OpenCVDNNBackend(object):
    def __init__(self, onnx_path, precision="fp16", tile_size=256):
        import cv2
        try:
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
        except cv2.error as exc:
            raise ModelLoadError("OpenCV DNN could not load %s (OpenCV %s may be too old): %s"
                                 % (onnx_path, cv2.__version__, str(exc).strip().splitlines()[-1]))
        device = "CPU"
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16 if precision == "fp16"
                                             else cv2.dnn.DNN_TARGET_CUDA)
                device = "GPU/CUDA"
        except Exception:
            device = "CPU"
        self.input_shape = (1, 3, tile_size, tile_size)
        self.input_dtype = np.dtype(np.float32)
        self.description = "OpenCV DNN %s (%s)" % (cv2.__version__, device)

    def infer(self, x):
        try:
            self.net.setInput(x.astype(np.float32, copy=False))
            return np.asarray(self.net.forward(), dtype=np.float32)
        except Exception as exc:
            if is_oom_error(exc):
                raise InsufficientResourcesError("Out of memory during inference: %s" % exc)
            raise

    def close(self):
        self.net = None


# ================================================================== factory
def _try(errors, name, fn):
    try:
        return fn()
    except (BackendUnavailableError, ModelLoadError) as exc:
        errors.append("%s: %s" % (name, exc))
        return None


def load_backend(cfg, log=print):
    """Create the inference backend described by ``cfg['model']``."""
    mcfg = cfg["model"]
    backend = str(mcfg["backend"]).lower()
    precision = str(mcfg["precision"]).lower()
    onnx_path = resolve_path(mcfg["onnx_path"])
    engine_path = resolve_path(mcfg["engine_path"])
    tile = int(cfg["processing"]["tile_size"])
    has_onnx = bool(onnx_path) and os.path.isfile(onnx_path)
    has_engine = bool(engine_path) and os.path.isfile(engine_path)

    if not has_onnx and not has_engine:
        raise ModelNotFoundError("No model found.\n  ONNX:   %s\n  Engine: %s" % (onnx_path, engine_path))

    errors = []
    if backend in ("auto", "tensorrt"):
        if not has_engine and has_onnx and mcfg.get("build_engine_if_missing") and _trt_installed():
            from .trt_builder import build_engine
            log("TensorRT engine not found - building it from the ONNX model (one-time step).")
            build_engine(onnx_path, engine_path, fp16=(precision == "fp16"),
                         workspace_mb=int(mcfg["workspace_mb"]), tile_size=tile, log=log)
            has_engine = True
        if has_engine:
            b = _try(errors, "TensorRT", lambda: TensorRTBackend(engine_path))
            if b:
                return b
        elif backend == "tensorrt":
            raise ModelNotFoundError("TensorRT engine not found: %s" % engine_path)
        if backend == "tensorrt":
            raise BackendUnavailableError("TensorRT backend failed:\n  " + "\n  ".join(errors))

    if not has_onnx:
        raise ModelNotFoundError("ONNX model not found: %s (needed for backend '%s')" % (onnx_path, backend)
                                 + ("\nOther errors:\n  " + "\n  ".join(errors) if errors else ""))

    if backend in ("auto", "onnxruntime"):
        b = _try(errors, "ONNX Runtime", lambda: OnnxRuntimeBackend(onnx_path, precision))
        if b:
            return b
        if backend == "onnxruntime":
            raise BackendUnavailableError("ONNX Runtime backend failed:\n  " + "\n  ".join(errors))

    b = _try(errors, "OpenCV DNN", lambda: OpenCVDNNBackend(onnx_path, precision, tile))
    if b:
        return b
    raise BackendUnavailableError("No inference backend could load the model:\n  " + "\n  ".join(errors))


def _trt_installed():
    try:
        import tensorrt  # noqa: F401
        import pycuda.driver  # noqa: F401
        return True
    except ImportError:
        return False


def environment_report():
    """Return a list of (component, status) strings - used by `main.py info`."""
    rows = []
    import platform
    import cv2
    rows.append(("Python", platform.python_version()))
    rows.append(("NumPy", np.__version__))
    cuda_cv = 0
    try:
        cuda_cv = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        pass
    rows.append(("OpenCV", "%s (CUDA devices: %d, GStreamer: %s)"
                 % (cv2.__version__, cuda_cv, _gstreamer_status(cv2))))
    try:
        import tensorrt
        rows.append(("TensorRT", tensorrt.__version__))
    except ImportError:
        rows.append(("TensorRT", "not installed"))
    try:
        import pycuda.driver as cuda
        cuda.init()
        dev = cuda.Device(0)
        rows.append(("PyCUDA / GPU", "%s, %.0f MB" % (dev.name(), dev.total_memory() / 1e6)))
    except Exception as exc:
        rows.append(("PyCUDA / GPU", "unavailable (%s)" % str(exc).splitlines()[0][:60]))
    try:
        import onnxruntime as ort
        rows.append(("ONNX Runtime", "%s %s" % (ort.__version__, ort.get_available_providers())))
    except ImportError:
        rows.append(("ONNX Runtime", "not installed"))
    model = "/proc/device-tree/model"
    if os.path.exists(model):
        with open(model) as fh:
            rows.append(("Board", fh.read().strip("\x00\n ")))
    return rows


def _gstreamer_status(cv2):
    try:
        info = cv2.getBuildInformation()
        for line in info.splitlines():
            if "GStreamer" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"

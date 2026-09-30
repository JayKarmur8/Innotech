"""Configuration loading.

The configuration lives in ``config.yaml``. Missing keys fall back to
``DEFAULTS`` so an incomplete (or missing) file never crashes the app.
"""

import copy
import os

from .errors import ConfigError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

DEFAULTS = {
    "model": {
        "name": "NAFNet-GoPro-width32",
        "onnx_path": "models/nafnet_gopro_w32_256.onnx",
        "engine_path": "models/nafnet_gopro_w32_256_fp16.engine",
        "backend": "auto",
        "precision": "fp16",
        "build_engine_if_missing": True,
        "workspace_mb": 1024,
    },
    "processing": {
        "tile_size": 256,
        "tile_overlap": 32,
        "max_input_side": 1280,
    },
    "camera": {
        "source": "usb",
        "device_index": 0,
        "csi_sensor_id": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "flip_method": 0,
        "reconnect_attempts": 3,
    },
    "realtime": {
        "process_width": 320,
        "view": "side_by_side",
        "display_width": 1280,
    },
    "output": {
        "save_dir": "results",
        "save_comparison": True,
    },
    "ui": {
        "preview_max_width": 520,
        "preview_max_height": 420,
    },
}

VALID_BACKENDS = ("auto", "tensorrt", "onnxruntime", "opencv")
VALID_PRECISIONS = ("fp16", "fp32")


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_path(path):
    """Resolve a config path relative to the project root."""
    if not path:
        return path
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_config(path=None):
    """Load ``config.yaml`` merged over the defaults and validate it."""
    path = path or DEFAULT_CONFIG_PATH
    user_cfg = {}
    if os.path.isfile(path):
        try:
            import yaml
        except ImportError:
            raise ConfigError("PyYAML is not installed, cannot read %s" % path,
                              hint="pip3 install pyyaml  (or: sudo apt install python3-yaml)")
        try:
            with open(path, "r") as fh:
                user_cfg = yaml.safe_load(fh) or {}
        except Exception as exc:  # yaml.YAMLError, IOError ...
            raise ConfigError("Could not parse config file %s: %s" % (path, exc))
        if not isinstance(user_cfg, dict):
            raise ConfigError("Config file %s must contain a YAML mapping" % path)
    elif path != DEFAULT_CONFIG_PATH:
        raise ConfigError("Config file not found: %s" % path)

    cfg = _merge(DEFAULTS, user_cfg)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    model = cfg["model"]
    if str(model["backend"]).lower() not in VALID_BACKENDS:
        raise ConfigError("model.backend must be one of %s, got %r"
                          % (", ".join(VALID_BACKENDS), model["backend"]))
    if str(model["precision"]).lower() not in VALID_PRECISIONS:
        raise ConfigError("model.precision must be fp16 or fp32, got %r" % model["precision"])
    proc = cfg["processing"]
    if int(proc["tile_size"]) < 32:
        raise ConfigError("processing.tile_size must be >= 32")
    if not 0 <= int(proc["tile_overlap"]) < int(proc["tile_size"]) // 2:
        raise ConfigError("processing.tile_overlap must be in [0, tile_size/2)")
    if int(proc["max_input_side"]) < 0:
        raise ConfigError("processing.max_input_side must be >= 0")
    if int(cfg["realtime"]["process_width"]) < 32:
        raise ConfigError("realtime.process_width must be >= 32")

"""Custom exceptions with user-friendly messages.

Every error raised by the pipeline derives from ``DeblurError`` so that the UI
layers can catch a single type and show a readable message instead of crashing.
"""


class DeblurError(Exception):
    """Base class for all errors raised by the deblurring system."""

    hint = ""

    def __init__(self, message, hint=None):
        super(DeblurError, self).__init__(message)
        if hint is not None:
            self.hint = hint

    def user_message(self):
        msg = str(self)
        if self.hint:
            msg += "\n\nHint: " + self.hint
        return msg


class ConfigError(DeblurError):
    hint = "Check config.yaml (see README section 'Configuring the model')."


class ModelNotFoundError(DeblurError):
    hint = ("Download and export the model first:\n"
            "  python3 tools/download_weights.py\n"
            "  python3 tools/export_onnx.py\n"
            "  python3 tools/build_engine.py   (on the Jetson Nano)\n"
            "See README section 'Model setup'.")


class BackendUnavailableError(DeblurError):
    hint = ("Install at least one inference backend: TensorRT + PyCUDA (JetPack), "
            "onnxruntime-gpu, or OpenCV >= 4.5 with DNN.")


class ModelLoadError(DeblurError):
    hint = ("The model file may be corrupted or built for a different TensorRT "
            "version. Rebuild the engine on this device with tools/build_engine.py.")


class InsufficientResourcesError(DeblurError):
    hint = ("The GPU/RAM ran out of memory. Try: a smaller tile_size / "
            "max_input_side in config.yaml, the FP16 TensorRT engine, the lite "
            "model, closing other apps, or adding swap (see README Troubleshooting).")


class UnsupportedImageError(DeblurError):
    hint = "Supported formats: .jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp"


class CameraError(DeblurError):
    hint = ("Check the cable, run 'ls /dev/video*' (USB) or "
            "'nvgstcapture-1.0' (CSI), and set camera.source / camera.device_index "
            "in config.yaml. Close other programs that use the camera.")

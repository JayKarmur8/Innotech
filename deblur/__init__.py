"""AI Image Deblurring System for NVIDIA Jetson Nano.

Pipeline:
    image / camera frame -> preprocessing -> AI model (TensorRT / ONNX Runtime /
    OpenCV-DNN) -> post-processing -> restored image -> metrics -> UI
"""

__version__ = "1.0.0"

"""Camera handling for USB (V4L2) and CSI (nvarguscamerasrc) cameras.

A background thread keeps grabbing frames and only the newest one is kept,
so a slow deblurring model never processes stale, buffered frames
(= minimal latency in real-time mode).
"""

import os
import threading
import time

import cv2

from .errors import CameraError


def csi_pipeline(sensor_id=0, width=1280, height=720, fps=30, flip_method=0):
    """GStreamer pipeline for Jetson CSI cameras (IMX219 / Raspberry Pi v2 ...)."""
    return ("nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1, format=(string)NV12 ! "
            "nvvidconv flip-method=%d ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1 max-buffers=1 sync=false"
            % (sensor_id, width, height, fps, flip_method, width, height))


def list_usb_cameras(max_index=8):
    return [i for i in range(max_index) if os.path.exists("/dev/video%d" % i)]


class Camera(object):
    """Threaded OpenCV camera. Use as a context manager or call open()/release()."""

    def __init__(self, cam_cfg):
        self.cfg = cam_cfg
        self.cap = None
        self.frame = None
        self.frame_id = 0
        self.error = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    # ----------------------------------------------------------------- open
    def _create_capture(self):
        c = self.cfg
        src = str(c.get("source", "usb")).lower()
        w, h, fps = int(c["width"]), int(c["height"]), int(c["fps"])
        if src == "csi":
            if "gstreamer" not in cv2.getBuildInformation().lower():
                raise CameraError("This OpenCV build has no GStreamer support (needed for CSI cameras)",
                                  hint="Use the OpenCV that ships with JetPack (/usr/lib/python3/dist-packages); "
                                       "do NOT pip install opencv-python on the Jetson.")
            pipe = csi_pipeline(int(c.get("csi_sensor_id", 0)), w, h, fps, int(c.get("flip_method", 0)))
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            desc = "CSI camera (sensor %s)" % c.get("csi_sensor_id", 0)
        elif src == "usb":
            idx = int(c.get("device_index", 0))
            if os.name == "posix" and os.path.isdir("/dev") and not os.path.exists("/dev/video%d" % idx):
                found = list_usb_cameras()
                raise CameraError("USB camera /dev/video%d not found (available: %s)"
                                  % (idx, ", ".join("/dev/video%d" % i for i in found) or "none"))
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if os.name == "posix" else cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # higher FPS on USB2
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            desc = "USB camera /dev/video%d" % idx
        else:  # video file or network stream
            if "://" not in src and not os.path.isfile(c["source"]):
                raise CameraError("Video source not found: %s" % c["source"])
            cap = cv2.VideoCapture(c["source"])
            desc = "video %s" % c["source"]
        if not cap.isOpened():
            cap.release()
            raise CameraError("Could not open %s" % desc)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise CameraError("%s opened but returned no frames" % desc)
        self.description = "%s %dx%d" % (desc, frame.shape[1], frame.shape[0])
        self.is_file = src not in ("usb", "csi") and "://" not in src
        return cap, frame

    def open(self):
        attempts = max(1, int(self.cfg.get("reconnect_attempts", 3)))
        last = None
        for i in range(attempts):
            try:
                self.cap, first = self._create_capture()
                break
            except CameraError as exc:
                last = exc
                if i + 1 < attempts:
                    time.sleep(1.0)
        else:
            raise last
        with self.lock:
            self.frame, self.frame_id, self.error = first, 1, None
        self.running = True
        self.thread = threading.Thread(target=self._reader, name="camera-reader")
        self.thread.daemon = True
        self.thread.start()
        return self

    # --------------------------------------------------------------- reader
    def _reader(self):
        failures = 0
        delay = 1.0 / max(1, int(self.cfg.get("fps", 30))) if getattr(self, "is_file", False) else 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if getattr(self, "is_file", False):   # loop video files for demos
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                failures += 1
                if failures > 50:
                    with self.lock:
                        self.error = CameraError("Camera stopped delivering frames (disconnected?)")
                    self.running = False
                    break
                time.sleep(0.02)
                continue
            failures = 0
            with self.lock:
                self.frame = frame
                self.frame_id += 1
            if delay:
                time.sleep(delay)

    def read(self, timeout=2.0, last_id=None):
        """Return (frame_id, frame copy). Waits for a frame newer than ``last_id``."""
        t0 = time.time()
        while True:
            with self.lock:
                if self.error is not None:
                    raise self.error
                if self.frame is not None and (last_id is None or self.frame_id != last_id):
                    return self.frame_id, self.frame.copy()
            if time.time() - t0 > timeout:
                raise CameraError("Timed out waiting for a camera frame")
            time.sleep(0.002)

    def snapshot(self):
        return self.read()[1]

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()

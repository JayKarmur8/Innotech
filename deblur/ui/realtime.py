"""Real-time camera deblurring in an OpenCV window (works without Tkinter).

Keys:  q / ESC quit   s save frame   v change view   p pause
       + / - increase / decrease the processing width (quality vs. FPS)
"""

import os
import time

import cv2
import numpy as np

from ..camera import Camera
from ..config import resolve_path
from ..errors import DeblurError
from ..image_io import (put_label, resize_to_width, save_image, side_by_side,
                        split_view, timestamped_name)
from ..metrics import FPSMeter

VIEWS = ("side_by_side", "output_only", "split")


def process_frame(engine, frame, process_width):
    """Deblur one camera frame at reduced resolution; returns (restored, result)."""
    small = resize_to_width(frame, min(process_width, frame.shape[1]))
    result = engine.deblur(small, max_input_side=0)
    return small, result


def compose_view(view, small_in, small_out, display_width):
    if view == "output_only":
        img = small_out
    elif view == "split":
        img = split_view(small_in, small_out)
    else:
        img = side_by_side(small_in, small_out, labels=("Camera (blurred)", "AI deblurred"))
    return resize_to_width(img, display_width)


def draw_stats(img, fps, infer_ms, total_ms, backend, process_shape):
    lines = ["FPS: %.1f" % fps,
             "Inference: %.1f ms  Total: %.1f ms" % (infer_ms, total_ms),
             "Process res: %dx%d" % (process_shape[1], process_shape[0]),
             backend[:60]]
    y = img.shape[0] - 12 - 26 * (len(lines) - 1)
    for line in lines:
        put_label(img, line, (10, y), 0.6, (0, 255, 255))
        y += 26
    return img


def run_realtime(engine, cfg, window="AI Deblurring - real-time (q to quit)"):
    rt = cfg["realtime"]
    process_width = int(rt["process_width"])
    view = rt.get("view", "side_by_side")
    view = view if view in VIEWS else VIEWS[0]
    display_width = int(rt.get("display_width", 1280))
    save_dir = resolve_path(cfg["output"]["save_dir"])
    fps = FPSMeter()
    paused = False
    last = None

    with Camera(cfg["camera"]) as cam:
        print("Camera: %s | model: %s" % (cam.description, engine.description))
        engine.warmup()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        last_id = None
        while True:
            if not paused:
                last_id, frame = cam.read(last_id=last_id)
                small, result = process_frame(engine, frame, process_width)
                fps.tick()
                shown = compose_view(view, small, result.image, display_width)
                draw_stats(shown, fps.fps, result.inference_ms, result.total_ms,
                           engine.description, small.shape)
                last = (frame, small, result, shown)
            if last is not None:
                cv2.imshow(window, last[3])
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("v"):
                view = VIEWS[(VIEWS.index(view) + 1) % len(VIEWS)]
            elif key == ord("p"):
                paused = not paused
            elif key in (ord("+"), ord("=")):
                process_width = min(process_width + 64, 1920)
                print("process width:", process_width)
            elif key in (ord("-"), ord("_")):
                process_width = max(process_width - 64, 128)
                print("process width:", process_width)
            elif key == ord("s") and last is not None:
                path = save_image(os.path.join(save_dir, timestamped_name("realtime")), last[3])
                save_image(path.replace(".png", "_restored.png"), last[2].image)
                print("saved", path)
    cv2.destroyAllWindows()

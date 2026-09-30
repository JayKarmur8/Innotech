"""Tkinter desktop GUI for the AI deblurring system.

Heavy work (model loading, inference, camera) runs in worker threads; results
are passed back through a queue and applied on the Tk main thread, so the
window never freezes.
"""

import os
import queue
import threading
import time
import traceback

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover
    raise ImportError("Tkinter is missing. Install it with: sudo apt install python3-tk")
try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    raise ImportError("Pillow ImageTk is missing. Install it with: "
                      "sudo apt install python3-pil python3-pil.imagetk")

from ..camera import Camera
from ..config import resolve_path
from ..errors import DeblurError
from ..image_io import (SUPPORTED_EXTENSIONS, difference_heatmap, fit_within,
                        load_image, save_image, side_by_side, split_view,
                        timestamped_name)
from ..inference import DeblurEngine
from ..metrics import FPSMeter, evaluate, format_metrics
from .realtime import compose_view, process_frame

BG = "#1e1f24"
PANEL = "#2a2c33"
FG = "#e8e8e8"
ACCENT = "#76b900"   # NVIDIA green


def error_text(exc):
    if isinstance(exc, DeblurError):
        return exc.user_message()
    return "%s: %s" % (type(exc).__name__, exc)


class DeblurApp(object):
    VIEWS = ("Side by side", "Split slider", "Difference map")

    def __init__(self, root, cfg):
        self.root = root
        self.cfg = cfg
        self.engine = None
        self.blurred = None       # BGR uint8
        self.restored = None
        self.ground_truth = None
        self.source_name = "image"
        self.busy = False
        self.realtime_running = False
        self.events = queue.Queue()
        self._photos = {}

        root.title("AI Image Deblurring - Jetson Nano")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build_widgets()
        self._set_buttons()
        self.root.after(30, self._poll_events)
        self.run_async(self._load_engine, "Loading AI model (first TensorRT build can take minutes)...")

    # ================================================================ layout
    def _build_widgets(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=6, font=("DejaVu Sans", 10))
        style.configure("Accent.TButton", foreground="white", background=ACCENT,
                        font=("DejaVu Sans", 11, "bold"))
        style.map("Accent.TButton", background=[("disabled", "#555"), ("active", "#8bd400")])

        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=10, pady=(10, 4))
        self.btn_upload = ttk.Button(bar, text="Upload Image", command=self.on_upload)
        self.btn_camera = ttk.Button(bar, text="Camera Capture", command=self.on_camera)
        self.btn_gt = ttk.Button(bar, text="Load Ground Truth", command=self.on_ground_truth)
        self.btn_deblur = ttk.Button(bar, text="Deblur", style="Accent.TButton", command=self.on_deblur)
        self.btn_save = ttk.Button(bar, text="Save Result", command=self.on_save)
        self.btn_live = ttk.Button(bar, text="Real-time Mode", command=self.on_realtime)
        for b in (self.btn_upload, self.btn_camera, self.btn_gt, self.btn_deblur, self.btn_save, self.btn_live):
            b.pack(side="left", padx=3)

        tk.Label(bar, text="  Compare:", bg=BG, fg=FG).pack(side="left")
        self.view_var = tk.StringVar(value=self.VIEWS[0])
        view_box = ttk.Combobox(bar, textvariable=self.view_var, values=self.VIEWS,
                                state="readonly", width=15)
        view_box.pack(side="left", padx=3)
        view_box.bind("<<ComboboxSelected>>", lambda e: self._refresh_previews())

        panels = tk.Frame(self.root, bg=BG)
        panels.pack(fill="both", expand=True, padx=10, pady=4)
        self.pw = int(self.cfg["ui"]["preview_max_width"])
        self.ph = int(self.cfg["ui"]["preview_max_height"])
        self.lbl_left_title, self.lbl_left = self._panel(panels, "Original (blurred)")
        self.lbl_right_title, self.lbl_right = self._panel(panels, "Deblurred (AI)")

        self.split_var = tk.DoubleVar(value=0.5)
        self.split_scale = ttk.Scale(self.root, from_=0.0, to=1.0, variable=self.split_var,
                                     command=lambda v: self._refresh_previews())

        info = self.info_frame = tk.Frame(self.root, bg=PANEL)
        info.pack(fill="x", padx=10, pady=(4, 10))
        self.stats_var = tk.StringVar(value="Processing time: -")
        self.metrics_var = tk.StringVar(value="Metrics: -")
        self.status_var = tk.StringVar(value="Starting...")
        tk.Label(info, textvariable=self.stats_var, bg=PANEL, fg=ACCENT, anchor="w",
                 font=("DejaVu Sans Mono", 11, "bold")).pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(info, textvariable=self.metrics_var, bg=PANEL, fg=FG, anchor="w", justify="left",
                 font=("DejaVu Sans Mono", 10)).pack(fill="x", padx=8)
        tk.Label(info, textvariable=self.status_var, bg=PANEL, fg="#9aa0aa", anchor="w",
                 font=("DejaVu Sans", 9)).pack(fill="x", padx=8, pady=(0, 6))

    def _panel(self, parent, title):
        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(side="left", fill="both", expand=True, padx=4)
        t = tk.Label(frame, text=title, bg=PANEL, fg=FG, font=("DejaVu Sans", 12, "bold"))
        t.pack(pady=(6, 2))
        # width/height are in characters while the label shows text
        lbl = tk.Label(frame, bg="#111", width=self.pw // 8, height=self.ph // 16,
                       text="No image", fg="#666")
        lbl.pack(padx=6, pady=(0, 6))
        return t, lbl

    # ============================================================= threading
    def run_async(self, fn, status, *args):
        """Run ``fn(*args)`` in a worker; its return value is posted as an event."""
        self.busy = True
        self._set_buttons()
        self.status_var.set(status)

        def worker():
            try:
                self.events.put(("done", fn(*args)))
            except Exception as exc:  # show every error in a dialog, never crash
                traceback.print_exc()
                self.events.put(("error", exc))
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "error":
                    self.busy = False
                    self.status_var.set("Error - see dialog")
                    messagebox.showerror("Error", error_text(payload))
                elif kind == "done":
                    self.busy = False
                    if callable(payload):
                        payload()           # UI update closure from the worker
                elif kind == "frame":
                    self._show_live(payload)
                elif kind == "live_stopped":
                    self._live_stopped(payload)
                self._set_buttons()
        except queue.Empty:
            pass
        self.root.after(30, self._poll_events)

    def _set_buttons(self):
        idle = not self.busy and not self.realtime_running
        ready = idle and self.engine is not None

        def st(b, on):
            b.state(["!disabled"] if on else ["disabled"])
        st(self.btn_upload, idle)
        st(self.btn_camera, idle)
        st(self.btn_gt, idle and self.blurred is not None)
        st(self.btn_deblur, ready and self.blurred is not None)
        st(self.btn_save, idle and self.restored is not None)
        st(self.btn_live, (ready or self.realtime_running) and not self.busy)
        self.btn_live.configure(text="Stop Real-time" if self.realtime_running else "Real-time Mode")

    # ================================================================ model
    def _load_engine(self):
        engine = DeblurEngine(self.cfg, log=lambda m: print(m))
        engine.warmup()

        def apply():
            self.engine = engine
            self.status_var.set("Model ready: %s  |  %s" % (engine.model_name, engine.description))
        return apply

    # ============================================================== actions
    def on_upload(self):
        types = [("Images", " ".join("*" + e for e in SUPPORTED_EXTENSIONS)), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select a blurred image", filetypes=types)
        if not path:
            return
        try:
            img = load_image(path)
        except DeblurError as exc:
            messagebox.showerror("Cannot open image", exc.user_message())
            return
        self._set_input(img, os.path.splitext(os.path.basename(path))[0])
        self.status_var.set("Loaded %s (%dx%d)" % (os.path.basename(path), img.shape[1], img.shape[0]))

    def on_ground_truth(self):
        path = filedialog.askopenfilename(title="Select the SHARP ground-truth image",
                                          filetypes=[("Images", " ".join("*" + e for e in SUPPORTED_EXTENSIONS))])
        if not path:
            return
        try:
            self.ground_truth = load_image(path)
        except DeblurError as exc:
            messagebox.showerror("Cannot open image", exc.user_message())
            return
        self.status_var.set("Ground truth loaded: %s" % os.path.basename(path))
        self._update_metrics()

    def on_camera(self):
        def grab():
            with Camera(self.cfg["camera"]) as cam:
                time.sleep(0.5)          # let auto-exposure settle
                frame = cam.snapshot()
                desc = cam.description

            def apply():
                self._set_input(frame, "camera")
                self.status_var.set("Captured frame from %s" % desc)
            return apply
        self.run_async(grab, "Opening camera...")

    def on_deblur(self):
        img = self.blurred

        def work():
            result = self.engine.deblur(img)
            metrics = evaluate(img, result.image, self.ground_truth)

            def apply():
                self.restored = result.image
                h, w = result.work_shape
                self.stats_var.set("Processing time: %.0f ms  (inference %.0f ms, %d tile%s @ %dx%d)"
                                   % (result.total_ms, result.inference_ms, result.tiles,
                                      "" if result.tiles == 1 else "s", w, h))
                self._show_metrics(metrics)
                self.status_var.set("Done - %s" % self.engine.description)
                self._refresh_previews()
            return apply
        self.run_async(work, "Deblurring with %s ..." % self.engine.model_name)

    def on_save(self):
        save_dir = resolve_path(self.cfg["output"]["save_dir"])
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        default = timestamped_name(self.source_name + "_deblurred")
        path = filedialog.asksaveasfilename(initialdir=save_dir, initialfile=default,
                                            defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        try:
            save_image(path, self.restored)
            msg = "Saved %s" % path
            if self.cfg["output"].get("save_comparison", True):
                base, ext = os.path.splitext(path)
                save_image(base + "_comparison" + ext, side_by_side(self.blurred, self.restored))
                msg += " (+ comparison)"
            self.status_var.set(msg)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    # ============================================================ real-time
    def on_realtime(self):
        if self.realtime_running:
            self.realtime_running = False     # worker notices and stops
            self.status_var.set("Stopping real-time mode...")
            return
        self.realtime_running = True
        self._set_buttons()
        self.lbl_left_title.configure(text="Live camera")
        self.lbl_right_title.configure(text="Live AI deblurred")
        t = threading.Thread(target=self._realtime_worker)
        t.daemon = True
        t.start()

    def _realtime_worker(self):
        err = None
        try:
            fps = FPSMeter()
            width = int(self.cfg["realtime"]["process_width"])
            with Camera(self.cfg["camera"]) as cam:
                last_id = None
                while self.realtime_running:
                    last_id, frame = cam.read(last_id=last_id)
                    small, result = process_frame(self.engine, frame, width)
                    fps.tick()
                    # keep the queue short: drop frames the UI has not shown yet
                    while self.events.qsize() > 2:
                        time.sleep(0.005)
                    self.events.put(("frame", (small, result, fps.fps, cam.description)))
        except Exception as exc:
            traceback.print_exc()
            err = exc
        self.events.put(("live_stopped", err))

    def _show_live(self, payload):
        small, result, fps, cam_desc = payload
        self.blurred, self.restored = small, result.image
        self.source_name = "live"
        self._refresh_previews()
        self.stats_var.set("FPS: %.1f  |  inference %.1f ms  |  total %.1f ms  |  %dx%d"
                           % (fps, result.inference_ms, result.total_ms, small.shape[1], small.shape[0]))
        self.status_var.set("Real-time: %s  |  %s" % (cam_desc, self.engine.description))

    def _live_stopped(self, err):
        self.realtime_running = False
        self.lbl_left_title.configure(text="Original (blurred)")
        self.lbl_right_title.configure(text="Deblurred (AI)")
        if err is not None:
            messagebox.showerror("Real-time mode stopped", error_text(err))
        else:
            self.status_var.set("Real-time mode stopped (last frame kept - you can save it)")
        if self.restored is not None:
            self._update_metrics()

    # ============================================================== display
    def _set_input(self, img, name):
        self.blurred, self.restored, self.ground_truth = img, None, None
        self.source_name = name
        self.stats_var.set("Processing time: -")
        self.metrics_var.set("Metrics: -  (press Deblur)")
        self._refresh_previews()

    def _update_metrics(self):
        if self.blurred is not None and self.restored is not None:
            self._show_metrics(evaluate(self.blurred, self.restored, self.ground_truth))

    def _show_metrics(self, m):
        self.metrics_var.set("\n".join(format_metrics(m)))

    def _to_photo(self, key, bgr):
        img = fit_within(bgr, self.pw, self.ph)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self._photos[key] = photo   # keep a reference or Tk drops the image
        return photo

    def _refresh_previews(self):
        view = self.view_var.get()
        if view == "Split slider" and self.restored is not None:
            self.split_scale.pack(fill="x", padx=16, before=self.info_frame)
        else:
            self.split_scale.pack_forget()
        if self.blurred is None:
            return
        self.lbl_left.configure(image=self._to_photo("left", self.blurred), text="",
                                width=0, height=0)
        if self.restored is None:
            self.lbl_right.configure(image="", text="Press Deblur", width=self.pw // 8, height=self.ph // 16)
            return
        if view == "Split slider":
            right = split_view(self.blurred, self.restored, self.split_var.get())
        elif view == "Difference map":
            right = difference_heatmap(self.blurred, self.restored)
        else:
            right = self.restored
        self.lbl_right.configure(image=self._to_photo("right", right), text="", width=0, height=0)

    def on_close(self):
        self.realtime_running = False
        time.sleep(0.1)
        if self.engine is not None:
            try:
                self.engine.close()
            except Exception:
                pass
        self.root.destroy()


def run_gui(cfg):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise DeblurError("Cannot open a window: %s" % exc,
                          hint="Run on the Jetson desktop, or 'export DISPLAY=:0' when using SSH. "
                               "Headless? use: python3 main.py image --input <file>")
    DeblurApp(root, cfg)
    root.mainloop()

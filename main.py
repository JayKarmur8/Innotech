#!/usr/bin/env python3
"""AI Image Deblurring System - command line entry point.

    python3 main.py                      # desktop GUI (default)
    python3 main.py gui
    python3 main.py image -i blurred.jpg [-g sharp.jpg] [-o out.png] [--show]
    python3 main.py camera               # real-time camera mode (OpenCV window)
    python3 main.py benchmark            # measure inference latency / FPS
    python3 main.py info                 # check GPU / TensorRT / camera setup

Common options: --config config.yaml  --backend auto|tensorrt|onnxruntime|opencv
"""

from __future__ import print_function

import argparse
import os
import sys
import time

from deblur.config import load_config, resolve_path
from deblur.errors import DeblurError


def build_parser():
    p = argparse.ArgumentParser(description="AI Image Deblurring System for NVIDIA Jetson Nano")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--backend", choices=["auto", "tensorrt", "onnxruntime", "opencv"],
                   help="override model.backend")
    p.add_argument("--precision", choices=["fp16", "fp32"], help="override model.precision")
    p.add_argument("--onnx", help="override model.onnx_path")
    p.add_argument("--engine", help="override model.engine_path")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("gui", help="desktop GUI (default)")

    im = sub.add_parser("image", help="deblur image file(s)")
    im.add_argument("-i", "--input", required=True, nargs="+", help="blurred image(s)")
    im.add_argument("-g", "--ground-truth", help="sharp ground-truth image for PSNR/SSIM")
    im.add_argument("-o", "--output", help="output file (single input) or folder")
    im.add_argument("--full-res", action="store_true", help="ignore max_input_side")
    im.add_argument("--show", action="store_true", help="show before/after window")

    cam = sub.add_parser("camera", help="real-time camera deblurring")
    cam.add_argument("--source", help="usb | csi | video file | URL")
    cam.add_argument("--device", type=int, help="USB camera index (/dev/videoN)")
    cam.add_argument("--width", type=int, help="processing width (smaller = faster)")

    b = sub.add_parser("benchmark", help="measure latency and FPS")
    b.add_argument("-n", "--runs", type=int, default=30)
    b.add_argument("--sizes", default="256x256,640x480,1280x720")

    sub.add_parser("info", help="print environment / GPU information")
    return p


def apply_overrides(cfg, args):
    for key, attr in (("backend", "backend"), ("precision", "precision"),
                      ("onnx_path", "onnx"), ("engine_path", "engine")):
        if getattr(args, attr, None):
            cfg["model"][key] = getattr(args, attr)
    if args.command == "camera":
        if args.source:
            cfg["camera"]["source"] = args.source
        if args.device is not None:
            cfg["camera"]["device_index"] = args.device
            cfg["camera"]["source"] = "usb"
        if args.width:
            cfg["realtime"]["process_width"] = args.width
    return cfg


# ------------------------------------------------------------------ commands
def cmd_image(cfg, args):
    import cv2
    from deblur.image_io import load_image, save_image, side_by_side
    from deblur.inference import DeblurEngine
    from deblur.metrics import evaluate, format_metrics

    # validate all inputs before the (slow) model load
    images = [(path, load_image(path)) for path in args.input]
    gt = load_image(args.ground_truth) if args.ground_truth else None
    engine = DeblurEngine(cfg)
    engine.warmup(1)
    save_dir = resolve_path(cfg["output"]["save_dir"])
    for path, img in images:
        res = engine.deblur(img, max_input_side=0 if args.full_res else None)
        stem = os.path.splitext(os.path.basename(path))[0]
        if args.output and len(args.input) == 1 and os.path.splitext(args.output)[1]:
            out_path = args.output
        else:
            out_path = os.path.join(args.output or save_dir, stem + "_deblurred.png")
        save_image(out_path, res.image)
        comp = side_by_side(img, res.image)
        if cfg["output"].get("save_comparison", True):
            save_image(os.path.splitext(out_path)[0] + "_comparison.png", comp)
        print("\n%s  (%dx%d)" % (path, img.shape[1], img.shape[0]))
        print("  output          : %s" % out_path)
        print("  processing time : %.1f ms (inference %.1f ms, %d tiles @ %dx%d)"
              % (res.total_ms, res.inference_ms, res.tiles, res.work_shape[1], res.work_shape[0]))
        for line in format_metrics(evaluate(img, res.image, gt)):
            print("  " + line)
        if args.show:
            from deblur.image_io import fit_within
            cv2.imshow("Before | After  (any key = next)", fit_within(comp, 1600, 900))
            cv2.waitKey(0)
    if args.show:
        cv2.destroyAllWindows()
    engine.close()


def cmd_camera(cfg, args):
    from deblur.inference import DeblurEngine
    from deblur.ui.realtime import run_realtime
    engine = DeblurEngine(cfg)
    try:
        run_realtime(engine, cfg)
    finally:
        engine.close()


def cmd_benchmark(cfg, args):
    import numpy as np
    from deblur.inference import DeblurEngine
    engine = DeblurEngine(cfg)
    engine.warmup(3)
    t = engine.tile_size
    x = np.random.rand(1, 3, t, t).astype(engine.backend.input_dtype)
    times = []
    for _ in range(args.runs):
        t0 = time.time()
        engine.backend.infer(x)
        times.append((time.time() - t0) * 1000)
    times.sort()
    print("\nBackend : %s" % engine.description)
    print("Tile %dx%d : mean %.1f ms | median %.1f ms | p90 %.1f ms | %.1f tiles/s"
          % (t, t, sum(times) / len(times), times[len(times) // 2],
             times[int(len(times) * 0.9) - 1], 1000.0 / (sum(times) / len(times))))
    for size in args.sizes.split(","):
        w, h = [int(v) for v in size.lower().split("x")]
        img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        runs = max(1, min(10, args.runs // 3))
        res = [engine.deblur(img, max_input_side=0) for _ in range(runs)][-1]
        print("Image %4dx%-4d: %7.1f ms total (%d tiles) -> %.2f FPS"
              % (w, h, res.total_ms, res.tiles, res.fps))
    engine.close()


def cmd_info(cfg, args):
    from deblur.camera import list_usb_cameras
    from deblur.model_loader import environment_report
    print("AI Image Deblurring System - environment check\n")
    for name, value in environment_report():
        print("  %-14s %s" % (name, value))
    print("  %-14s %s" % ("USB cameras", ", ".join("/dev/video%d" % i for i in list_usb_cameras()) or "none"))
    print("\nModel files:")
    for key in ("onnx_path", "engine_path"):
        path = resolve_path(cfg["model"][key])
        print("  %-12s %s  [%s]" % (key, path, "OK" if os.path.isfile(path) else "missing"))
    print("  backend = %s, precision = %s" % (cfg["model"]["backend"], cfg["model"]["precision"]))


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.command = args.command or "gui"
    try:
        cfg = apply_overrides(load_config(args.config), args)
        if args.command == "gui":
            from deblur.ui.gui import run_gui
            run_gui(cfg)
        elif args.command == "image":
            cmd_image(cfg, args)
        elif args.command == "camera":
            cmd_camera(cfg, args)
        elif args.command == "benchmark":
            cmd_benchmark(cfg, args)
        elif args.command == "info":
            cmd_info(cfg, args)
    except DeblurError as exc:
        print("\nERROR: " + exc.user_message(), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

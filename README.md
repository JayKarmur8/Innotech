# AI Image Deblurring System for NVIDIA Jetson Nano

Restores sharp images from blurred photos or live camera frames with a deep
network (**NAFNet**, trained on the GoPro motion-blur dataset), accelerated
with **TensorRT FP16** on the Jetson Nano GPU. OpenCV handles all image/camera I/O.

```
Blurred image / camera → OpenCV input → preprocessing (tiling, RGB, normalise)
  → NAFNet deblurring model → TensorRT FP16 (GPU) → post-processing (blend, resize back)
  → sharp image → PSNR / SSIM / sharpness / time / FPS → GUI, real-time view, save
```

## Features
- Tkinter GUI: Upload Image, Camera Capture, Deblur, Save Result, Real-time Mode, before/after
  previews, split-slider and difference-map comparison, processing time, PSNR/SSIM
- Real-time OpenCV window with FPS + inference time overlay
- Backends chosen automatically: **TensorRT** → ONNX Runtime (TensorRT/CUDA) → OpenCV DNN
- Tiled inference: any image size, output always keeps the **original dimensions**
- Friendly errors: missing model, bad image format, camera failure, out of GPU memory
- Headless CLI, benchmark, environment check, test-image generator, training script

## Project structure
```
main.py               entry point (gui | image | camera | benchmark | info)
config.yaml           all settings (model, tiles, camera, real-time, output)
deblur/
  config.py           config loading + validation
  errors.py           user-friendly exceptions
  model_loader.py     TensorRT / ONNX Runtime / OpenCV-DNN backends
  trt_builder.py      ONNX -> FP16 TensorRT engine
  preprocessing.py    resize, pad, tile, BGR->RGB, normalise, NCHW
  inference.py        DeblurEngine (tiles -> model -> blend), timing
  postprocessing.py   tensor -> image, tile blending, restore original size
  image_io.py         OpenCV load/save/resize/compare views/synthetic blur
  camera.py           USB (V4L2) + CSI (GStreamer) threaded capture
  metrics.py          PSNR, SSIM, sharpness, FPS meter
  ui/gui.py           Tkinter desktop app
  ui/realtime.py      OpenCV real-time viewer
tools/                download_weights, export_onnx, build_engine, train_lite,
                      make_test_images, nafnet_arch
tests/                unit + pipeline tests
```

## 1. Hardware / software
- Jetson Nano (4 GB recommended; 2 GB works with smaller settings), JetPack 4.6.x (Python 3.6,
  TensorRT 8.2, CUDA 10.2), 5 V 4 A power supply, fan, USB or CSI camera.
- Use the 10 W mode and max clocks for best speed:
  ```bash
  sudo nvpmodel -m 0 && sudo jetson_clocks
  ```

## 2. Install on Jetson Nano
```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv python3-numpy python3-yaml \
     python3-tk python3-pil python3-pil.imagetk python3-libnvinfer python3-dev
git clone <this repo> && cd Innotech
pip3 install pycuda          # needs nvcc on PATH:
#   export PATH=/usr/local/cuda/bin:$PATH ; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
**Do not `pip install opencv-python` on the Jetson** – it has no GStreamer/CSI support;
use the JetPack one (`python3-opencv`). If you use a virtualenv, create it with
`--system-site-packages`.

Add swap if you have a 2/4 GB board (engine building needs memory):
```bash
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

Verify: `python3 main.py info`

## 3. Model setup
The model is **NAFNet-GoPro-width32** (official MIT-licensed weights, 17 M parameters,
32.9 dB PSNR on GoPro). Steps (PyTorch is only needed to *export*, so do steps a–b on a
PC / Colab – PyTorch is not required on the Nano):

```bash
# a) download weights (needs: pip3 install gdown; or download manually, link printed on failure)
python3 tools/download_weights.py
# b) export to ONNX (fixed 256x256 tile, opset 11)      [PC: pip install torch onnx onnxruntime]
python3 tools/export_onnx.py
# copy models/nafnet_gopro_w32_256.onnx to the Jetson's models/ folder
# c) ON THE JETSON: build the FP16 TensorRT engine (5-15 min, once)
python3 tools/build_engine.py
```
Step (c) also happens automatically on first start if only the ONNX file exists
(`build_engine_if_missing: true`). Engines are device/TensorRT-version specific – always
build them on the Nano.

### Faster lightweight model (optional)
`nafnet32` is high quality but heavy for a Nano. For real-time use train the ~2 M-parameter
`lite` preset (on a PC/Colab GPU) and export it:
```bash
python3 tools/train_lite.py --blur-dir GoPro/train/input --sharp-dir GoPro/train/target
#   or, only sharp photos:  --sharp-dir photos/ --synthetic
python3 tools/export_onnx.py --preset lite --weights models/nafnet_lite.pth --output models/nafnet_lite_256.onnx
```
Then set `onnx_path` / `engine_path` in `config.yaml` (e.g. `models/nafnet_lite_256.onnx`,
`models/nafnet_lite_256_fp16.engine`) and rebuild the engine. You can also fine-tune the
official model on your own camera with `--preset nafnet32 --init <weights>`.

## 4. Run
```bash
python3 main.py                         # GUI
python3 main.py image -i samples/demo_blur_motion.png -g samples/demo_sharp.png --show
python3 main.py camera                  # real-time (q quit, s save, v change view, +/- resolution)
python3 main.py camera --source csi --width 320
python3 main.py benchmark               # latency + FPS
python3 main.py info                    # environment check
```
Over SSH for the GUI: `export DISPLAY=:0` (desktop logged in). Headless: use `image`.

Make test images (blurred + sharp ground truth, for PSNR/SSIM):
`python3 tools/make_test_images.py --input my_photo.jpg`. Real-photo ground truth: use the
GoPro test set (`blur` / `sharp` pairs).

**GUI workflow:** Upload Image (or Camera Capture) → *(optional)* Load Ground Truth →
Deblur → compare (side by side / split slider / difference map) → Save Result.

## 5. Camera
- **USB:** `ls /dev/video*`; set `camera.source: usb`, `device_index: N` (or `--device N`).
  Test: `v4l2-ctl --list-formats-ext`.
- **CSI (IMX219 etc.):** `camera.source: csi`, `csi_sensor_id: 0`. Test with
  `nvgstcapture-1.0`. If the image is upside down set `flip_method: 2`.
- Video file / RTSP: `--source path/or/url` (files loop, handy for demos without a camera).
- Frames are grabbed on a background thread and only the newest frame is processed
  (no growing lag).

## 6. GPU / TensorRT acceleration
1. `backend: auto` (default) uses the TensorRT engine when it exists.
2. `precision: fp16` builds a half-precision engine (about 2x faster, half memory).
   LayerNorm layers stay FP32 to avoid FP16 overflow artefacts.
3. Fallbacks: `onnxruntime-gpu` (install the Jetson wheel from
   https://elinux.org/Jetson_Zoo) or OpenCV DNN CUDA (needs OpenCV built with CUDA).
4. Confirm with `python3 main.py info` / the status line in the GUI (`TensorRT ... (GPU)`).

Tuning speed vs. quality (`config.yaml`):

| Setting | Effect |
|---|---|
| `model` preset `lite` | much faster, slightly lower quality |
| `processing.max_input_side` | downscale big photos before inference (result is resized back to the original size) |
| `processing.tile_overlap` | 0 = fastest, 32 = no visible seams |
| `realtime.process_width` | 256-320 for live video |

Real-time on the Nano needs the `lite` model or a small `process_width`; the full width32
model is meant for high-quality still images.

## 7. Troubleshooting
| Problem | Fix |
|---|---|
| `No model found` | run steps in *Model setup* |
| `TensorRT backend unavailable` | `sudo apt install python3-libnvinfer`; `pip3 install pycuda` (nvcc on PATH); virtualenv needs `--system-site-packages` |
| Engine fails to load / version mismatch | rebuild on this device: `python3 tools/build_engine.py` |
| Out of memory | use FP16, `--workspace 512`, add swap, smaller `max_input_side`, close browsers, `sudo systemctl set-default multi-user.target` (no desktop) for headless runs |
| Black/NaN patches | FP16 overflow: keep default FP32 LayerNorm or use `--fp32` |
| `Could not open USB camera` | check `ls /dev/video*`, permissions (`sudo usermod -aG video $USER`), other apps using it |
| CSI camera fails | `sudo systemctl restart nvargus-daemon`; check ribbon cable; OpenCV must be the JetPack build |
| GUI does not open | `export DISPLAY=:0`, install `python3-tk python3-pil.imagetk` |
| Slow | `sudo nvpmodel -m 0 && sudo jetson_clocks`, check backend is GPU, lower `process_width` |
| Result too sharp/noisy on non-blurred images | the model assumes motion blur; it does not fix defocus as well |

## Tests
`python3 -m pytest tests/` – checks tiling/blending (identity model must reproduce input
exactly), metrics, config and error handling; no GPU or weights needed.

## Credits
NAFNet: Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022
(https://github.com/megvii-research/NAFNet, MIT license). Pretrained on the GoPro dataset.

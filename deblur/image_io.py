"""Image processing helpers built on OpenCV.

Loading/validating/saving images, resizing, building before/after comparison
views and generating synthetic motion blur (for demos and PSNR/SSIM tests).
"""

import os
import time

import cv2
import numpy as np

from .errors import UnsupportedImageError

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------------------------------------------- I/O
def is_supported(path):
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def load_image(path):
    """Load an image as an 8-bit BGR array (H, W, 3).

    Handles grayscale, alpha channels and 16-bit images. Raises
    ``UnsupportedImageError`` with a clear message on any problem.
    """
    if not path or not os.path.isfile(path):
        raise UnsupportedImageError("Image file not found: %s" % path)
    if not is_supported(path):
        raise UnsupportedImageError(
            "Unsupported image format '%s' (%s)" % (os.path.splitext(path)[1], os.path.basename(path)))
    # np.fromfile + imdecode also works with non-ASCII paths.
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise UnsupportedImageError("OpenCV could not decode '%s' (corrupted or not an image)" % path)
    return to_bgr8(img)


def to_bgr8(img):
    """Convert any OpenCV image (gray, BGRA, 16-bit, float) to uint8 BGR."""
    if img.dtype == np.uint16:
        img = (img / 257.0).round().astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255.0, 0, 255).round().astype(np.uint8)
    elif img.dtype != np.uint8:
        raise UnsupportedImageError("Unsupported pixel type %s" % img.dtype)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] != 3:
        raise UnsupportedImageError("Unsupported number of channels: %d" % img.shape[2])
    return np.ascontiguousarray(img)


def save_image(path, img):
    """Save an image, creating the directory. Returns the path."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        path += ".png"
        ext = ".png"
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError("Could not encode image as %s" % ext)
    buf.tofile(path)
    return path


def timestamped_name(prefix="deblurred", ext=".png"):
    return "%s_%s%s" % (prefix, time.strftime("%Y%m%d_%H%M%S"), ext)


# ------------------------------------------------------------------- resizing
def resize_max_side(img, max_side):
    """Downscale so the longest side is <= max_side (never upscales)."""
    h, w = img.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return img
    scale = float(max_side) / max(h, w)
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


def resize_to_width(img, width):
    h, w = img.shape[:2]
    if w == width:
        return img
    height = max(1, int(round(h * float(width) / w)))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return cv2.resize(img, (width, height), interpolation=interp)


def fit_within(img, max_w, max_h):
    """Resize (up or down) so the image fits inside max_w x max_h."""
    h, w = img.shape[:2]
    scale = min(float(max_w) / w, float(max_h) / h)
    if abs(scale - 1.0) < 1e-3:
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interp)


def match_size(img, reference):
    """Resize ``img`` to the size of ``reference`` if they differ."""
    h, w = reference.shape[:2]
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)


# ------------------------------------------------------------- visualisation
def put_label(img, text, org=(10, 28), scale=0.7, color=(255, 255, 255)):
    """Draw readable text with a dark outline."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return img


def side_by_side(original, restored, labels=("Blurred input", "AI deblurred"), gap=6):
    restored = match_size(restored, original)
    a = put_label(original.copy(), labels[0])
    b = put_label(restored.copy(), labels[1], color=(120, 255, 120))
    sep = np.full((original.shape[0], gap, 3), 255, np.uint8)
    return np.hstack([a, sep, b])


def split_view(original, restored, position=0.5):
    """Left part = original, right part = restored, with a divider line."""
    restored = match_size(restored, original)
    h, w = original.shape[:2]
    x = int(np.clip(position, 0.0, 1.0) * w)
    out = restored.copy()
    out[:, :x] = original[:, :x]
    cv2.line(out, (x, 0), (x, h - 1), (0, 255, 255), 2)
    put_label(out, "Before", (10, 28))
    put_label(out, "After", (max(10, w - 90), 28), color=(120, 255, 120))
    return out


def difference_heatmap(original, restored, gain=4.0):
    """Colour map of |restored - original| (shows where the model changed detail)."""
    restored = match_size(restored, original)
    diff = cv2.absdiff(cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
    diff = np.clip(diff.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return put_label(heat, "Difference x%g" % gain)


# ------------------------------------------------------------ synthetic blur
def motion_blur_kernel(length=15, angle=0.0):
    """Linear motion-blur PSF of the given length (px) and angle (degrees)."""
    length = max(1, int(length))
    size = length if length % 2 == 1 else length + 1
    k = np.zeros((size, size), np.float32)
    k[size // 2, (size - length) // 2:(size - length) // 2 + length] = 1.0
    rot = cv2.getRotationMatrix2D((size / 2.0 - 0.5, size / 2.0 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, rot, (size, size), flags=cv2.INTER_LINEAR)
    s = k.sum()
    return k / s if s > 0 else k


def random_motion_kernel(max_len=21, rng=None):
    """Random non-linear camera-shake kernel (random walk trajectory)."""
    rng = rng or np.random
    size = max_len if max_len % 2 == 1 else max_len + 1
    steps = rng.randint(max(3, max_len // 3), max_len * 2)
    pos = np.array([0.0, 0.0])
    vel = rng.randn(2)
    pts = []
    for _ in range(steps):
        vel = 0.8 * vel + 0.4 * rng.randn(2)
        pos = pos + vel / max(1e-6, np.linalg.norm(vel)) * 0.7
        pts.append(pos.copy())
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    span = np.abs(pts).max()
    if span > size / 2.0 - 1:
        pts *= (size / 2.0 - 1) / span
    k = np.zeros((size, size), np.float32)
    for x, y in pts:
        cx, cy = int(round(x + size // 2)), int(round(y + size // 2))
        k[cy, cx] += 1.0
    k = cv2.GaussianBlur(k, (3, 3), 0)
    return k / k.sum()


def apply_blur(img, kernel, noise_sigma=0.0, rng=None):
    out = cv2.filter2D(img.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT)
    if noise_sigma > 0:
        rng = rng or np.random
        out += rng.randn(*out.shape).astype(np.float32) * noise_sigma
    return np.clip(out, 0, 255).round().astype(np.uint8)

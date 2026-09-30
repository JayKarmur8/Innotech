"""Image-quality and performance metrics.

* PSNR / SSIM  - full-reference, need the sharp ground-truth image
* sharpness    - no-reference (variance of the Laplacian), always available,
                 higher = more high-frequency detail
* FPSMeter     - smoothed frames-per-second for the real-time mode
"""

import collections
import time

import cv2
import numpy as np

from .image_io import match_size


def psnr(img, ref, max_val=255.0):
    """Peak signal-to-noise ratio in dB (higher is better)."""
    img = match_size(img, ref).astype(np.float64)
    mse = np.mean((img - ref.astype(np.float64)) ** 2)
    if mse <= 1e-10:
        return float("inf")
    return float(10.0 * np.log10(max_val ** 2 / mse))


def _ssim_channel(a, b):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ksize, sigma = (11, 11), 1.5
    mu_a = cv2.GaussianBlur(a, ksize, sigma)
    mu_b = cv2.GaussianBlur(b, ksize, sigma)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    s_a = cv2.GaussianBlur(a * a, ksize, sigma) - mu_a2
    s_b = cv2.GaussianBlur(b * b, ksize, sigma) - mu_b2
    s_ab = cv2.GaussianBlur(a * b, ksize, sigma) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * s_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (s_a + s_b + c2))
    return float(ssim_map[5:-5, 5:-5].mean()) if min(a.shape) > 10 else float(ssim_map.mean())


def ssim(img, ref):
    """Structural similarity (Wang et al. 2004), averaged over BGR channels (0..1)."""
    img = match_size(img, ref).astype(np.float64)
    ref = ref.astype(np.float64)
    return float(np.mean([_ssim_channel(img[:, :, c], ref[:, :, c]) for c in range(3)]))


def sharpness(img):
    """Variance of the Laplacian - a simple no-reference sharpness score."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate(blurred, restored, ground_truth=None):
    """Return a dict with all metrics that can be computed."""
    res = {
        "sharpness_in": sharpness(blurred),
        "sharpness_out": sharpness(restored),
    }
    res["sharpness_gain"] = res["sharpness_out"] / max(res["sharpness_in"], 1e-6)
    if ground_truth is not None:
        gt = match_size(ground_truth, blurred)
        res["psnr_in"] = psnr(blurred, gt)
        res["psnr_out"] = psnr(restored, gt)
        res["ssim_in"] = ssim(blurred, gt)
        res["ssim_out"] = ssim(restored, gt)
    return res


def format_metrics(m):
    lines = ["Sharpness (Laplacian var): %.1f -> %.1f  (x%.2f)"
             % (m["sharpness_in"], m["sharpness_out"], m["sharpness_gain"])]
    if "psnr_out" in m:
        lines.append("PSNR: %.2f dB -> %.2f dB  (%+.2f dB)"
                     % (m["psnr_in"], m["psnr_out"], m["psnr_out"] - m["psnr_in"]))
        lines.append("SSIM: %.4f -> %.4f  (%+.4f)"
                     % (m["ssim_in"], m["ssim_out"], m["ssim_out"] - m["ssim_in"]))
    else:
        lines.append("PSNR/SSIM: load a sharp ground-truth image to compute")
    return lines


class FPSMeter(object):
    """Moving-average FPS over the last ``window`` frames."""

    def __init__(self, window=30):
        self.times = collections.deque(maxlen=window)

    def tick(self):
        self.times.append(time.time())

    @property
    def fps(self):
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0

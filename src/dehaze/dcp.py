"""Dark Channel Prior dehazing (He, Sun & Tang, CVPR 2009) with guided-filter refinement.

Used as the classical, non-learned baseline.
"""

import cv2
import numpy as np


def dark_channel(img, patch=15):
    """Per-pixel minimum over RGB, then a min-filter over a patch (erosion)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(img.min(axis=2), kernel)


def estimate_airlight(img, dark, top=0.001):
    n = max(1, int(dark.size * top))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    return img.reshape(-1, 3)[idx].mean(axis=0)


def guided_filter(guide, src, radius, eps):
    box = lambda x: cv2.boxFilter(x, -1, (radius, radius))
    mean_i, mean_p = box(guide), box(src)
    var_i = box(guide * guide) - mean_i ** 2
    cov_ip = box(guide * src) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def dehaze_dcp(hazy, patch=15, omega=0.95, t_min=0.1, radius=40, eps=1e-3):
    """hazy: float32 RGB in [0, 1]. Returns the dehazed image in [0, 1]."""
    hazy = hazy.astype(np.float32)
    airlight = estimate_airlight(hazy, dark_channel(hazy, patch))
    t = 1 - omega * dark_channel(hazy / np.maximum(airlight, 1e-6), patch)
    t = guided_filter(cv2.cvtColor(hazy, cv2.COLOR_RGB2GRAY), t.astype(np.float32), radius, eps)
    t = np.clip(t, t_min, 1)[..., None]
    return np.clip((hazy - airlight) / t + airlight, 0, 1)

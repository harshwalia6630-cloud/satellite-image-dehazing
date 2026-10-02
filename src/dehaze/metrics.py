"""Image quality metrics. All inputs are float RGB arrays in [0, 1]."""

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .dcp import dark_channel


def psnr(pred, target):
    return float(peak_signal_noise_ratio(target, pred, data_range=1.0))


def ssim(pred, target):
    return float(structural_similarity(target, pred, channel_axis=2, data_range=1.0))


def ciede2000(pred, target):
    """Mean perceptual colour difference (lower is better; < 2.3 is roughly a 'just noticeable' difference)."""
    return float(deltaE_ciede2000(rgb2lab(target), rgb2lab(pred)).mean())


def residual_haze(pred, target, patch=15):
    """Haze left in (or wrongly removed from) the output.

    Haze raises the dark channel (He et al.): in clean outdoor imagery it is
    close to zero, and every bit of haze lifts it. Comparing the output's dark
    channel to the ground truth's measures how much haze remains, independent
    of small colour or texture errors. Lower is better.
    """
    return float(np.abs(dark_channel(pred.astype(np.float32), patch)
                        - dark_channel(target.astype(np.float32), patch)).mean())


ALL = {"psnr": psnr, "ssim": ssim, "ciede2000": ciede2000, "residual_haze": residual_haze}

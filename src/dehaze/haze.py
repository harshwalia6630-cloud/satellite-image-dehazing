"""Physically based haze synthesis for remote-sensing images.

Uses the atmospheric scattering model (Koschmieder / McCartney):

    I(x) = J(x) * t(x) + A * (1 - t(x)),   t(x) = exp(-beta * d(x))

J is the clean image, A the atmospheric light, t the transmission. Satellite
haze is not uniform, so the optical depth d(x) is a smooth fractal noise
field (sum of upsampled random grids at several scales) rather than a
constant. This is the same recipe used to build synthetic remote-sensing
dehazing benchmarks such as SateHaze1k and RS-Haze.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# haze levels used for the fixed validation / test sets
LEVELS = {
    "thin": (0.35, 0.70),
    "moderate": (0.70, 1.10),
    "thick": (1.10, 1.60),
}
TRAIN_BETA = (0.25, 1.75)


@dataclass
class HazeParams:
    beta: float
    airlight: np.ndarray          # (3,) RGB in [0, 1]
    density: np.ndarray           # (H, W) optical depth field in [0.4, 1.0]


def fractal_field(h, w, rng, octaves=(2, 4, 8, 16), persistence=0.55):
    """Smooth random field in [0, 1]: coarse blobs plus progressively finer, weaker detail."""
    field = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for cells in octaves:
        grid = rng.random((cells + 1, cells + 1)).astype(np.float32)
        field += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= persistence
    field /= total
    field -= field.min()
    return field / max(field.max(), 1e-6)


def sample_params(h, w, rng, beta_range):
    beta = rng.uniform(*beta_range)
    grey = rng.uniform(0.72, 0.95)
    tint = rng.normal(0, 0.025, 3)                      # slight blue/yellow cast
    airlight = np.clip(grey + tint + np.array([0, 0.005, 0.02]), 0.6, 1.0).astype(np.float32)
    density = 0.4 + 0.6 * fractal_field(h, w, rng)      # never completely haze-free
    return HazeParams(beta, airlight, density)


def apply_haze(clean, params):
    """clean: float32 RGB in [0, 1], shape (H, W, 3). Returns (hazy, transmission)."""
    t = np.exp(-params.beta * params.density)[..., None]
    t = np.clip(t, 0.05, 1.0)
    hazy = clean * t + params.airlight[None, None, :] * (1 - t)
    return np.clip(hazy, 0, 1).astype(np.float32), t[..., 0].astype(np.float32)


def add_haze(clean, rng, beta_range=TRAIN_BETA):
    h, w = clean.shape[:2]
    return apply_haze(clean, sample_params(h, w, rng, beta_range))

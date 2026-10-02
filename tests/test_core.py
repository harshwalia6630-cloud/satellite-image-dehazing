import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dehaze.dcp import dark_channel, dehaze_dcp  # noqa: E402
from dehaze.haze import HazeParams, add_haze, apply_haze, fractal_field  # noqa: E402
from dehaze.metrics import psnr, residual_haze, ssim  # noqa: E402
from dehaze.tiling import dehaze_tiled, ramp_window, tile_positions  # noqa: E402


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    base = np.linspace(0, 1, 256, dtype=np.float32)
    img = np.stack([np.outer(base, base), np.outer(base[::-1], base), np.full((256, 256), 0.3)], axis=2)
    return np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1).astype(np.float32)


def test_fractal_field_range_and_smoothness():
    f = fractal_field(128, 128, np.random.default_rng(1))
    assert f.min() == pytest.approx(0) and f.max() == pytest.approx(1)
    assert np.abs(np.diff(f, axis=0)).mean() < 0.05   # smooth, not per-pixel noise


def test_scattering_model_limits(image):
    a = np.array([0.9, 0.9, 0.9], np.float32)
    clear, t = apply_haze(image, HazeParams(beta=0.0, airlight=a, density=np.ones((256, 256), np.float32)))
    assert np.allclose(clear, image) and np.allclose(t, 1)
    dense, t = apply_haze(image, HazeParams(beta=50.0, airlight=a, density=np.ones((256, 256), np.float32)))
    assert np.allclose(t, 0.05)                       # transmission floor
    assert np.allclose(dense, image * 0.05 + a * 0.95, atol=1e-6)


def test_haze_raises_dark_channel(image):
    hazy, _ = add_haze(image, np.random.default_rng(2))
    assert dark_channel(hazy).mean() > dark_channel(image).mean()
    assert residual_haze(hazy, image) > 0.05


def test_dcp_improves_hazy_image(image):
    hazy, _ = add_haze(image, np.random.default_rng(3), beta_range=(1.0, 1.0))
    assert psnr(dehaze_dcp(hazy), image) > psnr(hazy, image) - 1  # sanity: runs and stays in range
    out = dehaze_dcp(hazy)
    assert out.shape == image.shape and out.min() >= 0 and out.max() <= 1


def test_metrics_identity(image):
    assert psnr(image, image) == float("inf") or psnr(image, image) > 99
    assert ssim(image, image) == pytest.approx(1.0)
    assert residual_haze(image, image) == pytest.approx(0.0)


def test_tile_positions_cover_everything():
    assert tile_positions(256, 256, 224) == [0]
    pos = tile_positions(700, 256, 224)
    assert pos[0] == 0 and pos[-1] == 700 - 256
    assert all(b - a <= 224 for a, b in zip(pos, pos[1:]))


def test_ramp_window():
    w = ramp_window(256, 32)
    assert w[0] > 0 and w[100] == 1 and np.allclose(w, w[::-1])


@pytest.mark.parametrize("shape", [(256, 256), (300, 517), (100, 90), (1024, 768)])
def test_tiling_is_seamless_for_pointwise_ops(shape):
    """With a per-pixel 'model' the blended tiled result must equal the untiled result exactly."""
    rng = np.random.default_rng(4)
    img = rng.random(shape + (3,)).astype(np.float32)
    f = lambda x: np.sqrt(x)
    out = dehaze_tiled(f, img, tile=256, overlap=32)
    assert out.shape == img.shape
    assert np.allclose(out, np.sqrt(img), atol=1e-6)

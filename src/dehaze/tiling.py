"""Patch-based inference for images of any size.

The image is split into overlapping tiles, each tile is dehazed independently,
and the outputs are blended with a smooth (raised-cosine) weight so tile
borders don't leave seams. Memory use is bounded by the tile size, so very
large satellite scenes can be processed on a laptop or in a browser.
web/js/dehaze.js implements the same algorithm.
"""

import numpy as np


def ramp_window(size, overlap):
    """1-D weight: raised-cosine ramps of length `overlap` at both ends, 1 in the middle."""
    w = np.ones(size, np.float32)
    if 0 < overlap < size:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap + 2)[1:-1])).astype(np.float32)
        w[:overlap], w[-overlap:] = ramp, ramp[::-1]
    return w


def tile_positions(length, tile, stride):
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile, stride))
    return pos + [length - tile]


def dehaze_tiled(predict, image, tile=256, overlap=32):
    """predict: f(batch NHWC float32 [0,1]) -> batch. image: HxWx3 float32 [0,1]."""
    h, w = image.shape[:2]
    # reflect-pad to at least one tile and to a multiple of 8 (three stride-2 stages)
    ph, pw = max(h, tile), max(w, tile)
    ph, pw = ph + (-ph) % 8, pw + (-pw) % 8
    padded = np.pad(image, ((0, ph - h), (0, pw - w), (0, 0)), mode="reflect")
    out = np.zeros_like(padded)
    weight = np.zeros(padded.shape[:2] + (1,), np.float32)
    th, tw = min(tile, ph), min(tile, pw)
    win = np.outer(ramp_window(th, overlap), ramp_window(tw, overlap))[..., None]
    for y in tile_positions(ph, th, th - overlap):
        for x in tile_positions(pw, tw, tw - overlap):
            pred = predict(padded[None, y:y + th, x:x + tw])[0]
            out[y:y + th, x:x + tw] += pred * win
            weight[y:y + th, x:x + tw] += win
    return np.clip(out / np.maximum(weight, 1e-6), 0, 1)[:h, :w]

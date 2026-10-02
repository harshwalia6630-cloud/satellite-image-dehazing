"""Evaluate on the fixed test set (1,100 images x 3 haze levels = 3,300 pairs).

Compares three methods on identical inputs:
    hazy input       no processing (lower bound)
    dark channel     He et al. 2009 + guided filter (classical baseline)
    ours             encoder-decoder network, patch-based (tiled) inference

Writes reports/metrics.json, reports/per_image.csv and reports/comparison.png.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dehaze import env  # noqa: E402

tf = env.setup()
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dehaze.dcp import dehaze_dcp  # noqa: E402
from dehaze.metrics import ALL  # noqa: E402
from dehaze.model import build_model  # noqa: E402
from dehaze.tiling import dehaze_tiled  # noqa: E402

METHODS = ("hazy input", "dark channel prior", "ours")


def load_model():
    model = build_model(tf)
    model.load_weights(str(ROOT / "models/dehaze_unet.weights.h5"))
    fn = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    return lambda batch: np.asarray(fn(tf.constant(batch, tf.float32)))


def main():
    test = np.load(ROOT / "data/processed/test.npz")
    clean = test["clean"].astype(np.float32) / 255
    hazy = test["hazy"].astype(np.float32) / 255
    predict = load_model()
    predict(hazy[:1])  # trace / warm up

    rows, times = [], {m: [] for m in METHODS}
    examples = {}
    for i in tqdm(range(len(hazy)), desc="test"):
        outs = {}
        t0 = time.perf_counter(); outs["hazy input"] = hazy[i]; times["hazy input"].append(0.0)
        t0 = time.perf_counter(); outs["dark channel prior"] = dehaze_dcp(hazy[i]); times["dark channel prior"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); outs["ours"] = dehaze_tiled(predict, hazy[i]); times["ours"].append(time.perf_counter() - t0)
        for m, out in outs.items():
            rows.append({"index": i, "level": test["levels"][i], "beta": float(test["betas"][i]), "method": m,
                         **{k: f(out, clean[i]) for k, f in ALL.items()}})
        if i in (3 * 41 + 2, 3 * 333 + 1, 3 * 640 + 2, 3 * 975 + 2):
            examples[i] = outs

    df = pd.DataFrame(rows)
    (ROOT / "reports").mkdir(exist_ok=True)
    df.to_csv(ROOT / "reports/per_image.csv", index=False)

    summary = {}
    for level in ("all", "thin", "moderate", "thick"):
        sub = df if level == "all" else df[df.level == level]
        summary[level] = {m: {k: round(float(sub[sub.method == m][k].mean()), 4) for k in ALL} for m in METHODS}

    def gain(metric, base, higher_is_better=True):
        b, o = summary["all"][base][metric], summary["all"]["ours"][metric]
        return round(100 * ((o - b) / b if higher_is_better else (b - o) / b), 1)

    report = {
        "test_pairs": int(len(hazy)),
        "summary": summary,
        "psnr_gain_vs_dcp_pct": gain("psnr", "dark channel prior"),
        "psnr_gain_vs_hazy_pct": gain("psnr", "hazy input"),
        "residual_haze_reduction_vs_dcp_pct": gain("residual_haze", "dark channel prior", False),
        "ciede2000_reduction_vs_dcp_pct": gain("ciede2000", "dark channel prior", False),
        "ours_beats_dcp_psnr_share": round(float(
            (df[df.method == "ours"].psnr.values > df[df.method == "dark channel prior"].psnr.values).mean()), 4),
        "seconds_per_256px_image": {m: round(float(np.mean(t)), 4) for m, t in times.items() if m != "hazy input"},
    }
    (ROOT / "reports/metrics.json").write_text(json.dumps(report, indent=2))

    # qualitative figure
    fig, axes = plt.subplots(len(examples), 4, figsize=(12, 3.1 * len(examples)))
    for r, (i, outs) in enumerate(examples.items()):
        for c, (title, img) in enumerate([("Hazy input", outs["hazy input"]), ("Dark channel prior", outs["dark channel prior"]),
                                          ("Ours", outs["ours"]), ("Ground truth", clean[i])]):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            if c < 3:
                ax.set_xlabel(f"PSNR {ALL['psnr'](img, clean[i]):.1f} dB", fontsize=9)
            if r == 0:
                ax.set_title(title)
        axes[r, 0].set_ylabel(f"{test['levels'][i]} haze", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/comparison.png", dpi=110)

    print(json.dumps({k: v for k, v in report.items() if k != "summary"}, indent=2))
    print(pd.DataFrame({m: summary["all"][m] for m in METHODS}).T.to_string())


if __name__ == "__main__":
    main()

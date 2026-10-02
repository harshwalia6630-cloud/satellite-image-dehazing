"""Plot loss and PSNR curves from reports/training_log.csv."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    log = pd.read_csv(ROOT / "reports/training_log.csv")
    epochs = log["epoch"] + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, log["loss"], label="train (random patches, fresh haze)")
    axes[0].plot(epochs, log["val_loss"], label="validation (fixed haze)")
    axes[0].set(title="Loss (L1 + 0.2·(1−SSIM))", xlabel="epoch")
    axes[1].plot(epochs, log["psnr_metric"], label="train")
    axes[1].plot(epochs, log["val_psnr_metric"], label="validation")
    axes[1].set(title="PSNR (dB)", xlabel="epoch")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/training_curves.png", dpi=120)


if __name__ == "__main__":
    main()

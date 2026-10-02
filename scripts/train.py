"""Train the encoder-decoder dehazing model on random patches.

Each epoch every training image gets *new* synthetic haze (random density,
airlight and spatial pattern), then a random 192x192 patch with a random
flip/rotation is cut out. The model never sees the same hazy input twice,
which is what lets a modest network generalise across haze levels.

Loss = L1 + 0.2 * (1 - SSIM). L1 keeps colours accurate; SSIM keeps edges and
texture sharp. The checkpoint with the best validation PSNR is kept.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dehaze import env  # noqa: E402

tf = env.setup()
from dehaze.haze import add_haze  # noqa: E402
from dehaze.model import build_model  # noqa: E402


def train_dataset(clean, patch, batch, seed):
    rng = np.random.default_rng(seed)

    def make_pair(i):
        img = clean[int(i)].astype(np.float32) / 255
        hazy, _ = add_haze(img, rng)
        y, x = rng.integers(0, img.shape[0] - patch + 1, 2)
        pair = np.concatenate([hazy, img], axis=2)[y:y + patch, x:x + patch]
        pair = np.rot90(pair, rng.integers(4))
        if rng.random() < 0.5:
            pair = pair[:, ::-1]
        return np.ascontiguousarray(pair, dtype=np.float32)

    ds = tf.data.Dataset.range(len(clean)).shuffle(len(clean), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda i: tf.numpy_function(make_pair, [i], tf.float32), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda p: (tf.ensure_shape(p[..., :3], (patch, patch, 3)), tf.ensure_shape(p[..., 3:], (patch, patch, 3))))
    return ds.batch(batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def val_dataset(val, batch):
    x, y = val["hazy"].astype(np.float32) / 255, val["clean"].astype(np.float32) / 255
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch).prefetch(tf.data.AUTOTUNE)


def loss_fn(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = tf.reduce_mean(tf.image.ssim(y_true, tf.clip_by_value(y_pred, 0, 1), max_val=1.0))
    return l1 + 0.2 * (1 - ssim)


def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, tf.clip_by_value(tf.cast(y_pred, tf.float32), 0, 1), max_val=1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("GPUs:", tf.config.list_physical_devices("GPU") or "none")

    train = np.load(ROOT / "data/processed/train.npz")["clean"]
    val = np.load(ROOT / "data/processed/val.npz")
    print(f"train images: {len(train)} | val pairs: {len(val['hazy'])}")

    model = build_model(tf)
    print(f"parameters: {model.count_params():,}")
    steps = len(train) // args.batch_size
    schedule = tf.keras.optimizers.schedules.CosineDecay(args.lr, steps * args.epochs, alpha=0.05)
    model.compile(optimizer=tf.keras.optimizers.Adam(schedule), loss=loss_fn, metrics=[psnr_metric])

    out = ROOT / "models/dehaze_unet.weights.h5"
    out.parent.mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    history = model.fit(
        train_dataset(train, args.patch, args.batch_size, args.seed),
        validation_data=val_dataset(val, args.batch_size),
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(out), monitor="val_psnr_metric", mode="max",
                                               save_best_only=True, save_weights_only=True, verbose=1),
            tf.keras.callbacks.CSVLogger(str(ROOT / "reports/training_log.csv")),
        ],
    )
    best = max(history.history["val_psnr_metric"])
    (ROOT / "reports/train_summary.json").write_text(json.dumps(
        {"parameters": int(model.count_params()), "epochs": args.epochs, "best_val_psnr": round(float(best), 3),
         "patch": args.patch, "batch_size": args.batch_size}, indent=2))
    print(f"best val PSNR {best:.2f} dB -> {out}")


if __name__ == "__main__":
    main()

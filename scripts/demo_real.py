"""Before/after figure on real hazy NASA MODIS imagery (no ground truth exists for these).

Writes docs/real_world.jpg. The sample images are public domain (NASA EO / GSFC).
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dehaze import env  # noqa: E402

tf = env.setup()
from dehaze.dcp import dark_channel  # noqa: E402
from dehaze.model import build_model  # noqa: E402
from dehaze.tiling import dehaze_tiled  # noqa: E402


def main(names=("river_of_haze_india.jpg", "seasonal_india_2024.jpg", "po_valley.jpg"), width=520):
    model = build_model(tf)
    model.load_weights(str(ROOT / "models/dehaze_unet.weights.h5"))
    fn = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    predict = lambda b: np.asarray(fn(tf.constant(b, tf.float32)))
    rows, report = [], {}
    for name in names:
        img = np.asarray(Image.open(ROOT / "web/samples" / name).convert("RGB"), np.float32) / 255
        out = dehaze_tiled(predict, img)
        report[name] = {"dark_channel_before": round(float(dark_channel(img).mean()), 4),
                        "dark_channel_after": round(float(dark_channel(out).mean()), 4)}
        pair = []
        for arr, label in ((img, "Hazy (NASA MODIS)"), (out, "Dehazed")):
            im = Image.fromarray((arr * 255).round().astype(np.uint8))
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
            d = ImageDraw.Draw(im)
            d.rectangle((6, 6, 12 + 7 * len(label), 26), fill=(0, 0, 0))
            d.text((10, 10), label, fill=(255, 255, 255))
            pair.append(np.asarray(im))
        rows.append(np.hstack([pair[0], np.full((pair[0].shape[0], 8, 3), 255, np.uint8), pair[1]]))
    gap = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    grid = np.vstack(sum(([r, gap] for r in rows), [])[:-1])
    (ROOT / "docs").mkdir(exist_ok=True)
    Image.fromarray(grid).save(ROOT / "docs/real_world.jpg", quality=88)
    (ROOT / "reports/real_world_haze.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

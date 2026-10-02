"""Resize the public-domain NASA hazy images for the website and write thumbnails.

Sources (Wikimedia Commons, NASA Earth Observatory / GSFC, public domain) are in assets/samples_src/.
"""

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "assets/samples_src", ROOT / "web/samples"

SAMPLES = [
    ("river_of_haze_india.jpg", "A river of haze over the Ganges plain, India (2012)", None),
    ("northern_india.jpg", "Haze over northern India (MODIS)", (0, 380, 900, 979)),
    ("eastern_china.jpg", "Haze over eastern China (MODIS, 2015-12-16)", (0, 40, 660, 620)),
    ("po_valley.jpg", "Haze in the Po Valley, Italy (MODIS, 2019-10-15)", (300, 440, 1200, 860)),
    ("seasonal_india_2024.jpg", "Seasonal haze over northern India (MODIS, 2024-11-02)", (60, 60, 1160, 900)),
]
CREDIT = "NASA Earth Observatory / Goddard Space Flight Center, public domain (via Wikimedia Commons)."


def main(max_side=1024, thumb=160):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    for name, title, crop in SAMPLES:
        img = Image.open(SRC / name).convert("RGB")
        if crop:   # focus on the hazy region instead of the whole MODIS swath
            img = img.crop(crop)
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        img.save(OUT / name, quality=90)
        t = img.copy()
        side = min(t.size)
        t = t.crop(((t.width - side) // 2, (t.height - side) // 2, (t.width + side) // 2, (t.height + side) // 2))
        t.thumbnail((thumb, thumb), Image.LANCZOS)
        t.save(OUT / name.replace(".jpg", "_thumb.jpg"), quality=85)
        meta.append({"file": name, "thumb": name.replace(".jpg", "_thumb.jpg"), "title": title, "credit": CREDIT,
                     "size": list(img.size)})
        print(name, img.size)
    (OUT / "samples.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

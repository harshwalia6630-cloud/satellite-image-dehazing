"""Export the trained model to ONNX for the website and verify it.

Writes
    web/model/dehaze_unet.onnx     dynamic H x W input (multiples of 8), NHWC float32 in [0, 1]
    web/model/config.json          tiling settings, test metrics, sample list
    tests/fixtures/tiling.json     a NASA sample crop + Python tiled output, for tests/web_parity.mjs
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dehaze import env  # noqa: E402

tf = env.setup()
import onnxruntime as ort  # noqa: E402
import tf2onnx  # noqa: E402

from dehaze.model import build_model  # noqa: E402
from dehaze.tiling import dehaze_tiled  # noqa: E402

WEB = ROOT / "web/model"


def main():
    tf.config.experimental.enable_tensor_float_32_execution(False)   # exact fp32 reference
    model = build_model(tf)
    model.load_weights(str(ROOT / "models/dehaze_unet.weights.h5"))
    WEB.mkdir(parents=True, exist_ok=True)
    out = WEB / "dehaze_unet.onnx"
    tf2onnx.convert.from_keras(model, input_signature=(tf.TensorSpec((None, None, None, 3), tf.float32, name="hazy"),),
                               opset=13, output_path=str(out))

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_predict = lambda x: sess.run(None, {"hazy": x.astype(np.float32)})[0]
    test = np.load(ROOT / "data/processed/test.npz")
    x = test["hazy"][:64].astype(np.float32) / 255
    keras_out = model.predict(x, verbose=0)
    onnx_out = np.concatenate([onnx_predict(x[i:i + 16]) for i in range(0, 64, 16)])
    diff = float(np.abs(keras_out - onnx_out).max())
    print(f"ONNX vs Keras on 64 test images: max |diff| = {diff:.2e}")
    assert diff < 1e-4

    # fixture for the JS tiling port: an odd-sized crop of a public-domain NASA sample, so the
    # fixture can be committed; the odd size forces padding and several overlapping tiles
    h, w = 301, 517
    big = np.asarray(Image.open(ROOT / "web/samples/po_valley.jpg").convert("RGB"), np.float32)[:h, :w] / 255
    tiled = dehaze_tiled(onnx_predict, big.astype(np.float32), tile=256, overlap=32)
    (ROOT / "tests/fixtures").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests/fixtures/tiling.json").write_text(json.dumps({
        "h": h, "w": w, "input_u8": (big * 255).round().astype(np.uint8).ravel().tolist(),
        "output": np.round(tiled, 6).ravel().tolist()}))

    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    samples = json.loads((ROOT / "web/samples/samples.json").read_text())
    (WEB / "config.json").write_text(json.dumps({"tile": 256, "overlap": 32, "metrics": metrics, "samples": samples}, indent=1))
    for p in sorted(WEB.iterdir()):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

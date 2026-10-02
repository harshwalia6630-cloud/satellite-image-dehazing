// Checks the browser tiling port (web/js/dehaze.js) against Python's dehaze_tiled + the same ONNX model.
// Fixture from `python scripts/export_web_model.py`.   Run: node tests/web_parity.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { dehazeTiled } from "../web/js/dehaze.js";

const root = new URL("../", import.meta.url);
const fx = JSON.parse(readFileSync(new URL("tests/fixtures/tiling.json", root)));
const session = await ort.InferenceSession.create(fileURLToPath(new URL("web/model/dehaze_unet.onnx", root)));
const rgb = Float32Array.from(fx.input_u8, v => v / 255);
const out = await dehazeTiled(ort, session, rgb, fx.h, fx.w, { tile: 256, overlap: 32 });

let maxDiff = 0;
for (let i = 0; i < out.length; i++) maxDiff = Math.max(maxDiff, Math.abs(out[i] - fx.output[i]));
console.log(`tiled dehazing ${fx.w}x${fx.h}: max |js - python| = ${maxDiff.toExponential(2)}`);
const ok = maxDiff < 1e-4;
console.log(ok ? "ALL PARITY CHECKS PASSED" : "FAILURE");
process.exit(ok ? 0 : 1);

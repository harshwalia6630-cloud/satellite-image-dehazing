// Patch-based dehazing in the browser: a JS port of src/dehaze/tiling.py.
// The image is reflect-padded, split into overlapping 256 px tiles, each tile runs through the
// ONNX encoder-decoder, and tiles are blended with raised-cosine weights so no seams appear.
// Runtime-agnostic: pass in the `ort` module (onnxruntime-web in the browser, onnxruntime-node in tests).

export function rampWindow(size, overlap) {
  const w = new Float32Array(size).fill(1);
  if (overlap > 0 && overlap < size) {
    for (let i = 0; i < overlap; i++) {
      const v = 0.5 - 0.5 * Math.cos((Math.PI * (i + 1)) / (overlap + 1));
      w[i] = v;
      w[size - 1 - i] = v;
    }
  }
  return w;
}

export function tilePositions(length, tile, stride) {
  if (length <= tile) return [0];
  const pos = [];
  for (let p = 0; p < length - tile; p += stride) pos.push(p);
  pos.push(length - tile);
  return pos;
}

// numpy.pad(mode="reflect") index mapping (edge pixel not repeated, reflects repeatedly if needed)
function reflectIndex(i, n) {
  if (n === 1) return 0;
  const period = 2 * (n - 1);
  i %= period;
  if (i < 0) i += period;
  return i < n ? i : period - i;
}

/**
 * @param rgb      Float32Array HxWx3 in [0,1]
 * @param onTile   optional progress callback (done, total)
 * @returns        Float32Array HxWx3 in [0,1]
 */
export async function dehazeTiled(ort, session, rgb, h, w, { tile = 256, overlap = 32, onTile } = {}) {
  let ph = Math.max(h, tile), pw = Math.max(w, tile);
  ph += (8 - (ph % 8)) % 8;
  pw += (8 - (pw % 8)) % 8;
  const padded = new Float32Array(ph * pw * 3);
  for (let y = 0; y < ph; y++) {
    const sy = reflectIndex(y, h);
    for (let x = 0; x < pw; x++) {
      const s = (sy * w + reflectIndex(x, w)) * 3, d = (y * pw + x) * 3;
      padded[d] = rgb[s]; padded[d + 1] = rgb[s + 1]; padded[d + 2] = rgb[s + 2];
    }
  }
  const th = Math.min(tile, ph), tw = Math.min(tile, pw);
  const wy = rampWindow(th, overlap), wx = rampWindow(tw, overlap);
  const out = new Float32Array(ph * pw * 3), weight = new Float32Array(ph * pw);
  const ys = tilePositions(ph, th, th - overlap), xs = tilePositions(pw, tw, tw - overlap);
  const total = ys.length * xs.length;
  const input = new Float32Array(th * tw * 3);
  const inputName = session.inputNames[0], outputName = session.outputNames[0];
  let done = 0;
  for (const y0 of ys) {
    for (const x0 of xs) {
      for (let y = 0; y < th; y++) input.set(padded.subarray(((y0 + y) * pw + x0) * 3, ((y0 + y) * pw + x0 + tw) * 3), y * tw * 3);
      const res = await session.run({ [inputName]: new ort.Tensor("float32", input, [1, th, tw, 3]) });
      const pred = res[outputName].data;
      for (let y = 0; y < th; y++) {
        for (let x = 0; x < tw; x++) {
          const wgt = wy[y] * wx[x], o = (y0 + y) * pw + x0 + x, p = (y * tw + x) * 3;
          out[o * 3] += pred[p] * wgt; out[o * 3 + 1] += pred[p + 1] * wgt; out[o * 3 + 2] += pred[p + 2] * wgt;
          weight[o] += wgt;
        }
      }
      onTile?.(++done, total);
    }
  }
  const result = new Float32Array(h * w * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = y * pw + x, r = (y * w + x) * 3, k = Math.max(weight[o], 1e-6);
      for (let c = 0; c < 3; c++) result[r + c] = Math.min(1, Math.max(0, out[o * 3 + c] / k));
    }
  }
  return result;
}

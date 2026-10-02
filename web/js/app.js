// onnxruntime-web is self-hosted: threaded WASM spawns workers, which browsers only allow from the same origin
import * as ort from "../vendor/ort/ort.wasm.min.mjs";
import { dehazeTiled } from "./dehaze.js";

ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;

const $ = id => document.getElementById(id);
const MAX_SIDE = 1024;               // larger images are downscaled to keep browser runtime reasonable
let session = null, config = null, busy = false;

// ---------- helpers
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) { onProgress(null); return new Uint8Array(await res.arrayBuffer()); }
  const reader = res.body.getReader(), buf = new Uint8Array(total);
  for (let loaded = 0; ;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, loaded); loaded += value.length; onProgress(loaded / total);
  }
  return buf;
}

function toRGB(imageData) {
  const n = imageData.width * imageData.height, px = imageData.data, rgb = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { rgb[i * 3] = px[i * 4] / 255; rgb[i * 3 + 1] = px[i * 4 + 1] / 255; rgb[i * 3 + 2] = px[i * 4 + 2] / 255; }
  return rgb;
}

function drawRGB(canvas, rgb, w, h) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    img.data[i * 4] = Math.round(rgb[i * 3] * 255); img.data[i * 4 + 1] = Math.round(rgb[i * 3 + 1] * 255);
    img.data[i * 4 + 2] = Math.round(rgb[i * 3 + 2] * 255); img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// mean dark channel (min over RGB, then over 15x15 blocks): a quick haze-density estimate (He et al. 2009)
function hazeDensity(rgb, w, h, block = 15) {
  let sum = 0, count = 0;
  for (let by = 0; by < h; by += block) {
    for (let bx = 0; bx < w; bx += block) {
      let m = 1;
      for (let y = by; y < Math.min(h, by + block); y++)
        for (let x = bx; x < Math.min(w, bx + block); x++) {
          const i = (y * w + x) * 3;
          m = Math.min(m, rgb[i], rgb[i + 1], rgb[i + 2]);
        }
      sum += m; count++;
    }
  }
  return sum / count;
}

function setBusy(on, text = "Dehazing…") {
  busy = on;
  $("busy").hidden = !on;
  $("busy-text").textContent = text;
  document.querySelectorAll(".sample").forEach(b => (b.disabled = on));
  $("upload").classList.toggle("disabled", on);
}

// ---------- processing
async function process(source, label, credit = "") {
  if (!session || busy) return;
  const scale = Math.min(1, MAX_SIDE / Math.max(source.width, source.height));
  const w = Math.round(source.width * scale), h = Math.round(source.height * scale);
  const c = new OffscreenCanvas(w, h), ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, w, h);
  const rgb = toRGB(ctx.getImageData(0, 0, w, h));

  $("placeholder").hidden = true;
  drawRGB($("before"), rgb, w, h);
  drawRGB($("after"), rgb, w, h);
  $("compare").classList.remove("ready");
  $("download").disabled = true;
  $("r-size").textContent = `${w} × ${h}${scale < 1 ? " (downscaled)" : ""}`;
  $("credit").textContent = credit;
  setBusy(true);
  const t0 = performance.now();
  try {
    const out = await dehazeTiled(ort, session, rgb, h, w, {
      tile: config.tile, overlap: config.overlap,
      onTile: (d, n) => { $("tile-bar").style.width = `${(d / n) * 100}%`; $("busy-text").textContent = `Dehazing patch ${d} / ${n}`; $("r-tiles").textContent = `${n} × ${config.tile}px`; },
    });
    drawRGB($("after"), out, w, h);
    $("r-time").textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`;
    $("r-haze").textContent = `${hazeDensity(rgb, w, h).toFixed(3)} → ${hazeDensity(out, w, h).toFixed(3)}`;
    $("compare").classList.add("ready");
    $("download").disabled = false;
    $("download").dataset.name = `${label.replace(/\.[^.]+$/, "")}_dehazed.png`;
  } catch (err) {
    console.error(err);
    $("r-time").textContent = "failed";
  } finally {
    setBusy(false);
    $("tile-bar").style.width = "0";
  }
}

// ---------- comparison slider
function setSlider(pct) {
  pct = Math.max(0, Math.min(100, pct));
  $("before").style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  $("handle").style.left = `${pct}%`;
  $("handle").setAttribute("aria-valuenow", Math.round(pct));
}
const compare = $("compare");
let dragging = false;
const pctFromEvent = e => {
  // the canvases use object-fit: contain, so map pointer x into the drawn image area
  const r = compare.getBoundingClientRect(), cw = $("after").width, ch = $("after").height;
  const s = Math.min(r.width / cw, r.height / ch), drawnW = cw * s, left = r.left + (r.width - drawnW) / 2;
  return ((e.clientX - left) / drawnW) * 100;
};
compare.addEventListener("pointerdown", e => { if (!compare.classList.contains("ready")) return; dragging = true; compare.setPointerCapture(e.pointerId); setSlider(pctFromEvent(e)); });
compare.addEventListener("pointermove", e => dragging && setSlider(pctFromEvent(e)));
compare.addEventListener("pointerup", () => (dragging = false));
$("handle").addEventListener("keydown", e => {
  const now = Number($("handle").getAttribute("aria-valuenow"));
  if (e.key === "ArrowLeft") setSlider(now - 5);
  if (e.key === "ArrowRight") setSlider(now + 5);
});
// keep the handle aligned with the image when the clip is relative to the canvas box
new ResizeObserver(() => setSlider(Number($("handle").getAttribute("aria-valuenow")))).observe(compare);

// ---------- wiring
$("file").addEventListener("change", async e => {
  const f = e.target.files[0];
  if (!f) return;
  try { await process(await createImageBitmap(f), f.name, "Your image (processed locally)."); }
  catch { $("status").textContent = "Could not read that image."; }
  e.target.value = "";
});
$("download").addEventListener("click", () => {
  $("after").toBlob(b => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b); a.download = $("download").dataset.name || "dehazed.png"; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, "image/png");
});

function renderResults(m) {
  const all = m.summary.all, rows = [["Hazy input (no processing)", all["hazy input"]], ["Dark channel prior", all["dark channel prior"]], ["This model", all.ours]];
  $("results-table").querySelector("tbody").innerHTML = rows.map(([name, r], i) =>
    `<tr class="${i === 2 ? "ours" : ""}"><td>${name}</td><td>${r.psnr.toFixed(2)}</td><td>${r.ssim.toFixed(3)}</td><td>${r.residual_haze.toFixed(3)}</td></tr>`).join("");
  $("by-level").textContent = ["thin", "moderate", "thick"].map(l => `${l} ${m.summary[l].ours.psnr.toFixed(1)} dB`).join(" · ");
  $("s-psnr").textContent = `${all.ours.psnr.toFixed(1)} dB`;
  $("s-gain").textContent = `+${m.psnr_gain_vs_dcp_pct}%`;
  $("s-ssim").textContent = all.ours.ssim.toFixed(3);
  $("s-haze").textContent = `${m.residual_haze_reduction_vs_dcp_pct}%`;
}

(async () => {
  try {
    config = await fetch("model/config.json").then(r => r.json());
    renderResults(config.metrics);
    for (const s of config.samples) {
      const b = document.createElement("button");
      b.className = "sample"; b.type = "button"; b.title = s.title; b.disabled = true;
      b.innerHTML = `<img src="samples/${s.thumb}" alt="${s.title}" loading="lazy">`;
      b.addEventListener("click", async () => {
        document.querySelectorAll(".sample").forEach(x => x.classList.toggle("active", x === b));
        const img = new Image();
        img.src = `samples/${s.file}`;
        await img.decode();
        process(img, s.file, `${s.title}. ${s.credit}`);
      });
      $("samples").appendChild(b);
    }
    const bytes = await fetchWithProgress("model/dehaze_unet.onnx", p => {
      $("status").textContent = p === null ? "Downloading model…" : `Downloading model… ${(p * 100).toFixed(0)}%`;
      if (p !== null) $("load-progress").firstElementChild.style.width = `${p * 100}%`;
    });
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    $("load-progress").firstElementChild.style.width = "0";
    $("status").textContent = `Model ready · ONNX Runtime Web (WASM, ${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads > 1 ? "s" : ""})`;
    document.querySelectorAll(".sample").forEach(b => (b.disabled = false));
    $("upload").classList.remove("disabled");
    document.querySelector(".sample")?.click();
  } catch (err) {
    console.error(err);
    $("status").textContent = "Could not load the model. Please refresh.";
  }
})();
$("upload").classList.add("disabled");

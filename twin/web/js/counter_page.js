// Latency counter page (see src/mhd/twin/screen_counter.py). Draws the twin counter code
// (magic 0b1011, render_id mod 2^20, render_ms mod 2^24, CRC-16/CCITT-FALSE; 64 bits) as an 8x8
// grid, dark = 1, framed by ArUco markers 0..3 (TL, TR, BR, BL). render_ms is on the server's
// monotonic clock (mhd.contracts.now_ms), synced over loopback with the lowest-RTT sample.
// ?static=RID,MS draws one fixed code (tests).
const CELL = 48, GRID = 8 * CELL, MARKER = 96, GAP = 24;
const EXTENT = GRID + 2 * (GAP + MARKER);                 // page units, square
const canvas = document.getElementById('c');
const info = document.getElementById('info');
const ctx = canvas.getContext('2d');
const params = new URLSearchParams(location.search);
const staticCode = params.get('static');
let offset = null, bestRtt = Infinity, renderId = 0;
const markers = [0, 1, 2, 3].map((k) => { const im = new Image(); im.src = `/counter/marker/${k}.png`; return im; });

function crc16(bytes) {
  let crc = 0xFFFF;
  for (const b of bytes) {
    crc ^= b << 8;
    for (let i = 0; i < 8; i++) { crc = (crc & 0x8000) ? ((crc << 1) ^ 0x1021) : (crc << 1); crc &= 0xFFFF; }
  }
  return crc;
}

export function encodeBits(rid, ms) {
  const bits = [];
  const r = rid % 1048576, m = ((Math.round(ms) % 16777216) + 16777216) % 16777216;
  for (let i = 3; i >= 0; i--) bits.push((0b1011 >> i) & 1);
  for (let i = 19; i >= 0; i--) bits.push(Math.floor(r / 2 ** i) % 2);
  for (let i = 23; i >= 0; i--) bits.push(Math.floor(m / 2 ** i) % 2);
  const bytes = [];
  for (let k = 0; k < 6; k++) { let v = 0; for (let j = 0; j < 8; j++) v = (v << 1) | bits[k * 8 + j]; bytes.push(v); }
  const crc = crc16(bytes);
  for (let i = 15; i >= 0; i--) bits.push((crc >> i) & 1);
  return bits;
}

async function syncClock() {
  for (let i = 0; i < 12; i++) {
    const t0 = performance.now();
    const r = await fetch('/clock', { cache: 'no-store' });
    const t1 = performance.now();
    const { now_ms } = await r.json();
    const rtt = t1 - t0;
    if (rtt < bestRtt || offset === null) { bestRtt = rtt; offset = now_ms - (t0 + t1) / 2; }
  }
}

function draw(rid, ms) {
  const w = canvas.width = window.innerWidth * devicePixelRatio;
  const h = canvas.height = window.innerHeight * devicePixelRatio;
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, h);
  const s = Math.min(w, h) * 0.94 / EXTENT;
  const ox = (w - EXTENT * s) / 2 + (GAP + MARKER) * s, oy = (h - EXTENT * s) / 2 + (GAP + MARKER) * s;
  const centers = [[-GAP - MARKER / 2, -GAP - MARKER / 2], [GRID + GAP + MARKER / 2, -GAP - MARKER / 2],
                   [GRID + GAP + MARKER / 2, GRID + GAP + MARKER / 2], [-GAP - MARKER / 2, GRID + GAP + MARKER / 2]];
  ctx.imageSmoothingEnabled = false;
  centers.forEach(([cx, cy], k) => {
    if (markers[k].complete) ctx.drawImage(markers[k], ox + (cx - MARKER / 2) * s, oy + (cy - MARKER / 2) * s, MARKER * s, MARKER * s);
  });
  const bits = encodeBits(rid, ms);
  ctx.fillStyle = '#000';
  for (let i = 0; i < 64; i++) {
    if (!bits[i]) continue;
    const r = Math.floor(i / 8), c = i % 8;
    ctx.fillRect(Math.floor(ox + c * CELL * s), Math.floor(oy + r * CELL * s), Math.ceil(CELL * s), Math.ceil(CELL * s));
  }
  window.__counter = { renderId: rid, renderMs: Math.round(ms), offset, bestRtt };
}

function frame(ts) {
  renderId += 1;
  draw(renderId, ts + offset);
  if (renderId % 30 === 0) info.textContent = `render id ${renderId}   clock sync RTT ${bestRtt.toFixed(2)} ms`;
  requestAnimationFrame(frame);
}

(async () => {
  if (staticCode) {
    const [rid, ms] = staticCode.split(',').map(Number);
    const go = () => { draw(rid, ms); info.textContent = `static ${rid}, ${ms}`; window.__ready = true; };
    Promise.all(markers.map((m) => m.decode())).then(go, go);
    return;
  }
  await syncClock();
  setInterval(syncClock, 10000);
  await Promise.all(markers.map((m) => m.decode()));
  window.__ready = true;
  requestAnimationFrame(frame);
})();

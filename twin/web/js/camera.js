// Drone camera page. The twin server pushes {"type":"pose"}; we render once and answer with
// binary [uint32 LE render_id][JPEG]. One frame in flight; the server paces the frame rate.
import * as THREE from 'three';
import { World, applyDroneCamera, makeRenderer, vfovDeg, webglInfo } from './world.js';

const W = 1280, H = 720;
const canvas = document.getElementById('c');
const hud = document.getElementById('hud');
const renderer = makeRenderer(canvas, W, H);
const world = new World();
const camera = new THREE.PerspectiveCamera(vfovDeg(75, W, H), W / H, 0.05, 300);
const params = new URLSearchParams(location.search);
const quality = parseFloat(params.get('q') || '0.92');
let frames = 0, lastStats = performance.now(), renderMs = 0, encodeMs = 0;

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/render?role=camera`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    const info = webglInfo(renderer);
    ws.send(JSON.stringify({ type: 'hello', role: 'camera', webgl: info.renderer, vendor: info.vendor,
                             ua: navigator.userAgent }));
    hud.textContent = 'connected: ' + info.renderer;
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return;
    const msg = JSON.parse(ev.data);
    if (msg.type === 'scene') {
      world.build(msg);
      camera.fov = vfovDeg(msg.camera.hfov_deg, W, H);
      camera.updateProjectionMatrix();
      return;
    }
    if (msg.type !== 'pose') return;
    const t0 = performance.now();
    world.update(msg);
    applyDroneCamera(camera, msg.camera);
    if (msg.camera.hfov_deg && Math.abs(camera.fov - vfovDeg(msg.camera.hfov_deg, W, H)) > 1e-3) {
      camera.fov = vfovDeg(msg.camera.hfov_deg, W, H); camera.updateProjectionMatrix();
    }
    renderer.render(world.scene, camera);
    const t1 = performance.now();
    canvas.toBlob((blob) => {
      const t2 = performance.now();
      renderMs = 0.9 * renderMs + 0.1 * (t1 - t0);
      encodeMs = 0.9 * encodeMs + 0.1 * (t2 - t1);
      const head = new Uint32Array([msg.render_id >>> 0]);
      if (ws.readyState === WebSocket.OPEN) ws.send(new Blob([head.buffer, blob]));
      frames++;
      const now = performance.now();
      if (now - lastStats > 2000) {
        const fps = frames * 1000 / (now - lastStats);
        ws.send(JSON.stringify({ type: 'stats', fps: +fps.toFixed(2), render_ms: +renderMs.toFixed(2),
                                 encode_ms: +encodeMs.toFixed(2) }));
        hud.textContent = `${fps.toFixed(1)} fps | render ${renderMs.toFixed(1)} ms | jpeg ${encodeMs.toFixed(1)} ms`;
        frames = 0; lastStats = now;
      }
    }, 'image/jpeg', quality);
  };
  ws.onclose = () => { hud.textContent = 'disconnected; retrying'; setTimeout(connect, 1000); };
  ws.onerror = () => ws.close();
}
window.__twin = { renderer, world, camera, webgl: () => webglInfo(renderer) };
connect();

// Chase view for the operator UI and the demo video: a camera behind and above the drone,
// with the drone-camera view as picture-in-picture. ?rec=1 makes the server pull frames
// (role chase_rec) for the MP4 recording; otherwise it is a passive viewer.
import * as THREE from 'three';
import { DRONE_LAYER, World, applyDroneCamera, enu, makeRenderer, vfovDeg, webglInfo } from './world.js';

const params = new URLSearchParams(location.search);
const rec = params.get('rec') === '1';
const W = rec ? 1280 : window.innerWidth, H = rec ? 720 : window.innerHeight;
const canvas = document.getElementById('c');
const hud = document.getElementById('hud');
if (rec) { canvas.style.width = W + 'px'; canvas.style.height = H + 'px'; }
const renderer = makeRenderer(canvas, W, H);
renderer.autoClear = false;
const world = new World();
const chaseCam = new THREE.PerspectiveCamera(55, W / H, 0.05, 300);
chaseCam.layers.enable(DRONE_LAYER);
const droneCam = new THREE.PerspectiveCamera(vfovDeg(75, 1280, 720), 1280 / 720, 0.05, 300);
const target = new THREE.Vector3(), desired = new THREE.Vector3();
let initialized = false;
let labels = [];

function placeChase(d) {
  const back = 2.6, up = 1.3;
  const fx = Math.sin(d.yaw), fy = Math.cos(d.yaw);
  desired.copy(enu(d.x - fx * back, d.y - fy * back, Math.max(0.3, d.z) + up));
  target.copy(enu(d.x + fx * 2.5, d.y + fy * 2.5, Math.max(0.2, d.z) * 0.6));
  if (!initialized) { chaseCam.position.copy(desired); initialized = true; }
  chaseCam.position.lerp(desired, 0.12);
  chaseCam.lookAt(target);
}

function draw(msg) {
  world.update(msg);
  placeChase(msg.drone);
  applyDroneCamera(droneCam, msg.camera);
  renderer.setViewport(0, 0, W, H); renderer.setScissorTest(false);
  renderer.clear();
  renderer.render(world.scene, chaseCam);
  // picture-in-picture: what the drone camera sees
  const pw = Math.round(W * 0.28), ph = Math.round(pw * 720 / 1280), m = 14;
  renderer.setScissorTest(true);
  renderer.setScissor(W - pw - m - 3, H - ph - m - 3, pw + 6, ph + 6);
  renderer.setClearColor('#ffffff', 1); renderer.clear();
  renderer.setScissor(W - pw - m, H - ph - m, pw, ph);
  renderer.setViewport(W - pw - m, H - ph - m, pw, ph);
  renderer.render(world.scene, droneCam);
  renderer.setScissorTest(false);
  renderer.setClearColor('#000000', 1);
  hud.textContent = `TWIN  ${world.desc ? world.desc.name.toUpperCase() : ''}  t=${msg.t.toFixed(1)} s  ` +
                    `phase ${msg.phase}  alt ${msg.drone.z.toFixed(2)} m  yaw ${(msg.drone.yaw * 180 / Math.PI).toFixed(0)}°`;
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/render?role=${rec ? 'chase_rec' : 'chase'}`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    const info = webglInfo(renderer);
    ws.send(JSON.stringify({ type: 'hello', role: rec ? 'chase_rec' : 'chase', webgl: info.renderer }));
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return;
    const msg = JSON.parse(ev.data);
    if (msg.type === 'scene') {
      world.build(msg); initialized = false;
      droneCam.fov = vfovDeg(msg.camera.hfov_deg, 1280, 720); droneCam.updateProjectionMatrix();
      return;
    }
    if (msg.type !== 'pose') return;
    draw(msg);
    if (rec) {
      canvas.toBlob((blob) => {
        const head = new Uint32Array([msg.render_id >>> 0]);
        if (ws.readyState === WebSocket.OPEN) ws.send(new Blob([head.buffer, blob]));
      }, 'image/jpeg', 0.9);
    }
  };
  ws.onclose = () => { hud.textContent = 'disconnected; retrying'; setTimeout(connect, 1000); };
  ws.onerror = () => ws.close();
}
if (!rec) {
  window.addEventListener('resize', () => location.reload());
}
window.__twin = { renderer, world, chaseCam, droneCam, webgl: () => webglInfo(renderer) };
connect();

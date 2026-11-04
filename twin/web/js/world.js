// Twin world for the Three.js renderer: ground, sky, walls, trees, props, people, cars, drone.
//
// Coordinates: the twin works in ENU meters (x east, y north, z up). Three.js is y-up, so
//   three.x = x, three.y = z, three.z = -y.
// Headings (drone yaw, actor heading) are clockwise from north, so an object facing heading h
// gets rotation.y = -h, with its own forward along local -z.
// The drone camera uses rotation order 'YXZ': (pitch, -yaw, -roll), matching
// src/mhd/twin/scenarios.py Camera (pitch + = up, roll + = right side down).
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

export function enu(x, y, z = 0) { return new THREE.Vector3(x, z, -y); }
export const DRONE_LAYER = 1;

function rng(seed) {             // small deterministic PRNG so every run looks the same
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

function canvasTexture(w, h, draw, repeat = [1, 1]) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  draw(g, w, h);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeat[0], repeat[1]);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

// ---------------------------------------------------------------- ground and sky
function grassTexture(sizeM) {
  return canvasTexture(1024, 1024, (g, w, h) => {
    const r = rng(11);
    g.fillStyle = '#4f7a35'; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 2600; i++) {          // soft patches
      const x = r() * w, y = r() * h, rad = 8 + r() * 40;
      const l = 28 + r() * 18, s = 35 + r() * 25, hue = 85 + r() * 25;
      g.fillStyle = `hsla(${hue},${s}%,${l}%,0.18)`;
      g.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2); g.fill();
    }
    for (let i = 0; i < 60000; i++) {         // blades
      const x = r() * w, y = r() * h, len = 3 + r() * 7, a = -Math.PI / 2 + (r() - 0.5) * 0.9;
      const l = 22 + r() * 30, hue = 80 + r() * 35;
      g.strokeStyle = `hsla(${hue},45%,${l}%,0.55)`;
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(a) * len, y + Math.sin(a) * len); g.stroke();
    }
    for (let i = 0; i < 400; i++) {           // a few dry spots and clover flowers
      g.fillStyle = r() < 0.7 ? 'rgba(150,140,80,0.25)' : 'rgba(240,240,230,0.7)';
      g.fillRect(r() * w, r() * h, 2 + r() * 2, 2 + r() * 2);
    }
  }, [sizeM / 2.5, sizeM / 2.5]);
}

function asphaltTexture() {
  return canvasTexture(512, 512, (g, w, h) => {
    const r = rng(5);
    g.fillStyle = '#3d3f42'; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 30000; i++) {
      const l = 18 + r() * 40;
      g.fillStyle = `hsl(210,4%,${l}%)`;
      g.fillRect(r() * w, r() * h, 1 + r() * 2, 1 + r() * 2);
    }
  }, [6, 1]);
}

function sidingTexture(color) {
  return canvasTexture(512, 512, (g, w, h) => {
    g.fillStyle = color; g.fillRect(0, 0, w, h);
    for (let y = 0; y < h; y += 24) {
      g.fillStyle = 'rgba(0,0,0,0.13)'; g.fillRect(0, y + 20, w, 4);
      g.fillStyle = 'rgba(255,255,255,0.10)'; g.fillRect(0, y, w, 3);
    }
  }, [4, 2]);
}

function plankTexture(color) {
  return canvasTexture(512, 256, (g, w, h) => {
    const r = rng(3);
    g.fillStyle = color; g.fillRect(0, 0, w, h);
    for (let x = 0; x < w; x += 32) {
      g.fillStyle = `rgba(0,0,0,${0.10 + r() * 0.1})`; g.fillRect(x, 0, 3, h);
      g.fillStyle = `rgba(255,255,255,${0.04 + r() * 0.05})`; g.fillRect(x + 6, 0, 10, h);
      for (let k = 0; k < 6; k++) {           // wood grain
        g.strokeStyle = 'rgba(60,35,15,0.18)';
        g.beginPath(); const gx = x + 4 + r() * 26; g.moveTo(gx, 0); g.lineTo(gx + (r() - 0.5) * 6, h); g.stroke();
      }
    }
  }, [6, 1]);
}

function skyTexture() {
  const t = canvasTexture(4, 512, (g, w, h) => {
    const grd = g.createLinearGradient(0, 0, 0, h);
    grd.addColorStop(0.0, '#3f7fd0');
    grd.addColorStop(0.42, '#8fbbe8');
    grd.addColorStop(0.5, '#cfe1f2');
    grd.addColorStop(0.52, '#b9c6b0');
    grd.addColorStop(1.0, '#6d7f5c');
    g.fillStyle = grd; g.fillRect(0, 0, w, h);
  });
  t.mapping = THREE.EquirectangularReflectionMapping;
  return t;
}

// ---------------------------------------------------------------- people
// Skeleton for a 1.74 m person (scaled by height_m / 1.74). These numbers are duplicated in
// src/mhd/twin/scenarios.py (PERSON_SKELETON), which computes the ground-truth boxes from the same
// forward kinematics; tests/twin/test_scenarios.py guards the copy. Change both together.
export const SKEL = {
  refH: 1.74, hipY: 0.92, hipX: 0.095, thigh: 0.45, thighR: 0.075, shin: 0.42, shinR: 0.058,
  footW: 0.10, footH: 0.075, footL: 0.26, footDy: -0.02, footDz: -0.06,
  pelvisY: 0.06, torsoY: 0.33, torsoH: 0.46, torsoRTop: 0.19, torsoRBot: 0.165, torsoZ: 0.62,
  chestY: 0.56, neckY: 0.6, headY: 0.703, headR: 0.105, headSy: 1.12, headSx: 0.92,
  shoulderX: 0.215, shoulderY: 0.53, shoulderAbd: 0.06, upperArm: 0.30, upperArmR: 0.052,
  elbow: 0.29, forearm: 0.26, forearmR: 0.045, hand: 0.29,
  stride: 0.95, hipSwing: 0.38, kneeBase: 0.08, kneeSwing: 0.5, kneeRest: 0.03, ankleSwing: 0.15,
  armSwing: 0.3, elbowBase: 0.18, elbowSwing: 0.12, bob: 0.012,
};
const PERSON_REF_H = SKEL.refH;

function mat(color, rough = 0.85, extra = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: 0.0, ...extra });
}

function limb(radius, length, material) {
  // capsule hanging down from its pivot (pivot at the top)
  const g = new THREE.CapsuleGeometry(radius, Math.max(0.001, length - 2 * radius), 6, 12);
  const m = new THREE.Mesh(g, material);
  m.position.y = -length / 2;
  m.castShadow = true;
  return m;
}

export function buildPerson(desc) {
  const c = desc.colors || {};
  const K = SKEL;
  const shirt = mat(c.shirt || '#1f4fc4', 0.82);
  const pants = mat(c.pants || '#333333', 0.9);
  const skin = mat(c.skin || '#c58c6d', 0.6);
  const hair = mat(c.hair || '#2b1d14', 0.95);
  const shoes = mat(c.shoes || '#2a2a2a', 0.7);
  const s = (desc.height_m || PERSON_REF_H) / PERSON_REF_H;

  const root = new THREE.Group();
  const body = new THREE.Group();          // the whole figure; bobs while walking
  body.scale.setScalar(s);
  root.add(body);
  const hipY = K.hipY;
  const pelvis = new THREE.Mesh(new RoundedBoxGeometry(0.34, 0.2, 0.21, 3, 0.07), pants);
  pelvis.position.y = hipY + K.pelvisY; pelvis.castShadow = true; body.add(pelvis);
  const torsoG = new THREE.CylinderGeometry(K.torsoRTop, K.torsoRBot, K.torsoH, 20, 1);
  torsoG.scale(1, 1, K.torsoZ);           // elliptical cross-section
  const torso = new THREE.Mesh(torsoG, shirt);
  torso.position.y = hipY + K.torsoY; torso.castShadow = true; body.add(torso);
  const chest = new THREE.Mesh(new THREE.SphereGeometry(K.torsoRTop, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2), shirt);
  chest.scale.set(1, 0.42, K.torsoZ); chest.position.y = hipY + K.chestY; chest.castShadow = true; body.add(chest);
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.052, 0.058, 0.1, 12), skin);
  neck.position.y = hipY + K.neckY; body.add(neck);
  const head = new THREE.Mesh(new THREE.SphereGeometry(K.headR, 24, 18), skin);
  head.scale.set(K.headSx, K.headSy, 1.0); head.position.y = hipY + K.headY; head.castShadow = true; body.add(head);
  const hairCap = new THREE.Mesh(new THREE.SphereGeometry(0.112, 24, 14, 0, Math.PI * 2, 0, Math.PI * 0.55), hair);
  hairCap.scale.set(0.95, 1.08, 1.04); hairCap.position.set(0, hipY + K.headY + 0.003, 0.006); hairCap.rotation.x = 0.25;
  body.add(hairCap);
  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.018, 8, 6), skin);
  nose.position.set(0, hipY + K.headY - 0.01, -0.103); body.add(nose);
  for (const ex of [-0.035, 0.035]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.011, 8, 6), mat('#1a1a1a', 0.3));
    eye.position.set(ex, hipY + K.headY + 0.02, -0.094); body.add(eye);
  }
  // legs: hip pivot -> thigh -> knee pivot -> shin -> ankle pivot -> shoe
  const legs = [];
  for (const side of [-1, 1]) {
    const hip = new THREE.Group(); hip.position.set(side * K.hipX, hipY, 0); body.add(hip);
    hip.add(limb(K.thighR, K.thigh, pants));
    const knee = new THREE.Group(); knee.position.y = -K.thigh; hip.add(knee);
    knee.add(limb(K.shinR, K.shin, pants));
    const ankle = new THREE.Group(); ankle.position.y = -K.shin; knee.add(ankle);
    const foot = new THREE.Mesh(new RoundedBoxGeometry(K.footW, K.footH, K.footL, 2, 0.03), shoes);
    foot.position.set(0, K.footDy, K.footDz); foot.castShadow = true; ankle.add(foot);
    legs.push({ hip, knee, ankle, side });
  }
  // arms: shoulder pivot -> upper arm -> elbow pivot -> forearm -> hand (long sleeves)
  const arms = [];
  for (const side of [-1, 1]) {
    const sh = new THREE.Group(); sh.position.set(side * K.shoulderX, hipY + K.shoulderY, 0); body.add(sh);
    const cap = new THREE.Mesh(new THREE.SphereGeometry(0.062, 14, 10), shirt); cap.castShadow = true; sh.add(cap);
    sh.add(limb(K.upperArmR, K.upperArm, shirt));
    const el = new THREE.Group(); el.position.y = -K.elbow; sh.add(el);
    el.add(limb(K.forearmR, K.forearm, shirt));
    const hand = new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 10), skin);
    hand.scale.set(0.8, 1.1, 0.6); hand.position.y = -K.hand; hand.castShadow = true; el.add(hand);
    sh.rotation.z = side * K.shoulderAbd;  // arms hang slightly away from the body
    arms.push({ sh, el, side });
  }
  root.userData = { kind: 'person', body, legs, arms };
  return root;
}

// Joint angles from the distance walked; stateless, so the Python truth can reproduce them.
// + rotation.x swings a hanging limb forward (toward local -z, the facing direction).
export function gaitAngles(dist, speed) {
  const K = SKEL;
  const w = speed > 0.01 ? 1 : 0;
  const ph = (dist / K.stride) * Math.PI * 2;
  const out = { bob: K.bob * Math.abs(Math.cos(ph)) * w, legs: {}, arms: {} };
  for (const side of [-1, 1]) {
    const p = ph + (side > 0 ? Math.PI : 0);
    out.legs[side] = {
      hip: -K.hipSwing * Math.sin(p) * w,
      knee: -((K.kneeBase + K.kneeSwing * Math.max(0, Math.sin(p - 1.2))) * w + K.kneeRest),
      ankle: K.ankleSwing * Math.max(0, -Math.sin(p)) * w,
    };
    const pa = ph + (side > 0 ? 0 : Math.PI);
    out.arms[side] = {
      shoulder: -K.armSwing * Math.sin(pa) * w,
      elbow: (K.elbowBase + K.elbowSwing * Math.max(0, -Math.sin(pa))) * (0.4 + 0.6 * w),
    };
  }
  return out;
}

function animatePerson(obj, st) {
  const u = obj.userData;
  const g = gaitAngles(st.dist, st.speed);
  for (const leg of u.legs) {
    const a = g.legs[leg.side];
    leg.hip.rotation.x = a.hip; leg.knee.rotation.x = a.knee; leg.ankle.rotation.x = a.ankle;
  }
  for (const arm of u.arms) {
    const a = g.arms[arm.side];
    arm.sh.rotation.x = a.shoulder; arm.el.rotation.x = a.elbow;
  }
  u.body.position.y = g.bob;
}

// ---------------------------------------------------------------- cars
export function buildCar(desc) {
  const [L, W, H] = desc.size_m || [4.3, 1.8, 1.45];
  const toy = L < 1.0;
  const paint = new THREE.MeshPhysicalMaterial({ color: (desc.colors || {}).paint || '#1e9e3a', roughness: 0.32,
                                                 metalness: 0.25, clearcoat: 1.0, clearcoatRoughness: 0.12 });
  const glass = new THREE.MeshPhysicalMaterial({ color: '#1d242b', roughness: 0.08, metalness: 0.3 });
  const tire = mat('#141414', 0.9);
  const hub = mat('#b9bec4', 0.4, { metalness: 0.6 });
  const root = new THREE.Group();
  const wheelR = toy ? 0.3 * H : 0.22 * H;
  const bodyH = toy ? 0.45 * H : 0.5 * H;
  const bodyY = wheelR * (toy ? 0.9 : 0.75) + bodyH / 2;
  const body = new THREE.Mesh(new RoundedBoxGeometry(W * (toy ? 0.78 : 0.98), bodyH, L, 4, Math.min(0.12, bodyH * 0.4)), paint);
  body.position.y = bodyY; body.castShadow = true; root.add(body);
  const cabH = toy ? 0.38 * H : 0.42 * H;
  const cabin = new THREE.Mesh(new RoundedBoxGeometry(W * (toy ? 0.62 : 0.86), cabH, L * (toy ? 0.42 : 0.5), 4,
                                                      Math.min(0.1, cabH * 0.35)), glass);
  cabin.position.set(0, bodyY + bodyH / 2 + cabH / 2 - 0.01 * H, L * 0.06); cabin.castShadow = true; root.add(cabin);
  const roof = new THREE.Mesh(new RoundedBoxGeometry(W * (toy ? 0.64 : 0.88), cabH * 0.16, L * (toy ? 0.36 : 0.44), 2,
                                                     cabH * 0.05), paint);
  roof.position.set(0, bodyY + bodyH / 2 + cabH - 0.01 * H, L * 0.06); roof.castShadow = true; root.add(roof);
  const hood = new THREE.Mesh(new RoundedBoxGeometry(W * (toy ? 0.7 : 0.94), bodyH * 0.12, L * 0.3, 2, bodyH * 0.05), paint);
  hood.position.set(0, bodyY + bodyH / 2, -L * 0.32); root.add(hood);
  // lights
  const head = mat('#fffbe8', 0.2, { emissive: '#fff4c0', emissiveIntensity: 0.5 });
  const tail = mat('#b0141a', 0.3, { emissive: '#600000', emissiveIntensity: 0.4 });
  for (const sx of [-1, 1]) {
    const hl = new THREE.Mesh(new THREE.BoxGeometry(W * 0.16, bodyH * 0.18, 0.01 + L * 0.01), head);
    hl.position.set(sx * W * 0.3, bodyY + bodyH * 0.18, -L / 2 - 0.003); root.add(hl);
    const tl = new THREE.Mesh(new THREE.BoxGeometry(W * 0.16, bodyH * 0.16, 0.01 + L * 0.01), tail);
    tl.position.set(sx * W * 0.3, bodyY + bodyH * 0.2, L / 2 + 0.003); root.add(tl);
  }
  // wheels
  const wheels = [];
  const tw = toy ? W * 0.24 : W * 0.13;
  for (const [sx, sz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    const wg = new THREE.Group();
    wg.position.set(sx * (W / 2 - tw / 2 + (toy ? 0.02 * W : 0)), wheelR, sz * L * 0.33);
    const t = new THREE.Mesh(new THREE.CylinderGeometry(wheelR, wheelR, tw, 24), tire);
    t.rotation.z = Math.PI / 2; t.castShadow = true; wg.add(t);
    const h = new THREE.Mesh(new THREE.CylinderGeometry(wheelR * 0.55, wheelR * 0.55, tw * 1.02, 16), hub);
    h.rotation.z = Math.PI / 2; wg.add(h);
    root.add(wg); wheels.push(wg);
  }
  if (toy) {   // RC buggy details: rear wing and antenna
    const wing = new THREE.Mesh(new THREE.BoxGeometry(W * 0.9, H * 0.04, L * 0.14), paint);
    wing.position.set(0, bodyY + bodyH / 2 + H * 0.28, L * 0.43); wing.castShadow = true; root.add(wing);
    for (const sx of [-1, 1]) {
      const st = new THREE.Mesh(new THREE.BoxGeometry(W * 0.04, H * 0.28, L * 0.04), mat('#222222'));
      st.position.set(sx * W * 0.25, bodyY + bodyH / 2 + H * 0.14, L * 0.43); root.add(st);
    }
    const ant = new THREE.Mesh(new THREE.CylinderGeometry(0.002, 0.002, H * 1.1, 4), mat('#111111'));
    ant.position.set(W * 0.25, bodyY + bodyH / 2 + H * 0.55, L * 0.25); root.add(ant);
  }
  root.userData = { kind: 'car', wheels, wheelR };
  return root;
}

function animateCar(obj, st) {
  const u = obj.userData;
  for (const w of u.wheels) w.rotation.x = -st.dist / u.wheelR;
}

// ---------------------------------------------------------------- static scenery
function buildTree(t, r) {
  const g = new THREE.Group();
  const h = t.height || 4.0, rad = t.radius || 1.2;
  const trunkH = h - rad * 1.2;
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.1 * rad * 0.9, 0.14 * rad, trunkH, 10), mat('#5b4129', 0.95));
  trunk.position.y = trunkH / 2; trunk.castShadow = true; g.add(trunk);
  const leafMats = ['#2f6b2a', '#3b7a31', '#4b8a39', '#2a5e26'].map((c) => mat(c, 0.95, { flatShading: true }));
  for (let i = 0; i < 7; i++) {
    const s = rad * (0.45 + r() * 0.35);
    const m = new THREE.Mesh(new THREE.IcosahedronGeometry(s, 1), leafMats[i % leafMats.length]);
    const a = r() * Math.PI * 2, d = rad * 0.45 * r();
    m.position.set(Math.cos(a) * d, h - rad + (r() - 0.3) * rad * 0.7, Math.sin(a) * d);
    m.castShadow = true; g.add(m);
  }
  g.position.copy(enu(t.x, t.y, 0));
  return g;
}

function buildWall(w) {
  const len = Math.hypot(w.x2 - w.x1, w.y2 - w.y1);
  const h = w.height || 2.5;
  const isHouse = w.kind === 'house';
  const thick = isHouse ? 0.3 : 0.06;
  const tex = isHouse ? sidingTexture(w.color || '#d9cdb4') : plankTexture(w.color || '#8b6a48');
  tex.repeat.set(len / (isHouse ? 3 : 2.5), 1);
  const m = new THREE.Mesh(new THREE.BoxGeometry(len, h, thick), mat('#ffffff', 0.9, { map: tex }));
  const g = new THREE.Group();
  m.position.y = h / 2; m.castShadow = true; m.receiveShadow = true; g.add(m);
  if (isHouse) {     // windows, a door, and a roof edge
    const winM = new THREE.MeshPhysicalMaterial({ color: '#2b3a48', roughness: 0.1, metalness: 0.4 });
    const frameM = mat('#f4f1ea', 0.6);
    const n = Math.max(1, Math.floor(len / 3.2));
    for (let i = 0; i < n; i++) {
      const x = -len / 2 + (i + 0.5) * (len / n);
      if (Math.abs(x) < 0.8) {
        const door = new THREE.Mesh(new THREE.BoxGeometry(0.95, 2.05, 0.05), mat('#6b3e26', 0.6));
        door.position.set(x, 1.03, -thick / 2 - 0.02); g.add(door);
        continue;
      }
      const fr = new THREE.Mesh(new THREE.BoxGeometry(1.25, 1.15, 0.04), frameM);
      fr.position.set(x, 1.65, -thick / 2 - 0.02); g.add(fr);
      const wi = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.0, 0.05), winM);
      wi.position.set(x, 1.65, -thick / 2 - 0.03); g.add(wi);
    }
    const eave = new THREE.Mesh(new THREE.BoxGeometry(len + 0.4, 0.18, 0.7), mat('#5a5048', 0.8));
    eave.position.set(0, h + 0.05, -0.1); eave.castShadow = true; g.add(eave);
  } else {          // fence posts and a top rail
    const postM = mat(w.color || '#8b6a48', 0.9);
    for (let d = 0; d <= len + 0.01; d += 2.0) {
      const p = new THREE.Mesh(new THREE.BoxGeometry(0.1, h + 0.08, 0.1), postM);
      p.position.set(-len / 2 + d, (h + 0.08) / 2, -0.05); p.castShadow = true; g.add(p);
    }
  }
  // orient: the wall's local x runs from (x1,y1) to (x2,y2)
  const mid = enu((w.x1 + w.x2) / 2, (w.y1 + w.y2) / 2, 0);
  g.position.copy(mid);
  g.rotation.y = Math.atan2(w.y2 - w.y1, w.x2 - w.x1);
  // face the inside of the arena (toward the origin) for windows and doors
  const nx = -(w.y2 - w.y1), ny = (w.x2 - w.x1);
  const toOrigin = -((w.x1 + w.x2) / 2) * nx - ((w.y1 + w.y2) / 2) * ny;
  if (toOrigin < 0) g.rotation.y += Math.PI;
  return g;
}

function buildProp(p) {
  const g = new THREE.Group();
  if (p.kind === 'bench') {
    const wood = mat('#8a5a34', 0.8), iron = mat('#2b2b2b', 0.5, { metalness: 0.5 });
    const seat = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.06, 0.42), wood); seat.position.y = 0.45; g.add(seat);
    const back = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.35, 0.05), wood); back.position.set(0, 0.75, 0.19); g.add(back);
    for (const sx of [-0.65, 0.65]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.45, 0.42), iron); leg.position.set(sx, 0.22, 0); g.add(leg);
    }
  } else if (p.kind === 'planter') {
    const box = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.45, 0.45), mat('#7a7068', 0.9)); box.position.y = 0.22; g.add(box);
    const r = rng(Math.floor(p.x * 100 + p.y * 7 + 99));
    for (let i = 0; i < 5; i++) {
      const b = new THREE.Mesh(new THREE.IcosahedronGeometry(0.18 + r() * 0.1, 1), mat(r() < 0.5 ? '#3f7d34' : '#c9566e', 0.9, { flatShading: true }));
      b.position.set(-0.4 + i * 0.2, 0.55 + r() * 0.08, (r() - 0.5) * 0.15); g.add(b);
    }
  } else if (p.kind === 'cone') {
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.32, 16), mat('#f26b1d', 0.6)); cone.position.y = 0.16; g.add(cone);
    const band = new THREE.Mesh(new THREE.CylinderGeometry(0.066, 0.083, 0.06, 16), mat('#f2f2f2', 0.5)); band.position.y = 0.17; g.add(band);
    const base = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.02, 0.26), mat('#222222')); base.position.y = 0.01; g.add(base);
  } else if (p.kind === 'sign') {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.06, 1.1, 0.06), mat('#666666', 0.6)); post.position.y = 0.55; g.add(post);
    const tex = canvasTexture(512, 192, (c, w, h) => {
      c.fillStyle = '#f5f5f0'; c.fillRect(0, 0, w, h);
      c.strokeStyle = '#c62828'; c.lineWidth = 14; c.strokeRect(8, 8, w - 16, h - 16);
      c.fillStyle = '#1a1a1a'; c.font = 'bold 62px Helvetica, Arial'; c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText(p.text || 'PRIVATE', w / 2, h / 2);
    });
    const board = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.3, 0.03),
      [mat('#dddddd'), mat('#dddddd'), mat('#dddddd'), mat('#dddddd'), mat('#ffffff', 0.7, { map: tex }), mat('#dddddd')]);
    board.position.y = 1.0; g.add(board);
  }
  g.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  g.position.copy(enu(p.x, p.y, 0));
  g.rotation.y = -THREE.MathUtils.degToRad(p.heading_deg || 0) + Math.PI;   // fronts face the heading
  return g;
}

function buildTrack(tr) {
  // flat elliptical ring of asphalt with white edge lines
  const g = new THREE.Group();
  const shape = (rx, ry) => {
    const pts = [];
    for (let i = 0; i <= 96; i++) { const a = (i / 96) * Math.PI * 2; pts.push(new THREE.Vector2(rx * Math.cos(a), ry * Math.sin(a))); }
    return pts;
  };
  const w = tr.width || 0.5;
  const outer = new THREE.Shape(shape(tr.rx + w / 2, tr.ry + w / 2));
  outer.holes.push(new THREE.Path(shape(tr.rx - w / 2, tr.ry - w / 2)));
  const ring = new THREE.Mesh(new THREE.ShapeGeometry(outer, 96), mat('#ffffff', 0.95, { map: asphaltTexture() }));
  ring.rotation.x = -Math.PI / 2; ring.position.y = 0.004; ring.receiveShadow = true; g.add(ring);
  for (const off of [w / 2 - 0.03, -w / 2 + 0.03]) {
    const line = new THREE.Shape(shape(tr.rx + off + 0.02, tr.ry + off + 0.02));
    line.holes.push(new THREE.Path(shape(tr.rx + off - 0.02, tr.ry + off - 0.02)));
    const lm = new THREE.Mesh(new THREE.ShapeGeometry(line, 96), mat('#f2f2f2', 0.7));
    lm.rotation.x = -Math.PI / 2; lm.position.y = 0.006; g.add(lm);
  }
  // shape is drawn in (x, y) of ENU; rotation.x = -pi/2 maps shape y to -z = +north
  g.position.copy(enu(tr.cx, tr.cy, 0));
  return g;
}

export function buildDrone() {
  const g = new THREE.Group();
  const shell = mat('#e8e8e8', 0.5), dark = mat('#2a2a2a', 0.6);
  const body = new THREE.Mesh(new RoundedBoxGeometry(0.1, 0.04, 0.14, 3, 0.015), shell); g.add(body);
  for (const [sx, sz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.012, 0.12), dark);
    arm.position.set(sx * 0.05, 0, sz * 0.05); arm.rotation.y = sx * sz * Math.PI / 4; g.add(arm);
    const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.004, 20),
      new THREE.MeshStandardMaterial({ color: '#333333', transparent: true, opacity: 0.35 }));
    rotor.position.set(sx * 0.09, 0.02, sz * 0.09); g.add(rotor);
  }
  const lens = new THREE.Mesh(new THREE.SphereGeometry(0.012, 10, 8), mat('#111111', 0.2));
  lens.position.set(0, -0.005, -0.072); g.add(lens);
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.008, 8, 6), mat('#ff2222', 0.3, { emissive: '#ff0000', emissiveIntensity: 1 }));
  led.position.set(0, 0.022, 0.06); g.add(led);
  // Layer 1 only: the chase camera sees the drone, the drone's own camera does not.
  g.traverse((o) => { if (o.isMesh) o.castShadow = true; o.layers.set(DRONE_LAYER); });
  g.scale.setScalar(1.6);          // the X69 is about 25 cm across; drawn a bit larger for the chase view
  return g;
}

// ---------------------------------------------------------------- the world
export class World {
  constructor() {
    this.scene = new THREE.Scene();
    this.actors = new Map();
    this.drone = null;
    this.desc = null;
  }

  build(desc) {
    this.desc = desc;
    const scene = this.scene;
    scene.clear();
    this.actors.clear();
    const world = desc.world || {};
    const ground = world.ground || {};
    const size = ground.size_m || 60;
    scene.background = skyTexture();
    scene.fog = new THREE.Fog('#c9d8e6', 28, 90);
    const hemi = new THREE.HemisphereLight('#cfe3ff', '#4a5a38', 0.9);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight('#fff3e0', 2.4);
    const az = THREE.MathUtils.degToRad((world.sun || {}).azimuth_deg ?? 135);
    const el = THREE.MathUtils.degToRad((world.sun || {}).elevation_deg ?? 50);
    sun.position.copy(enu(Math.sin(az) * Math.cos(el) * 30, Math.cos(az) * Math.cos(el) * 30, Math.sin(el) * 30));
    sun.target.position.copy(enu(0, 4, 0));
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera; sc.left = -14; sc.right = 14; sc.top = 14; sc.bottom = -14; sc.near = 1; sc.far = 80;
    sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02;
    scene.add(sun); scene.add(sun.target);

    const groundTex = grassTexture(size);
    const gm = new THREE.Mesh(new THREE.PlaneGeometry(size, size), mat('#ffffff', 1.0, { map: groundTex }));
    gm.rotation.x = -Math.PI / 2; gm.receiveShadow = true; scene.add(gm);
    // texture quality < 1 means a smoother, lower-contrast ground (harder for optical flow)
    const tq = ground.texture ?? 1.0;
    if (tq < 0.5) gm.material.color.set('#9fb08a');

    const r = rng(42);
    for (const w of world.walls || []) scene.add(buildWall(w));
    for (const t of world.trees || []) scene.add(buildTree(t, r));
    for (const p of world.props || []) scene.add(buildProp(p));
    if (world.track) scene.add(buildTrack(world.track));
    for (const a of desc.actors || []) {
      const obj = a.kind === 'car' ? buildCar(a) : buildPerson(a);
      obj.traverse((o) => { if (o.isMesh) o.receiveShadow = true; });
      scene.add(obj);
      this.actors.set(a.id, obj);
    }
    this.drone = buildDrone();
    scene.add(this.drone);
  }

  update(pose) {
    for (const st of pose.actors || []) {
      const obj = this.actors.get(st.id);
      if (!obj) continue;
      obj.position.copy(enu(st.x, st.y, 0));
      obj.rotation.y = -st.heading;
      if (obj.userData.kind === 'person') animatePerson(obj, st); else animateCar(obj, st);
    }
    if (this.drone && pose.drone) {
      const d = pose.drone;
      this.drone.position.copy(enu(d.x, d.y, Math.max(0.03, d.z + 0.03)));
      this.drone.rotation.order = 'YXZ';
      this.drone.rotation.set(d.pitch, -d.yaw, -d.roll);
    }
  }
}

export function vfovDeg(hfovDeg, w, h) {
  const hf = THREE.MathUtils.degToRad(hfovDeg);
  return THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(hf / 2) * h / w));
}

export function applyDroneCamera(cam, c) {
  cam.position.copy(enu(c.x, c.y, c.z));
  cam.rotation.order = 'YXZ';
  cam.rotation.set(c.pitch, -c.yaw, -c.roll);
}

export function makeRenderer(canvas, w, h) {
  const r = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
  r.setPixelRatio(1);
  r.setSize(w, h, false);
  r.shadowMap.enabled = true;
  r.shadowMap.type = THREE.PCFSoftShadowMap;
  r.toneMapping = THREE.ACESFilmicToneMapping;
  r.toneMappingExposure = 1.0;
  r.outputColorSpace = THREE.SRGBColorSpace;
  return r;
}

export function webglInfo(renderer) {
  const gl = renderer.getContext();
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  return {
    renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    vendor: ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
  };
}

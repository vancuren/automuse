/**
 * Where the operator UI finds the drone backend and the twin (new in this fork).
 *
 * Everything comes from Vite env vars that are read when the dev server starts or when
 * `npm run build` runs, so one build always talks to exactly one backend (shown in the footer).
 *
 *   VITE_BACKEND_URL     http(s) base of drone-backend       default http://localhost:8000
 *   VITE_TWIN_CHASE_URL  chase view shown in twin mode        default http://localhost:8766/chase
 *   VITE_PACK_WARN_S     per-pack flight time that warns      default 300 (5:00)
 *   VITE_PACK_LAND_S     per-pack flight time that auto-lands default 390 (6:30)
 *   VITE_PACKS           pack ids offered for pack swap       default "1,2"
 *   VITE_AXES_RELEASE_MESSAGE  "1": when stick input stops (key up, blur, hidden), send ONE
 *                        all-zero axes message so the governor releases at once instead of
 *                        after its 0.25 s staleness timeout. Default off: axes only while nonzero.
 */
const env = import.meta.env;

function stripTrailingSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

/** http://host:port of drone-backend (no trailing slash). */
export const BACKEND_URL = stripTrailingSlash(env.VITE_BACKEND_URL || "http://localhost:8000");

/** Same host as BACKEND_URL with ws:// (or wss:// for https://). */
export const BACKEND_WS_URL = BACKEND_URL.replace(/^http/, "ws");

export const TWIN_CHASE_URL = env.VITE_TWIN_CHASE_URL || "http://localhost:8766/chase";

export const PACK_WARN_S = Number(env.VITE_PACK_WARN_S || 300);
export const PACK_LAND_S = Number(env.VITE_PACK_LAND_S || 390);
export const AXES_RELEASE_MESSAGE = env.VITE_AXES_RELEASE_MESSAGE === "1";

export const PACK_IDS = (env.VITE_PACKS || "1,2")
  .split(",")
  .map((s: string) => s.trim())
  .filter(Boolean);

/** Every backend endpoint the UI uses, in one place. */
export const URLS = {
  control: `${BACKEND_WS_URL}/ws`,                 // axes, cmd (and hb from the worker)
  telemetry: `${BACKEND_WS_URL}/ws/telemetry`,     // Telemetry at 10 Hz
  agent: `${BACKEND_WS_URL}/ws/agent?role=ui`,     // AgentEvent, Candidates in; ptt and Select out
  overlays: `${BACKEND_WS_URL}/ws/overlays`,       // upstream rect overlays (TestPage only)
  mjpeg: `${BACKEND_URL}/mjpeg`,
  command: `${BACKEND_URL}/command`,               // HTTP fallback for LAND and E-STOP
  capabilities: `${BACKEND_URL}/capabilities`,
  plugins: `${BACKEND_URL}/plugins`,
} as const;

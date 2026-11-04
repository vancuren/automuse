# Operator UI (forked from turbodrone's frontend, Apache-2.0)

The dashboard for the voice-commanded follow drone. The full write-up is in [`docs/ws/ui.md`](../../docs/ws/ui.md): run commands, the operator cheat sheet, the wire behavior, test results, and known gaps.

```bash
source ../../env.sh && npm ci                       # once; installs into ./node_modules
VITE_BACKEND_URL=http://localhost:8000 npm run build && npx vite preview   # flights: http://127.0.0.1:5173
npm run dev                                          # development against http://localhost:8000
npm run mock & npm run dev:mock                      # no backend: mock on :8600, UI on http://127.0.0.1:5273
```

The safety keys are:
- **Esc**: E-STOP;
- **L**: LAND;
- **hold Space**: push-to-talk;
- **1-9**: pick a look-alike;
- **W/A/S/D and arrows**: human sticks, sent only while a key is held.

TAKEOFF is a button only. In REAL mode, type ARM first.

Kept from turbodrone: the MJPEG video feed, keyboard and gamepad input, and the overall app shell. New: everything in `src/lib`, `src/workers`, the hooks `useAgentLink`, `useHeartbeat`, `useHotkeys` and `useTelemetry`, the dashboard components, and `mock/`.

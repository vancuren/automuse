import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Bound to loopback only: the operator UI must never be reachable from the drone's Wi-Fi.
// The backend URL is VITE_BACKEND_URL (see src/config.ts), read when vite starts.
export default defineConfig({
  plugins: [react()],
  server: { host: '127.0.0.1', port: 5173, strictPort: true },
  preview: { host: '127.0.0.1', port: 5173, strictPort: true },
  worker: { format: 'es' },
})

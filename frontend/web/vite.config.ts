import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const API = process.env.VITE_API_BASE || 'http://127.0.0.1:8032'
const DEV_HOST = process.env.VITE_DEV_HOST || '127.0.0.1'
const DEV_PORT = Number(process.env.VITE_DEV_PORT || '5173')
const PREVIEW_PORT = Number(process.env.VITE_PREVIEW_PORT || DEV_PORT)

export default defineConfig({
  plugins: [react()],
  server: {
    host: DEV_HOST,
    port: DEV_PORT,
    strictPort: true,
    proxy: {
      '/api': {
        target: API,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  preview: {
    host: DEV_HOST,
    port: PREVIEW_PORT,
  },
})

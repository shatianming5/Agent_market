import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const API = process.env.VITE_API_BASE || 'http://127.0.0.1:8032'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: API,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})

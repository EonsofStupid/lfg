import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/',
  server: {
    host: true,
    port: 5174,
    proxy: { '/api': 'http://127.0.0.1:4317' },
    watch: {
      // fs.watch's native change events are unreliable over this project's
      // SMB-mounted network share and crash the process with ECONNRESET;
      // polling avoids relying on OS-level watch notifications.
      usePolling: true,
      interval: 500
    }
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts'
  }
})

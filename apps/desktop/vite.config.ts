import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Everything is bundled. No CDN, no remote font, no runtime fetch to anywhere
 * outside the machine - a security appliance that pulls assets from the Internet
 * at render time is not offline, however offline its backend is.
 */
export default defineConfig({
  plugins: [react()],
  // Relative base so the bundle works when loaded from the filesystem by Tauri.
  base: './',
  build: {
    outDir: 'dist',
    target: 'es2022',
    assetsInlineLimit: 0,
    sourcemap: true,
  },
  server: {
    // Loopback only. The dev server is not a LAN service.
    host: '127.0.0.1',
    port: 5183,
    strictPort: true,
  },
  clearScreen: false,
});

import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

// GitHub Pages project site: https://haixuantao.github.io/zealot/
// `public/` (wasm demos, bench pages, checkpoints) is copied verbatim into
// dist/, so those paths stay exactly what the demos expect.
export default defineConfig({
  base: '/zealot/',
  plugins: [react()],
  server: {port: 3000},
});

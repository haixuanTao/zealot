import puppeteer from 'puppeteer-core';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const URL = process.env.DEMO_URL ?? 'http://localhost:3123/demos/g1_web/';

// Headed by default: headless Chrome throttles rAF-driven wasm apps. HEADLESS=1
// to force headless. Screenshot capture runs off the compositor, so it works
// even while the page's main thread is blocked (e.g. during the long wasm
// pipeline-compile); DOM state polling is best-effort with a short timeout.
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: process.env.HEADLESS === '1' ? 'new' : false,
  protocolTimeout: 600000,
  args: [
    '--enable-unsafe-webgpu',
    '--use-angle=metal',
    '--no-first-run',
    '--window-size=1280,900',
  ],
});

const page = await browser.newPage();
await page.setViewport({width: 1280, height: 900});
page.on('console', (msg) => console.log(`[console:${msg.type()}] ${msg.text().slice(0, 2500)}`));
page.on('pageerror', (err) => console.log(`[pageerror] ${String(err).slice(0, 500)}`));

await page.goto(URL, {waitUntil: 'load', timeout: 60000});
console.log('[nav] page loaded');

const withTimeout = (p, ms) =>
  Promise.race([p, new Promise((r) => setTimeout(() => r('(main thread busy)'), ms))]);

let prev = 0;
for (const t of [15, 45, 90, 150, 240]) {
  await new Promise((r) => setTimeout(r, (t - prev) * 1000));
  prev = t;
  await page.screenshot({path: `/tmp/demo_t${t}.png`});
  const state = await withTimeout(
    page.evaluate(() => {
      const loading = document.getElementById('loading');
      const canvas = document.querySelector('canvas');
      return JSON.stringify({
        loadingVisible: loading ? getComputedStyle(loading).display !== 'none' : null,
        hasCanvas: !!canvas,
        canvasSize: canvas ? [canvas.width, canvas.height] : null,
      });
    }),
    5000,
  );
  console.log(`[t=${t}s] screenshot saved; state: ${state}`);
}

await browser.close();

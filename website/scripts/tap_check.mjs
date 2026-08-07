// Tap-to-walk check: load the demo, let it boot, tap a ground point away
// from the pack, and screenshot the approach so target semantics can be
// eyeballed (do the robots converge on the MARKER, not lane-shifted copies?).
import puppeteer from 'puppeteer-core';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const URL = process.env.DEMO_URL ?? 'http://localhost:3123/zealot/demos/g1_web/?n=4';
const TAP = (process.env.TAP ?? '900,420').split(',').map(Number);

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: process.env.HEADLESS === '1' ? 'new' : false,
  protocolTimeout: 600000,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-first-run', '--window-size=1280,900'],
});
const page = (await browser.pages())[0] ?? (await browser.newPage());
page.on('console', (m) => {
  if (/nav:/.test(m.text())) console.log('[console]', m.text());
});
await page.setViewport({width: 1280, height: 900});
await page.goto(URL, {waitUntil: 'load', timeout: 120000});
console.log('[nav] page loaded');

const sleep = (s) => new Promise((r) => setTimeout(r, s * 1000));
await sleep(25); // boot + spawn settle
await page.screenshot({path: '/tmp/tap_t0.png'});
// A tap = press + release with no drag (drag orbits the camera).
await page.mouse.click(TAP[0], TAP[1]);
console.log(`[nav] tapped ${TAP[0]},${TAP[1]}`);
for (const t of [3, 10, 20, 35]) {
  await sleep(t === 3 ? 3 : t - [3, 10, 20, 35][[3, 10, 20, 35].indexOf(t) - 1]);
  await page.screenshot({path: `/tmp/tap_t${t}.png`});
  console.log(`[nav] shot t=${t}s`);
}
await browser.close();

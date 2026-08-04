// Perf matrix for the nexus demo. Reports fps / RT / SPEED / falls — a
// frozen sim (spd 0) is a FAILURE, whatever the other numbers say.
import puppeteer from 'puppeteer-core';
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const BASE = (process.env.SITE_URL ?? 'http://localhost:3410/zealot/') + 'demos/g1_terrain_web/index.html';
const CASES = (process.env.CASES ?? 'n=3&lvl=4&slope=2|n=1&lvl=4&slope=2').split('|');
const browser = await puppeteer.launch({
  executablePath: CHROME, headless: false, protocolTimeout: 300000,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-first-run', '--window-size=1400,950'],
});
for (const q of CASES) {
  const page = await browser.newPage();
  await page.setViewport({width: 1400, height: 950});
  await page.goto(`${BASE}?${q}`, {waitUntil: 'load', timeout: 120000});
  const samples = [];
  for (let i = 0; i < 60 && samples.length < 25; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const m = (await page.evaluate(() => document.title))
      .match(/falls=(\d+).*spd=([\d.]+) cmd=([\d.]+) rt=(\d+)% fps=(\d+)/);
    if (m) samples.push({falls: +m[1], spd: +m[2], rt: +m[4], fps: +m[5]});
  }
  const s = samples.slice(5);
  if (!s.length) { console.log(`${q}: never started`); await page.close(); continue; }
  const avg = (k) => (s.reduce((a, x) => a + x[k], 0) / s.length).toFixed(1);
  const spd = (s.reduce((a, x) => a + x.spd, 0) / s.length).toFixed(2);
  const frozen = +spd < 0.05 ? '  *** FROZEN ***' : '';
  console.log(`${q.padEnd(28)} fps ${avg('fps')}  rt ${avg('rt')}%  spd ${spd}  falls ${s.at(-1).falls - s[0].falls}/${s.length}s${frozen}`);
  await page.close();
}
await browser.close();

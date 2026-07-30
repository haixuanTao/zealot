// Tap-to-bump check for the nexus wasm demo. Its HUD is canvas-drawn, so we
// screenshot instead of reading text — and we must run FOCUSED, since a
// background tab throttles rAF and the demo loop barely ticks.
import puppeteer from 'puppeteer-core';
const URL = process.env.URL || 'http://localhost:3330/zealot/demos/g1_terrain_web/?n=1&lvl=4&slope=2';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false, protocolTimeout: 600000, args: ['--no-first-run', '--window-size=900,700'],
});
const page = await browser.newPage();
await page.setViewport({width: 900, height: 700});
await page.goto(URL, {waitUntil: 'load', timeout: 120000});
await new Promise((r) => setTimeout(r, 60000));   // wasm compile
await page.bringToFront();
await page.mouse.click(600, 450);
await page.screenshot({path: '/tmp/tap_0.png'});
for (const n of [1, 2]) {
  await page.keyboard.press('ArrowUp');
  await new Promise((r) => setTimeout(r, 4000));
  await page.screenshot({path: `/tmp/tap_${n}.png`});
}
await page.keyboard.press('Space');
await new Promise((r) => setTimeout(r, 4000));
await page.screenshot({path: '/tmp/tap_stop.png'});
await browser.close();
console.log('wrote /tmp/tap_{0,1,2,stop}.png');

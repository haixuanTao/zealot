// Verify the demo's live driving: hold ArrowUp in a FOCUSED window (rAF is
// throttled in background tabs, so the sim barely ticks otherwise) and
// compare the robot's on-screen position before and after.
import puppeteer from 'puppeteer-core';

const URL = process.env.URL || 'http://localhost:3240/zealot/demos/g1_terrain_web/?n=1&lvl=4&slope=5';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false,
  protocolTimeout: 600000,
  args: ['--no-first-run', '--window-size=900,700'],
});
const page = await browser.newPage();
await page.setViewport({width: 900, height: 700});
await page.goto(URL, {waitUntil: 'load', timeout: 120000});

// Wait for the wasm to compile and the first frames to render.
await new Promise((r) => setTimeout(r, 60000));
await page.screenshot({path: '/tmp/drive_before.png'});

await page.bringToFront();
await page.mouse.click(600, 400);            // focus the canvas
await page.keyboard.down('ArrowUp');
await new Promise((r) => setTimeout(r, 8000));
await page.keyboard.up('ArrowUp');
await page.screenshot({path: '/tmp/drive_after.png'});

await new Promise((r) => setTimeout(r, 3000));
await page.screenshot({path: '/tmp/drive_released.png'});
await browser.close();
console.log('wrote /tmp/drive_{before,after,released}.png');

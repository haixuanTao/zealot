// Tap-to-bump check for the sim2sim pages: each arrow press should add 0.2 and
// latch; space should zero. The HUD carries the command, so read it back.
import puppeteer from 'puppeteer-core';
const URL = process.env.URL;
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false, protocolTimeout: 600000, args: ['--no-first-run', '--window-size=900,700'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('PAGE ERROR:', e.message));
await page.goto(URL, {waitUntil: 'load', timeout: 120000});
// Wait until the HUD actually reports a command (wasm compile can take a while).
await page.waitForFunction(() => /cmd [-\d.]+/.test(document.getElementById('hud')?.textContent || ''), {timeout: 180000});
const cmd = () => page.$eval('#hud', (e) => e.textContent.match(/cmd [-\d.]+/)?.[0] ?? '?');
await page.bringToFront();
await page.mouse.click(450, 400);
console.log('start:      ', await cmd());
for (const n of [1, 2, 3]) {
  await page.keyboard.press('ArrowUp');
  await new Promise((r) => setTimeout(r, 1500));
  console.log(`after tap ${n}:`, await cmd());
}
await page.keyboard.press('ArrowDown');
await new Promise((r) => setTimeout(r, 1500));
console.log('after down: ', await cmd());
await page.keyboard.press('Space');
await new Promise((r) => setTimeout(r, 1500));
console.log('after space:', await cmd());
await browser.close();

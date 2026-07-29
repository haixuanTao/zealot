import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false, protocolTimeout: 120000,
  args: ['--no-first-run', '--window-size=1280,900'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 300)));
page.on('console', (m) => console.log('[console]', m.text().slice(0, 300)));
await page.goto('http://localhost:3123/bench/three_rapier_bench?auto=1&ckpt=g1_v10', {waitUntil: 'load'});
try { await page.waitForFunction(() => document.getElementById('hud').textContent.includes('DONE'), {timeout: 90000}); } catch (e) { console.log('TIMEOUT'); }
console.log('HUD:', await page.evaluate(() => document.getElementById('hud').textContent));
await page.screenshot({path: '/tmp/three_dbg.png'});
await browser.close();

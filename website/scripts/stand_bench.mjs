import puppeteer from 'puppeteer-core';
const url = process.argv[2];
const secs = parseInt(process.argv[3] || '30');
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false, protocolTimeout: 120000,
  args: ['--no-first-run', '--window-size=1280,900'],
});
const page = await browser.newPage();
await page.goto(url, {waitUntil: 'load'});
await new Promise(r => setTimeout(r, secs * 1000));
console.log('HUD:', await page.evaluate(() => document.getElementById('hud').textContent));
await page.screenshot({path: '/tmp/stand_bench.png'});
await browser.close();

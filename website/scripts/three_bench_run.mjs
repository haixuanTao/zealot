import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false,
  protocolTimeout: 600000,
  args: ['--no-first-run', '--window-size=1280,900'],
});
const page = await browser.newPage();
await page.setViewport({width: 1280, height: 900});
page.on('pageerror', (e) => console.log('[pageerror] ' + e));
page.on('console', (m) => { const t = m.text(); if (t.startsWith('[bench]')) console.log(t); });
await page.goto('http://localhost:3123/bench/three_rapier_bench.html?auto=1', {waitUntil: 'load', timeout: 60000});
await page.waitForFunction(() => document.getElementById('hud').textContent.includes('DONE'), {timeout: 300000});
await page.screenshot({path: '/tmp/three_bench.png'});
await browser.close();

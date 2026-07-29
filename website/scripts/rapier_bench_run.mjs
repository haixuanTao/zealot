import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false,
  protocolTimeout: 600000,
  args: ['--no-first-run', '--window-size=900,700'],
});
const page = await browser.newPage();
page.on('console', (m) => { const t = m.text(); if (t.startsWith('[bench]')) console.log(t); });
await page.goto('http://localhost:3123/bench/rapier_bench.html', {waitUntil: 'load', timeout: 60000});
await page.waitForFunction(() => document.getElementById('out').textContent.includes('DONE'), {timeout: 300000});
await browser.close();

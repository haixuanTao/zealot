// Read the demo's diagnostic title over time (Chrome baseline).
import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false, protocolTimeout: 600000, args: ['--no-first-run', '--window-size=800,600'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('PAGE ERROR:', e.message));
page.on('console', (m) => { const t = m.text(); if (/error|panic|warn/i.test(t)) console.log('CONSOLE:', t.slice(0, 160)); });
await page.goto(process.env.URL, {waitUntil: 'load', timeout: 120000});
await page.bringToFront();
for (let i = 0; i < 6; i++) {
  await new Promise((r) => setTimeout(r, 12000));
  console.log(`t=${(i + 1) * 12}s`, await page.title());
}
await browser.close();

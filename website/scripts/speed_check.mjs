// Report the HUD's mean speed for a given checkpoint, so v21 and v24 can be
// compared on identical terrain/commands.
import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false, protocolTimeout: 600000, args: ['--no-first-run', '--window-size=900,700'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('PAGE ERROR:', e.message));
await page.goto(process.env.URL, {waitUntil: 'load', timeout: 120000});
await page.waitForFunction(() => /mean speed/.test(document.getElementById('hud')?.textContent || ''), {timeout: 180000});
await page.bringToFront();
for (const t of [10, 20, 30]) {
  await new Promise((r) => setTimeout(r, 10000));
  const line = await page.$eval('#hud', (e) => {
    const txt = e.textContent;
    return [txt.match(/mean speed: [\d.]+ m\/s \(cmd [-\d.]+\)/)?.[0], txt.match(/falls: \d+/)?.[0]].join('  ');
  });
  console.log(`t=${t}s`, line);
}
await browser.close();

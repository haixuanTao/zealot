// Which engine the site opens on, per browser identity. Safari, Firefox and
// iOS cannot run the WebGPU demo, so the front page has to land them on a CPU
// engine instead — this checks that it does, and that the nexus tab is still
// what Chrome gets. UA spoofing tests OUR sniffing, not WebKit itself.
//
//   node scripts/check-browser-fallback.mjs          (against the local preview)
//   SITE_URL=https://haixuantao.github.io/zealot/ node scripts/check-browser-fallback.mjs
import puppeteer from 'puppeteer-core';
import {existsSync} from 'node:fs';
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const URL = process.env.SITE_URL ?? 'http://localhost:3410/zealot/';

const UAS = [
  ['Chrome (desktop)', null, null],
  ['Safari 17 (macOS)',
   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15', 0],
  ['Firefox (desktop)',
   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:127.0) Gecko/20100101 Firefox/127.0', 0],
  ['Safari (iPhone)',
   'Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1', 5],
  ['Chrome (iPad, iPadOS UA)',
   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/126 Mobile/15E148 Safari/604.1', 5],
];

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: 'new', protocolTimeout: 120000,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-first-run'],
});
for (const [name, ua, touch] of UAS) {
  const page = await browser.newPage();
  if (ua) {
    await page.setUserAgent(ua);
    await page.evaluateOnNewDocument((t) => {
      Object.defineProperty(navigator, 'maxTouchPoints', {get: () => t});
    }, touch);
  }
  await page.goto(URL, {waitUntil: 'load', timeout: 60000});
  await new Promise((r) => setTimeout(r, 1500));
  const s = await page.evaluate(() => ({
    tab: document.querySelector('.tabActive')?.textContent ?? '(none)',
    notice: document.querySelector('.notice')?.textContent ?? null,
    warning: document.querySelector('.warning')?.textContent ?? null,
    frame: (document.querySelector('iframe')?.src ?? '').split('/zealot/')[1] ?? '(none)',
  }));
  console.log(`${name.padEnd(26)} tab=${s.tab.padEnd(18)} frame=${s.frame.split('?')[0]}`);
  if (s.notice) console.log(`${' '.repeat(28)}notice: ${s.notice}`);
  if (s.warning) console.log(`${' '.repeat(28)}warning: ${s.warning.slice(0, 80)}…`);
  await page.close();
}
await browser.close();

// Spoofed UAs only prove OUR sniffing. Firefox is the one browser here that
// can be driven for real, and it is where the fallback was reported broken —
// so run it for real too when it is installed.
const FF = '/Applications/Firefox.app/Contents/MacOS/firefox';
if (existsSync(FF)) {
  const ff = await puppeteer.launch({
    browser: 'firefox',
    executablePath: FF,
    headless: false,
    protocolTimeout: 120000,
  });
  const page = await ff.newPage();
  await page.goto(URL, {waitUntil: 'load', timeout: 90000});
  await new Promise((r) => setTimeout(r, 4000));
  const s = await page.evaluate(() => ({
    tab: document.querySelector('.tabActive')?.textContent ?? '(none)',
    script: document.querySelector('script[type=module][src]')?.src.split('/').pop(),
    notice: !!document.querySelector('.notice'),
  }));
  console.log(
    `${'Firefox (REAL)'.padEnd(26)} tab=${s.tab.padEnd(18)} bundle=${s.script} notice=${s.notice}`,
  );
  await ff.close();
} else {
  console.log('Firefox not installed — skipped the real-browser pass');
}

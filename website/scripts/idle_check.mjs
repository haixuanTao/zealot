// No input at all: what command does the demo start with?
import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false, protocolTimeout: 600000, args: ['--no-first-run', '--window-size=900,700'],
});
const page = await browser.newPage();
await page.setViewport({width: 900, height: 700});
page.on('pageerror', (e) => console.log('PAGE ERROR:', e.message));
await page.goto(process.env.URL, {waitUntil: 'load', timeout: 120000});
await new Promise((r) => setTimeout(r, 55000));
await page.screenshot({path: '/tmp/idle_a.png'});
await new Promise((r) => setTimeout(r, 15000));
await page.screenshot({path: '/tmp/idle_b.png'});
await browser.close();
console.log('wrote /tmp/idle_{a,b}.png (no clicks, no keys)');

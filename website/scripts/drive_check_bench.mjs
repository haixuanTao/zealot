// Same focused-window drive check, for the sim2sim pages: the HUD is DOM
// text there, so we can read the command back directly.
import puppeteer from 'puppeteer-core';
const URL = process.env.URL;
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false, protocolTimeout: 600000, args: ['--no-first-run', '--window-size=900,700'],
});
const page = await browser.newPage();
await page.goto(URL, {waitUntil: 'load', timeout: 120000});
await new Promise((r) => setTimeout(r, 12000));
const hud = () => page.$eval('#hud', (e) => e.textContent.match(/cmd [-\d.]+/)?.[0] ?? '?');
console.log('idle:   ', await hud());
await page.bringToFront();
await page.mouse.click(450, 400);
await page.keyboard.down('ArrowUp');
await new Promise((r) => setTimeout(r, 4000));
console.log('holding:', await hud());
await page.keyboard.up('ArrowUp');
await new Promise((r) => setTimeout(r, 3000));
console.log('released:', await hud());
await browser.close();

// Repro harness: run the MuJoCo bench page focused for N wall seconds and
// dump every [bench] console line (reset reasons included).
import puppeteer from 'puppeteer-core';
const URL = process.env.URL || 'http://localhost:3000/bench/three_mujoco_bench?n=3&lvl=4&slope=10';
const SECS = parseInt(process.env.SECS || '90');
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false,
  protocolTimeout: 600000,
  args: ['--no-first-run', '--window-size=900,700'],
});
const page = await browser.newPage();
page.on('console', (m) => { const t = m.text(); if (t.startsWith('[bench]')) console.log(t); });
await page.goto(URL, { waitUntil: 'load', timeout: 60000 });
await new Promise((r) => setTimeout(r, SECS * 1000));
await browser.close();

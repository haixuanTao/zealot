import puppeteer from 'puppeteer-core';
const url = process.argv[2] || 'http://localhost:3123/demos/g1_web/';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: false, protocolTimeout: 300000,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-first-run', '--window-size=1280,900'],
});
const page = await browser.newPage();
await page.evaluateOnNewDocument(() => {
  window.__gpu = { submits: 0, maps: 0, dispatches: 0, encoders: 0 };
  const oq = GPUQueue.prototype.submit;
  GPUQueue.prototype.submit = function (...a) { window.__gpu.submits++; return oq.apply(this, a); };
  const om = GPUBuffer.prototype.mapAsync;
  GPUBuffer.prototype.mapAsync = function (...a) { window.__gpu.maps++; return om.apply(this, a); };
  const od = GPUComputePassEncoder.prototype.dispatchWorkgroups;
  GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...a) { window.__gpu.dispatches++; return od.apply(this, a); };
  const oe = GPUDevice.prototype.createCommandEncoder;
  GPUDevice.prototype.createCommandEncoder = function (...a) { window.__gpu.encoders++; return oe.apply(this, a); };
  window.__gpu.passes = 0;
  const op = GPUCommandEncoder.prototype.beginComputePass;
  GPUCommandEncoder.prototype.beginComputePass = function (...a) { window.__gpu.passes++; return op.apply(this, a); };
  window.__gpu.pipelines = 0;
  const osp = GPUComputePassEncoder.prototype.setPipeline;
  GPUComputePassEncoder.prototype.setPipeline = function (...a) { window.__gpu.pipelines++; return osp.apply(this, a); };
});
await page.goto(url, {waitUntil: 'load', timeout: 60000});
await new Promise(r => setTimeout(r, 20000));            // let it settle
const a = await page.evaluate(() => ({...window.__gpu}));
await new Promise(r => setTimeout(r, 10000));            // 10 s window
const b = await page.evaluate(() => ({...window.__gpu}));
const d = Object.fromEntries(Object.keys(a).map(k => [k, (b[k] - a[k]) / 10]));
console.log('per second:', JSON.stringify(d));
console.log('per control step (assume 50/s):', JSON.stringify(
  Object.fromEntries(Object.entries(d).map(([k, v]) => [k, +(v / 50).toFixed(1)]))));
await browser.close();

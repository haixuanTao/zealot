// WGSL validation harness: which nexus kernel does the browser reject, and on
// which line of the WGSL Chrome actually saw?
//
// The demo's shader errors surface as `Error while parsing WGSL: :299:21 ...`
// — a line number into WGSL that exists only inside Chrome. naga generates it
// from the rust-gpu SPIR-V at load time, so there is no file on disk to open,
// and nothing in the message names the kernel. This harness hooks
// `createShaderModule` before the wasm runs, keeps every module's source, and
// asks each one for its compilation info; failing modules are written to
// OUT_DIR so `:299` becomes a line you can read next to the .rs it came from.
//
// Unlike the other harnesses this one may run headless: module creation
// happens during init, not off rAF, so there is no throttling to dodge.
// HEADED=1 if a driver misbehaves without a window.
//
//   node scripts/shader_check.mjs
//   DEMO_URL='http://localhost:3000/zealot/demos/g1_web/?n=10' node scripts/shader_check.mjs
//
// Knobs: DEMO_URL, OUT_DIR, WAIT_MS, HEADED=1, CHROME (binary), CHROME_ARGS
// (extra flags, space-separated).
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const CHROME = process.env.CHROME
  ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const URL = process.env.DEMO_URL
  ?? 'http://localhost:3000/zealot/demos/g1_terrain_web/?n=1&lvl=4';
const OUT_DIR = process.env.OUT_DIR ?? '/tmp/zealot-wgsl';
// Pipelines are built as the sim needs them, and the wasm compile alone runs
// ~60s (see drive_check.mjs) — so poll well past "the page loaded".
const WAIT_MS = Number(process.env.WAIT_MS ?? 180000);

fs.rmSync(OUT_DIR, {recursive: true, force: true});
fs.mkdirSync(OUT_DIR, {recursive: true});

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: process.env.HEADED === '1' ? false : 'new',
  protocolTimeout: 600000,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--no-first-run',
    '--window-size=1280,900',
    // e.g. CHROME_ARGS='--use-angle=metal' to match the other harnesses, or
    // '--use-angle=swiftshader' to validate WGSL on a machine with no GPU.
    ...(process.env.CHROME_ARGS ? process.env.CHROME_ARGS.split(' ').filter(Boolean) : []),
  ],
});

const page = await browser.newPage();
page.on('pageerror', (err) => console.log(`[pageerror] ${String(err).slice(0, 300)}`));

// Installed before ANY page script: the wasm creates its modules during init,
// so a hook applied after navigation would miss every one of them.
await page.evaluateOnNewDocument(() => {
  window.__wgsl = [];
  const proto = GPUDevice.prototype;
  const orig = proto.createShaderModule;
  proto.createShaderModule = function (desc) {
    const mod = orig.call(this, desc);
    const rec = {label: desc.label ?? `module_${window.__wgsl.length}`, code: desc.code, msgs: null};
    window.__wgsl.push(rec);
    // Resolve on the microtask queue rather than awaiting here: the caller is
    // synchronous wasm, and getCompilationInfo() must not block its return.
    mod.getCompilationInfo?.()
      .then((info) => {
        rec.msgs = info.messages.map((m) => ({
          type: m.type, line: m.lineNum, pos: m.linePos, message: m.message,
        }));
      })
      .catch((e) => { rec.msgs = [{type: 'error', line: 0, pos: 0, message: String(e)}]; });
    return mod;
  };
});

await page.goto(URL, {waitUntil: 'load', timeout: 120000});
console.log(`[nav] ${URL}`);

// The main thread blocks hard during pipeline compilation, so every evaluate()
// is best-effort — keep polling and take the last answer that came back.
const withTimeout = (p, ms) => Promise.race([p, new Promise((r) => setTimeout(() => r(null), ms))]);
const readAll = () => withTimeout(
  page.evaluate(() => JSON.stringify(window.__wgsl ?? [])),
  10000,
);

let mods = [];
const deadline = Date.now() + WAIT_MS;
let settled = 0;
while (Date.now() < deadline) {
  await new Promise((r) => setTimeout(r, 3000));
  const raw = await readAll();
  if (!raw) { process.stdout.write('.'); continue; }
  mods = JSON.parse(raw);
  const done = mods.filter((m) => m.msgs !== null).length;
  if (done !== settled) {
    settled = done;
    console.log(`\n[poll] ${done}/${mods.length} modules reported`);
  } else {
    process.stdout.write('.');
  }
  // Every module created so far has answered, and at least one exists: the
  // init burst is over. A later lazily-built pipeline needs a longer WAIT_MS.
  if (mods.length && done === mods.length) break;
}
console.log('');

if (!mods.length) {
  console.log('No shader modules were created — the page died before touching WebGPU.');
  console.log('Check the console for a wasm panic or a 404 on pkg/example_bg.wasm.');
  await browser.close();
  process.exit(1);
}

let bad = 0;
for (const [i, m] of mods.entries()) {
  const errs = (m.msgs ?? []).filter((x) => x.type === 'error');
  const name = `${String(i).padStart(3, '0')}_${m.label.replace(/[^\w.-]+/g, '_')}`;
  if (!errs.length) {
    console.log(`  ok    ${m.label}`);
    continue;
  }
  bad++;
  const file = path.join(OUT_DIR, `${name}.wgsl`);
  fs.writeFileSync(file, m.code ?? '');
  console.log(`  FAIL  ${m.label}  ->  ${file}`);
  const lines = (m.code ?? '').split('\n');
  for (const e of errs) {
    console.log(`        ${e.line}:${e.pos}  ${e.message.split('\n')[0]}`);
    // The offending line plus its neighbours: enough to recognise the kernel
    // without opening the dump.
    for (let l = Math.max(1, e.line - 2); l <= Math.min(lines.length, e.line + 2); l++) {
      console.log(`        ${l === e.line ? '>' : ' '} ${String(l).padStart(5)} | ${lines[l - 1]}`);
    }
  }
}

console.log(`\n${mods.length} modules, ${bad} rejected. Dumps in ${OUT_DIR}`);
await browser.close();
process.exit(bad ? 1 : 0);

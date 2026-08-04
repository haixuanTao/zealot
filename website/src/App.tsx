import {useEffect, useRef, useState} from 'react';

const BASE = import.meta.env.BASE_URL; // '/zealot/'

/// The three engines, all running the SAME policy on the SAME terrain.
/// `path` is relative to BASE and lives in public/.
const DEMOS = [
  {
    name: 'nexus',
    path: 'demos/g1_terrain_web/',
    title: 'nexus (WebGPU)',
    source:
      'https://github.com/haixuanTao/zealot/blob/master/examples/biped/g1_web_demo.rs',
    needsWebGPU: true,
  },
  {
    name: 'rapier',
    path: 'bench/three_rapier_bench.html',
    title: 'rapier.js (CPU wasm)',
    source:
      'https://github.com/haixuanTao/zealot/blob/master/website/public/bench/three_rapier_bench.html',
    needsWebGPU: false,
  },
  {
    name: 'mujoco',
    path: 'bench/three_mujoco_bench.html',
    title: 'MuJoCo (CPU wasm)',
    source:
      'https://github.com/haixuanTao/zealot/blob/master/website/public/bench/three_mujoco_bench.html',
    needsWebGPU: false,
  },
] as const;

type DemoName = (typeof DEMOS)[number]['name'];

/// Scene knobs. The terrain is baked into each engine's scene at startup, so
/// changing one reloads the demo iframe with new URL params.
// n = 1 by default: measured on an M-series iGPU, one robot runs at ~98% of
// real time and three at ~83% (the physics is dispatch-latency-bound; the
// missing 17% needs engine-level kernel fusion, not a setting). The Robots
// slider makes the trade explicit instead of shipping it as slow-motion.
const DEFAULTS = {n: 1, lvl: 4, amp: 100, slope: 2, terrain: true, ckpt: ''};
type Knobs = typeof DEFAULTS;

/// The page's own query string seeds the knobs, and Apply writes them back, so
/// a link like `…/zealot/?ckpt=owner/repo&n=1` opens on that policy — pointing
/// someone at a checkpoint is just sending them a URL.
function knobsFromUrl(): Knobs {
  if (typeof location === 'undefined') return DEFAULTS;
  const q = new URLSearchParams(location.search);
  const num = (k: keyof Knobs, lo: number, hi: number) => {
    const v = Number(q.get(k));
    return q.has(k) && Number.isFinite(v) ? Math.min(hi, Math.max(lo, Math.round(v))) : null;
  };
  return {
    n: num('n', 1, 20) ?? DEFAULTS.n,
    lvl: num('lvl', 0, 19) ?? DEFAULTS.lvl,
    amp: num('amp', 0, 300) ?? DEFAULTS.amp,
    slope: num('slope', 0, 20) ?? DEFAULTS.slope,
    terrain: q.get('terrain') !== '0',
    ckpt: q.get('ckpt') ?? DEFAULTS.ckpt,
  };
}

/// Reflect the applied scene into the address bar without adding history
/// entries — the link in the URL bar is always the one to share.
function writeUrl(k: Knobs) {
  if (typeof history === 'undefined') return;
  const q = new URLSearchParams();
  (Object.keys(DEFAULTS) as (keyof Knobs)[]).forEach((key) => {
    if (k[key] !== DEFAULTS[key]) q.set(key, String(k[key] === true ? 1 : k[key] === false ? 0 : k[key]));
  });
  const qs = q.toString();
  history.replaceState(null, '', qs ? `${location.pathname}?${qs}` : location.pathname);
}

/// Published zealot policies. Every engine below can load any checkpoint in
/// here — same weights, three different physics engines — and the demos also
/// accept `owner/repo/file.safetensors` or a full URL, so this repo is a
/// default, not a restriction.
const HF_REPO = 'haixuantao/zealot-g1-locomotion';

type Ckpt = {value: string; label: string; repo: string};

/// Anything a person might paste, reduced to "which repo" or "which file".
/// Mirrors the demos' own parser, so the field accepts a handle copied off a
/// model page, the page URL, a link to one file, or a direct URL elsewhere.
export function parseCkpt(spec: string): {repo?: string; url?: string} {
  const clean = spec.trim().replace(/\/+$/, '');
  if (!clean) return {};
  const hub = clean.match(/^(?:https?:\/\/(?:huggingface\.co|hf\.co)\/|hf\.co\/|hf:)(.+)$/);
  if (hub) {
    const p = hub[1].split('?')[0].split('/');
    if (p.length > 4 && (p[2] === 'blob' || p[2] === 'resolve'))
      return {url: `https://huggingface.co/${p[0]}/${p[1]}/resolve/${p[3]}/${p.slice(4).join('/')}`};
    if (p.length >= 2) return {repo: `${p[0]}/${p[1]}`};
  }
  if (/^https?:\/\//.test(clean)) return {url: clean};
  const p = clean.split('/');
  if (p.length >= 3)
    return {url: `https://huggingface.co/${p[0]}/${p[1]}/resolve/main/${p.slice(2).join('/')}`};
  if (p.length === 2) return {repo: clean};
  return {url: clean};
}

/// Newest first by the numbers in the filename — `g1_v24_iter32780` before
/// `g1_v21_iter4560` — so a repo's latest checkpoint is the one preselected.
export function newestFirst(files: string[]): string[] {
  const nums = (s: string) => (s.match(/\d+/g) ?? []).map(Number);
  return [...files].sort((a, b) => {
    const [x, y] = [nums(a), nums(b)];
    for (let i = 0; i < Math.max(x.length, y.length); i++) {
      const d = (y[i] ?? -1) - (x[i] ?? -1);
      if (d) return d;
    }
    return b.localeCompare(a);
  });
}

/// The `.safetensors` in a Hugging Face repo, newest first. The Hub API is
/// public and CORS-clean, so this works straight from the browser.
async function listRepo(repo: string): Promise<Ckpt[]> {
  const r = await fetch(`https://huggingface.co/api/models/${repo}`);
  if (!r.ok)
    throw new Error(
      // The Hub answers 401 for a repo that does not exist as well as for a
      // private one — from out here they are the same thing.
      r.status === 404 || r.status === 401
        ? 'no such public repo'
        : `Hub lookup failed (${r.status})`,
    );
  const files = newestFirst(
    ((await r.json()).siblings ?? [])
      .map((s: {rfilename: string}) => s.rfilename)
      .filter((f: string) => f.endsWith('.safetensors')),
  );
  if (!files.length) throw new Error('no .safetensors in that repo');
  return files.map((f) => ({
    value: `${repo}/${f}`,
    label: f.replace(/\.safetensors$/, ''),
    repo,
  }));
}

/// The default repo's checkpoints, loaded once for the dropdown. Best-effort:
/// if the Hub is unreachable the picker still offers the built-in policy.
function useHfCheckpoints(): Ckpt[] {
  const [list, setList] = useState<Ckpt[]>([]);
  useEffect(() => {
    let alive = true;
    listRepo(HF_REPO)
      .then((l) => alive && setList(l))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);
  return list;
}

function demoSrc(name: DemoName, k: Knobs): string {
  const demo = DEMOS.find((d) => d.name === name)!;
  const terrain = `lvl=${k.lvl}&amp=${k.amp}&slope=${k.slope}`;
  // The nexus demo is always on terrain; the sim2sim tabs take a flag.
  const query =
    name === 'nexus' ? `?${terrain}&n=${k.n}` : k.terrain ? `?${terrain}&n=${k.n}` : `?n=${k.n}`;
  // Empty = the checkpoint embedded in the demo. `ckpt` is last so that a
  // pasted URL keeps working even if it carries its own query string.
  const ckpt = k.ckpt ? `&ckpt=${encodeURIComponent(k.ckpt)}` : '';
  return `${BASE}${demo.path}${query}${ckpt}`;
}

/// Why this browser can't run the nexus demo, if it can't. Safari and Firefox
/// both expose `navigator.gpu` yet mis-run the physics, so this sniffs rather
/// than feature-detects; every iOS browser is WebKit underneath whatever name
/// it wears, so the platform is what counts there, not the brand.
///
/// Synchronous on purpose: the answer picks the opening tab, and deciding it
/// after mount would download the 17 MB WebGPU module before switching away
/// from it.
function nexusBlockedBy(): string | null {
  if (typeof navigator === 'undefined') return null;
  const ua = navigator.userAgent;
  // iPadOS reports itself as a Mac; the touch points give it away. Desktop
  // Chrome/Edge is excluded from that second test by its `Chrome/<version>`
  // token — iPadOS Chrome says `CriOS` instead — so a Mac that merely has a
  // touch device attached is not mistaken for an iPad and demoted off the GPU
  // demo it can actually run.
  const ios =
    /iphone|ipad|ipod/i.test(ua) ||
    (/macintosh/i.test(ua) && navigator.maxTouchPoints > 1 && !/chrome\/\d/i.test(ua));
  if (ios) return 'iOS';
  if (/firefox|fxios/i.test(ua)) return 'Firefox';
  if (/safari/i.test(ua) && !/chrome|chromium|android|crios|edg/i.test(ua)) return 'Safari';
  if (!(navigator as {gpu?: unknown}).gpu) return 'this browser';
  return null;
}

/// Where to land someone who cannot run the GPU demo. MuJoCo is the closest
/// match to what nexus shows (0.35 vs 0.36 m/s on the same policy) and the
/// name people know — but it ships a 9.7 MB wasm, so phones get rapier, which
/// streams a small module off a CDN and starts almost immediately.
function fallbackDemo(blocker: string | null): DemoName {
  return blocker === 'iOS' ? 'rapier' : 'mujoco';
}

function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="knob">
      <span className="knobLabel">{props.label}</span>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}

const REPO = 'haixuanTao/zealot';

/// GitHub link with a live star count. The count is best-effort: GitHub's
/// unauthenticated API is rate-limited per IP, so a failure (or a rate-limit)
/// just leaves a plain link, and a successful count is cached for the session.
function GitHubButton({large}: {large?: boolean}) {
  const [stars, setStars] = useState<number | null>(null);

  useEffect(() => {
    const cached = sessionStorage.getItem('ghStars');
    if (cached !== null) {
      setStars(Number(cached));
      return;
    }
    let alive = true;
    fetch(`https://api.github.com/repos/${REPO}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        const n = d?.stargazers_count;
        if (alive && typeof n === 'number') {
          sessionStorage.setItem('ghStars', String(n));
          setStars(n);
        }
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  // GitHub's own button widget (ghbtns.com) shape: a "Star" button with the
  // octocat, then a separate count bubble pointing back at it.
  return (
    <span className={`ghWidget${large ? ' ghWidgetLarge' : ''}`}>
      <a
        className="ghStarBtn"
        href={`https://github.com/${REPO}`}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={`Star ${REPO} on GitHub`}
      >
        <svg viewBox="0 0 16 16" aria-hidden="true" className="ghMark">
          <path
            fill="currentColor"
            d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
          />
        </svg>
        {/* Name the repo in the prose CTA: the paragraph next to it talks
            about nexus, so a bare "Star" reads as if it were nexus's. */}
        {large ? 'Star zealot' : 'Star'}
      </a>
      {stars !== null && (
        <a
          className="ghCount"
          href={`https://github.com/${REPO}/stargazers`}
          target="_blank"
          rel="noopener noreferrer"
        >
          {stars >= 1000 ? `${(stars / 1000).toFixed(1)}k` : stars}
        </a>
      )}
    </span>
  );
}

/// The demo iframes forward downward wheel events instead of zooming, so the
/// page keeps scrolling even while the cursor is over a canvas.
function useForwardedScroll() {
  useEffect(() => {
    const onMessage = (e: MessageEvent) => {
      if (e.origin !== location.origin) return;
      const dy = (e.data as {zealotScroll?: number} | null)?.zealotScroll;
      if (typeof dy === 'number') window.scrollBy({top: dy});
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, []);
}

/// Policy picker. The dropdown holds the published checkpoints; the field
/// below takes ANY Hugging Face handle or URL — paste `owner/repo` and the
/// repo's checkpoints are looked up and folded into the dropdown, newest
/// preselected, so every checkpoint someone publishes is one paste away.
function PolicyPicker({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: Ckpt[];
}) {
  const [text, setText] = useState('');
  const [found, setFound] = useState<Ckpt[]>([]);
  const [status, setStatus] = useState<{msg: string; bad?: boolean} | null>(null);

  // Resolve once typing pauses — the whole thing behind one debounce, so a
  // half-typed handle never fires a lookup or overwrites the selection.
  useEffect(() => {
    const spec = text.trim();
    if (!spec) {
      setStatus(null);
      setFound([]);
      return;
    }
    let alive = true;
    const t = setTimeout(() => {
      const {repo, url} = parseCkpt(spec);
      if (url) {
        setFound([{value: url, label: url.split('/').pop() ?? url, repo: 'URL'}]);
        setStatus({msg: 'direct link — Apply to run it'});
        onChange(url);
        return;
      }
      if (!repo) return;
      setStatus({msg: `looking up ${repo}…`});
      listRepo(repo)
        .then((list) => {
          if (!alive) return;
          setFound(list);
          setStatus({
            msg: `${list.length} checkpoint${list.length === 1 ? '' : 's'} in ${repo}`,
          });
          onChange(list[0].value); // newest
        })
        .catch((e) => {
          if (!alive) return;
          setFound([]);
          setStatus({msg: `${repo}: ${e.message}`, bad: true});
        });
    }, 500);
    return () => {
      alive = false;
      clearTimeout(t);
    };
    // `onChange` is recreated each render; re-running on it would refetch per
    // keystroke, which is exactly what the debounce is here to prevent.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  // Anything found by pasting joins the built-in list, without duplicates.
  const all = [...options, ...found.filter((f) => !options.some((o) => o.value === f.value))];

  return (
    <div className="policy">
      <label className="knob">
        <span className="knobLabel">Policy</span>
        <select className="knobSelect" value={value} onChange={(e) => onChange(e.target.value)}>
          <option value="">g1_walk_v24 (built in)</option>
          {all.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
              {o.repo === HF_REPO ? ' — 🤗' : o.repo === 'URL' ? ' — link' : ` — 🤗 ${o.repo}`}
            </option>
          ))}
        </select>
      </label>
      <input
        className="knobInput"
        value={text}
        placeholder="or paste a 🤗 handle / URL"
        spellCheck={false}
        onChange={(e) => setText(e.target.value)}
      />
      {status && (
        <span className={`policyStatus${status.bad ? ' policyStatusBad' : ''}`}>{status.msg}</span>
      )}
    </div>
  );
}

function Demo() {
  useForwardedScroll();
  // Open on an engine this browser can actually run. Safari, Firefox and iOS
  // get a CPU engine on the front page — the same policy and the same terrain,
  // just stepped on the CPU — instead of a demo that would fail in front of
  // them. The nexus tab stays one click away, with the warning.
  const [blocker] = useState(nexusBlockedBy);
  const [selected, setSelected] = useState<DemoName>(() =>
    blocker ? fallbackDemo(blocker) : 'nexus',
  );
  const checkpoints = useHfCheckpoints();
  const [knobs, setKnobs] = useState<Knobs>(knobsFromUrl);
  const [applied, setApplied] = useState<Knobs>(knobsFromUrl);
  const [reloading, setReloading] = useState(false);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const current = DEMOS.find((d) => d.name === selected)!;
  const dirty = (Object.keys(DEFAULTS) as (keyof Knobs)[]).some((k) => knobs[k] !== applied[k]);

  // Blank the iframe before swapping so the old WebGPU context is released.
  const remount = (next: () => void) => {
    setReloading(true);
    if (iframeRef.current) iframeRef.current.src = 'about:blank';
    setTimeout(() => {
      next();
      setReloading(false);
    }, 400);
  };

  // On the nexus tab, say it will not work. On a CPU tab, say why that is the
  // one being shown — otherwise the headline engine is missing with no
  // explanation.
  // Every iOS browser is WebKit, so "try Chrome" is useless advice there —
  // it has to be a desktop.
  const elsewhere =
    blocker === 'iOS'
      ? 'desktop Chrome (or another Chromium-based browser)'
      : 'Chrome (or another Chromium-based browser)';
  const warn = !blocker
    ? null
    : current.needsWebGPU
      ? blocker === 'this browser'
        ? `WebGPU is not available in this browser. The simulation runs its physics as WebGPU compute shaders and requires ${elsewhere}.`
        : `This demo does not work on ${blocker}. Its WebGPU implementation doesn't run the physics correctly — please use ${elsewhere}.`
      : {
          note: `${blocker} can't run the WebGPU demo, so this is the CPU engine — same policy, same terrain, physics stepped on the CPU. For the GPU one, open this page in ${elsewhere}.`,
        };

  return (
    <div className="demo">
      <div className="toolbar">
        <div className="tabs">
          {DEMOS.map((d) => (
            <button
              key={d.name}
              className={`tab${selected === d.name ? ' tabActive' : ''}`}
              onClick={() => selected !== d.name && remount(() => setSelected(d.name))}
            >
              {d.title}
            </button>
          ))}
        </div>

        <div className="knobs">
          <PolicyPicker
            value={knobs.ckpt}
            options={checkpoints}
            onChange={(ckpt) => setKnobs({...knobs, ckpt})}
          />
          {selected !== 'nexus' && (
            <label className="knob knobCheck">
              <input
                type="checkbox"
                checked={knobs.terrain}
                onChange={(e) => setKnobs({...knobs, terrain: e.target.checked})}
              />
              <span className="knobLabel">Terrain</span>
            </label>
          )}
          <Slider
            label={`Robots ${knobs.n}`}
            value={knobs.n}
            min={1}
            max={20}
            onChange={(n) => setKnobs({...knobs, n})}
          />
          <Slider
            label={`Difficulty ${knobs.lvl}/19`}
            value={knobs.lvl}
            min={0}
            max={19}
            onChange={(lvl) => setKnobs({...knobs, lvl})}
          />
          <Slider
            label={`Roughness ${knobs.amp}%`}
            value={knobs.amp}
            min={0}
            max={300}
            step={25}
            onChange={(amp) => setKnobs({...knobs, amp})}
          />
          <Slider
            label={`Slope ${knobs.slope}°`}
            value={knobs.slope}
            min={0}
            max={20}
            onChange={(slope) => setKnobs({...knobs, slope})}
          />
          <button
            className="apply"
            disabled={!dirty}
            onClick={() =>
              remount(() => {
                setApplied(knobs);
                writeUrl(knobs);
              })
            }
          >
            {dirty ? 'Apply' : 'Applied'}
          </button>
          <GitHubButton />
        </div>
      </div>

      {typeof warn === 'string' && <div className="warning">{warn}</div>}
      {warn && typeof warn !== 'string' && <div className="notice">{warn.note}</div>}

      <div className="viewer">
        {reloading ? (
          <div className="placeholder">Loading…</div>
        ) : (
          <iframe
            ref={iframeRef}
            key={`${selected}-${applied.n}-${applied.lvl}-${applied.amp}-${applied.slope}-${applied.terrain}-${applied.ckpt}`}
            src={demoSrc(selected, applied)}
            title={current.title}
            className="viewerFrame"
          />
        )}
        <a
          className="sourceLink"
          href={current.source}
          target="_blank"
          rel="noopener noreferrer"
        >
          &lt;/&gt; Source
        </a>
        <a
          className="scrollCue"
          href="#more"
          onClick={(e) => {
            e.preventDefault();
            document.getElementById('more')?.scrollIntoView({behavior: 'smooth'});
            history.replaceState(null, '', '#more');
          }}
        >
          <span>What is this?</span>
          <span className="scrollCueArrow">↓</span>
        </a>
      </div>
    </div>
  );
}

/// Scroll sections: a block of text plus either a screenshot or, for the
/// toolchain slide, the compiler cards.
type Slide = {
  kicker: string;
  title: string;
  body: string;
  img?: string;
  /// Fit the image inside the frame (logos) instead of cropping to fill.
  contain?: boolean;
  /// Render the GPU compiler cards as this slide's illustration.
  cards?: boolean;
};

const SLIDES: Slide[] = [
  {
    img: 'img/slides/nexus-logo.png',
    contain: true,
    kicker: 'nexus',
    title: '100% Rust simulator, on any GPU',
    body: "Physics is nexus, dimforge's GPU multiphysics engine: the whole solver is compute shaders written in Rust via Rust-GPU. The same code runs through WebGPU in your browser, and through CUDA or Metal natively — no Python, no CUDA C, no per-backend rewrite.",
  },
  {
    img: 'img/slides/nexus-terrain.jpg',
    kicker: 'zealot-rl',
    title: 'rsl_rl, ported to Rust — and to every GPU',
    body: 'zealot-rl is the rsl_rl tier rewritten in Rust: the model definition and the whole training pipeline — actor-critic network, autodiff, PPO, GAE, Adam — are Rust, not a Python front-end over a C++ core. It runs on vortx and khal, the same portable GPU layer nexus is built on, so the learning half is no more platform-bound than the physics half.',
  },
  {
    cards: true,
    kicker: 'GPU ready',
    title: 'One Rust source, every GPU backend',
    body: 'The kernels are written once, in Rust, then compiled three ways — no second implementation to keep in sync. On an RTX 5090 the native-CUDA path runs 2.4–4.3× faster than WebGPU while staying bit-exact against it.',
  },
  {
    img: 'img/slides/mujoco-sim2sim.jpg',
    kicker: 'Validation',
    title: 'Cross-checked against MuJoCo and rapier',
    body: 'A policy is only trustworthy if it survives a different solver. The same checkpoint runs sim2sim in the browser on rapier.js and on the official MuJoCo WebAssembly build — the reference engine — walking bit-identical terrain, so you can watch where the engines agree and where they diverge.',
  },
];

/// The GPU toolchain, named. Same Rust source behind each of these.
const GPU_STACK = [
  {
    name: 'rust-gpu',
    href: 'https://github.com/Rust-GPU/rust-gpu',
    what: 'Rust → SPIR-V',
    where: 'WebGPU · Metal · Vulkan',
  },
  {
    name: 'cuda-oxide',
    href: 'https://github.com/NVlabs/cuda-oxide',
    what: 'Rust → PTX',
    where: 'native CUDA',
  },
  {
    name: 'cutile-rs',
    href: 'https://github.com/NVlabs/cutile-rs',
    what: 'tiled tf32 GEMMs',
    where: 'tensor cores',
  },
] as const;

function GpuChips() {
  return (
    <div className="gpuChips">
      {GPU_STACK.map((t) => (
        <a
          className="gpuChip"
          key={t.name}
          href={t.href}
          target="_blank"
          rel="noopener noreferrer"
        >
          <code>{t.name}</code>
          <span className="gpuChipWhat">{t.what}</span>
          <span className="gpuChipWhere">{t.where}</span>
        </a>
      ))}
    </div>
  );
}

function Slides() {
  return (
    <section className="slides">
      {SLIDES.map((s, i) => {
        const text = (
          <div className="slideText">
            <span className="kicker">{s.kicker}</span>
            <h2>{s.title}</h2>
            <p>{s.body}</p>
          </div>
        );

        // The GPU-toolchain slide carries the compiler table instead of a
        // screenshot: the three cards ARE its illustration.
        if (s.cards) {
          return (
            <article className="slide slideWide" key={s.title}>
              {text}
              <GpuChips />
            </article>
          );
        }

        return (
          <article className={`slide${i % 2 ? ' slideFlip' : ''}`} key={s.title}>
            <div className="slideMedia">
              <img
                src={`${BASE}${s.img}`}
                alt={s.title}
                loading="lazy"
                className={s.contain ? 'containFit' : undefined}
              />
            </div>
            {text}
          </article>
        );
      })}
    </section>
  );
}

const FEATURES = [
  ['🦿', 'Velocity-Tracking Locomotion', 'The humanoid tracks commanded forward/lateral/turn velocities — steer it live with the command sliders in the demo.'],
  ['⚡', 'GPU-Vectorized Training', 'Thousands of parallel environments step on the GPU through nexus rigid-body physics.'],
  ['🦀', 'All-Rust PPO', 'Actor-critic MLPs, GAE, and Adam implemented in Rust — checkpoints are plain safetensors.'],
  ['🌐', 'Runs in Your Browser', 'The same env + policy compile to WebAssembly; the demo is the real simulation, in realtime.'],
  ['🤖', 'Real Robot Model', 'The MuJoCo (MJCF) model of the Unitree G1 humanoid, with real PD gains and domain randomization.'],
  ['🧪', 'Sim-to-Sim Validated', 'Policies are cross-checked between nexus, rapier (CPU), and MuJoCo before deployment.'],
] as const;

function Story() {
  return (
    <main id="more" className="story">
      <Slides />
      <section className="prose">
        <h2>Robot Learning, All in Rust, All on the GPU</h2>
        <p>
          Zealot trains humanoid locomotion policies with PPO — reinforcement learning where
          physics simulation, observation/reward computation, and the policy network all run on
          the GPU. Physics is <a href="https://nexus.dimforge.com">nexus</a>, dimforge's GPU
          multiphysics engine: compute shaders written in Rust via{' '}
          <a href="https://github.com/Rust-GPU/rust-gpu">rust-gpu</a>, executed through WebGPU (or
          CUDA / Metal natively). No Python, no PyTorch — the model definition and the training
          pipeline are both Rust, end to end.
        </p>
        <p>
          Because the stack targets WebGPU, the same physics engine compiles to WebAssembly and
          runs in your browser: the demo above walks Unitree G1 humanoids over rough terrain in
          realtime — actual GPU rigid-body simulation, not a video — and the same policy runs
          sim2sim on rapier.js and MuJoCo for comparison, on bit-identical terrain.
        </p>
        <div className="ctaRow">
          <GitHubButton large />
          <a
            className="btn btnOutline"
            href="https://github.com/dimforge/nexus"
            target="_blank"
            rel="noopener noreferrer"
          >
            dimforge/nexus
          </a>
        </div>
      </section>

      <section className="featureGrid">
        {FEATURES.map(([icon, title, body]) => (
          <div className="feature" key={title}>
            <span className="featureIcon">{icon}</span>
            <h3>{title}</h3>
            <p>{body}</p>
          </div>
        ))}
      </section>
    </main>
  );
}

export default function App() {
  return (
    <>
      <Demo />
      <Story />
    </>
  );
}

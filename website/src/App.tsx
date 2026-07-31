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
const DEFAULTS = {n: 3, lvl: 4, amp: 100, slope: 2, terrain: true, ckpt: ''};
type Knobs = typeof DEFAULTS;

/// Published zealot policies. Every engine below can load any checkpoint in
/// here — same weights, three different physics engines — and the demos also
/// accept `owner/repo/file.safetensors` or a full URL, so this repo is a
/// default, not a restriction.
const HF_REPO = 'haixuantao/zealot-g1-locomotion';

type Ckpt = {value: string; label: string};

/// The repo's `.safetensors`, newest first. Best-effort: the Hub API is public
/// and CORS-enabled, but if it is unreachable the picker just offers the
/// checkpoint baked into the demo.
function useHfCheckpoints(): Ckpt[] {
  const [list, setList] = useState<Ckpt[]>([]);
  useEffect(() => {
    let alive = true;
    fetch(`https://huggingface.co/api/models/${HF_REPO}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        const files: string[] = (d?.siblings ?? [])
          .map((s: {rfilename: string}) => s.rfilename)
          .filter((f: string) => f.endsWith('.safetensors'));
        files.sort().reverse();
        if (alive)
          setList(
            files.map((f) => ({
              value: `${HF_REPO}/${f}`,
              label: f.replace(/\.safetensors$/, ''),
            })),
          );
      })
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

function useUnsupportedBrowser(): string | null {
  const [name, setName] = useState<string | null>(null);
  useEffect(() => {
    const ua = navigator.userAgent;
    // Both expose navigator.gpu yet mis-run the physics, so sniff rather
    // than feature-detect.
    if (/firefox|fxios/i.test(ua)) setName('Firefox');
    else if (/safari/i.test(ua) && !/chrome|chromium|android|crios|edg/i.test(ua))
      setName('Safari');
  }, []);
  return name;
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

/// Policy picker: the published checkpoints, plus a field for any other
/// Hugging Face repo or URL. Loading one re-runs the SAME weights in whichever
/// engine is on screen.
function PolicyPicker({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: Ckpt[];
}) {
  const known = value === '' || options.some((o) => o.value === value);
  const [custom, setCustom] = useState(!known);
  return (
    <label className="knob">
      <span className="knobLabel">Policy</span>
      <select
        className="knobSelect"
        value={custom ? '__custom' : value}
        onChange={(e) => {
          if (e.target.value === '__custom') {
            setCustom(true);
          } else {
            setCustom(false);
            onChange(e.target.value);
          }
        }}
      >
        <option value="">g1_walk_v24 (built in)</option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label} — 🤗
          </option>
        ))}
        <option value="__custom">Other…</option>
      </select>
      {custom && (
        <input
          className="knobInput"
          value={value}
          placeholder="owner/repo/file.safetensors"
          spellCheck={false}
          onChange={(e) => onChange(e.target.value.trim())}
        />
      )}
    </label>
  );
}

function Demo() {
  useForwardedScroll();
  const [selected, setSelected] = useState<DemoName>('nexus');
  const checkpoints = useHfCheckpoints();
  const [knobs, setKnobs] = useState<Knobs>(DEFAULTS);
  const [applied, setApplied] = useState<Knobs>(DEFAULTS);
  const [reloading, setReloading] = useState(false);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const unsupported = useUnsupportedBrowser();
  const webgpuMissing = typeof navigator !== 'undefined' && !(navigator as any).gpu;

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

  const warn =
    current.needsWebGPU && unsupported
      ? `This demo does not work on ${unsupported}. Its WebGPU implementation doesn't run the physics correctly — please use Chrome (or another Chromium-based browser).`
      : current.needsWebGPU && webgpuMissing
        ? 'WebGPU is not available in this browser. The simulation runs its physics as WebGPU compute shaders and requires Chrome (or another Chromium-based browser).'
        : null;

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
            onClick={() => remount(() => setApplied(knobs))}
          >
            {dirty ? 'Apply' : 'Applied'}
          </button>
          <GitHubButton />
        </div>
      </div>

      {warn && <div className="warning">{warn}</div>}

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

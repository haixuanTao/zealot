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
const DEFAULTS = {n: 3, lvl: 4, amp: 100, slope: 5, terrain: true};
type Knobs = typeof DEFAULTS;

function demoSrc(name: DemoName, k: Knobs): string {
  const demo = DEMOS.find((d) => d.name === name)!;
  const terrain = `lvl=${k.lvl}&amp=${k.amp}&slope=${k.slope}`;
  // The nexus demo is always on terrain; the sim2sim tabs take a flag.
  const query =
    name === 'nexus' ? `?${terrain}&n=${k.n}` : k.terrain ? `?${terrain}&n=${k.n}` : `?n=${k.n}`;
  return `${BASE}${demo.path}${query}`;
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

function Demo() {
  useForwardedScroll();
  const [selected, setSelected] = useState<DemoName>('nexus');
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
        </div>
      </div>

      {warn && <div className="warning">{warn}</div>}

      <div className="viewer">
        {reloading ? (
          <div className="placeholder">Loading…</div>
        ) : (
          <iframe
            ref={iframeRef}
            key={`${selected}-${applied.n}-${applied.lvl}-${applied.amp}-${applied.slope}-${applied.terrain}`}
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

/// Scroll sections: one image + one block of text each, alternating sides.
const SLIDES = [
  {
    img: 'img/slides/nexus-logo.png',
    contain: true,
    kicker: 'nexus',
    title: '100% Rust simulator, on any GPU',
    body: "Physics is nexus, dimforge's GPU multiphysics engine: the whole solver is compute shaders written in Rust via Rust-GPU. The same code runs through WebGPU in your browser, and through CUDA or Metal natively — no Python, no CUDA C, no per-backend rewrite.",
  },
  {
    img: 'img/slides/nexus-terrain.jpg',
    kicker: 'Training',
    title: 'The simulator is the demo',
    body: 'Thousands of environments step in parallel on the GPU while PPO — actor-critic MLPs, GAE and Adam, all in Rust — learns from them. What you scrolled past is that exact environment compiled to WebAssembly: the real training simulator, not a recording.',
  },
  {
    img: 'img/slides/nexus-fleet.jpg',
    kicker: 'Terrain',
    title: 'Rough ground, generated deterministically',
    body: 'Box plateaus, waves and noise fields laid out in difficulty rows, harder with distance. Drag the sliders and the terrain regenerates from the same seeded generator every engine shares — so difficulty, roughness and slope mean exactly the same thing in all three.',
  },
  {
    img: 'img/slides/mujoco-sim2sim.jpg',
    kicker: 'Validation',
    title: 'Cross-checked against MuJoCo and rapier',
    body: 'A policy is only trustworthy if it survives a different solver. The same checkpoint runs sim2sim in the browser on rapier.js and on the official MuJoCo WebAssembly build — the reference engine — walking bit-identical terrain, so you can watch where the engines agree and where they diverge.',
  },
] as const;

function Slides() {
  return (
    <section className="slides">
      {SLIDES.map((s, i) => (
        <article className={`slide${i % 2 ? ' slideFlip' : ''}`} key={s.title}>
          <div className="slideMedia">
            <img
              src={`${BASE}${s.img}`}
              alt={s.title}
              loading="lazy"
              className={'contain' in s && s.contain ? 'containFit' : undefined}
            />
          </div>
          <div className="slideText">
            <span className="kicker">{s.kicker}</span>
            <h2>{s.title}</h2>
            <p>{s.body}</p>
          </div>
        </article>
      ))}
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
          <a href="https://github.com/Rust-GPU/rust-gpu">Rust-GPU</a>, executed through WebGPU (or
          CUDA / Metal natively). No Python, no PyTorch — the whole training loop is Rust.
        </p>
        <p>
          Because the stack targets WebGPU, the same physics engine compiles to WebAssembly and
          runs in your browser: the demo above walks Unitree G1 humanoids over rough terrain in
          realtime — actual GPU rigid-body simulation, not a video — and the same policy runs
          sim2sim on rapier.js and MuJoCo for comparison, on bit-identical terrain.
        </p>
        <div className="ctaRow">
          <a className="btn" href="https://github.com/haixuanTao/zealot">
            GitHub
          </a>
          <a className="btn btnOutline" href="https://nexus.dimforge.com">
            nexus
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

import {useEffect, useState, useRef, type ReactNode} from 'react';
import Layout from '@theme/Layout';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {Intro, Features} from '../components/AboutContent';
import styles from './demos.module.css';

// The Unitree G1 walking zealot's released v7 policy on the zealot training
// env (nexus GPU physics), live in your browser (see scripts/build-demos.sh).
// Each wasm module is a full app (physics + policy + renderer + UI).
const demos = [
  {
    name: 'g1-terrain',
    path: 'demos/g1_terrain_web/',
    title: 'nexus (WebGPU)',
    description:
      'Ten G1 humanoids walking over the randomized rough-terrain strips they were trained on, in one batched zealot/nexus GPU sim (Stand / Walk / Turn commands in the panel; sliders set spawn difficulty, roughness, and slope)',
    source: 'https://github.com/haixuanTao/zealot/blob/master/examples/biped/g1_web_demo.rs',
  },
  {
    name: 'sim2sim-rapier',
    path: 'bench/three_rapier_bench.html',
    title: 'rapier.js (CPU wasm)',
    description:
      'The same G1 + v19 policy running sim2sim on the typical web physics stack — three.js WebGL rendering with rapier.js CPU physics — so you can compare it against the nexus WebGPU demos (no WebGPU needed, runs in any browser)',
    source:
      'https://github.com/haixuanTao/zealot/blob/master/website/static/bench/three_rapier_bench.html',
  },
  {
    name: 'sim2sim-mujoco',
    path: 'bench/three_mujoco_bench.html',
    title: 'MuJoCo (CPU wasm)',
    description:
      'The reference engine in your browser: the official MuJoCo WebAssembly build stepping the 29-DOF playground G1 with the same v19 policy — the in-browser twin of the Python cross-engine validator (terrain = native MuJoCo heightfield from the same generator; no WebGPU needed)',
    source:
      'https://github.com/haixuanTao/zealot/blob/master/website/static/bench/three_mujoco_bench.html',
  },
];

// Terrain-shape knobs, shared by the rough-terrain demo and the rapier
// comparison ('on' only applies to the latter — the terrain demo is always on
// terrain, the rapier tab defaults to flat ground like the fleet demo). The
// terrain is baked into the physics scene at startup, so applying a change
// reloads the demo iframe with these as URL params.
const TERRAIN_DEFAULTS = {on: true, lvl: 4, amp: 100, slope: 5, n: 3};
type TerrainKnobs = typeof TERRAIN_DEFAULTS;

export default function Demos(): ReactNode {
  const [selected, setSelected] = useState<string | null>(null);
  const [activeDemo, setActiveDemo] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [webgpuSupported, setWebgpuSupported] = useState(true);
  const [unsupportedBrowser, setUnsupportedBrowser] = useState<string | null>(null);
  const [terrain, setTerrain] = useState<TerrainKnobs>(TERRAIN_DEFAULTS);
  const [appliedTerrain, setAppliedTerrain] = useState<TerrainKnobs>(TERRAIN_DEFAULTS);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const rootBase = useBaseUrl('/');

  // Physics runs as WebGPU compute shaders (the policy MLP runs in wasm on the
  // CPU): no WebGPU, no demo.
  useEffect(() => {
    setWebgpuSupported(typeof navigator !== 'undefined' && !!(navigator as any).gpu);
    // Safari and Firefox may expose WebGPU, but their compute shaders don't
    // run the physics correctly — warn even when navigator.gpu exists.
    if (typeof navigator !== 'undefined') {
      const ua = navigator.userAgent;
      if (/firefox|fxios/i.test(ua)) {
        setUnsupportedBrowser('Firefox');
      } else if (/safari/i.test(ua) && !/chrome|chromium|android|crios|edg/i.test(ua)) {
        setUnsupportedBrowser('Safari');
      }
    }
  }, []);

  // Handle URL hash for deep linking
  useEffect(() => {
    const hash = window.location.hash.slice(1);
    if (hash && demos.some((d) => d.name === hash)) {
      setSelected(hash);
    } else {
      setSelected('g1-terrain');
    }

    const handleHashChange = () => {
      const newHash = window.location.hash.slice(1);
      if (newHash && demos.some((d) => d.name === newHash)) setSelected(newHash);
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Handle demo transitions - clear iframe first to release WebGPU context
  useEffect(() => {
    if (selected === activeDemo) return;

    setIsLoading(true);

    // Force iframe cleanup by setting src to blank first
    if (iframeRef.current) {
      iframeRef.current.src = 'about:blank';
    }
    setActiveDemo(null);

    // Wait for the iframe to be cleared and GPU context to be released
    const timer = setTimeout(() => {
      setActiveDemo(selected);
      setIsLoading(false);
    }, 500);

    return () => clearTimeout(timer);
  }, [selected]);

  const handleSelect = (name: string) => {
    setSelected(name);
    window.location.hash = name;
  };

  // Reload the terrain demo with the chosen knobs, going through the same
  // blank-first dance as demo switches so the old WebGPU context is released.
  const applyTerrain = () => {
    setIsLoading(true);
    if (iframeRef.current) {
      iframeRef.current.src = 'about:blank';
    }
    setActiveDemo(null);
    setTimeout(() => {
      setAppliedTerrain(terrain);
      setActiveDemo(selected);
      setIsLoading(false);
    }, 500);
  };

  const terrainDirty =
    terrain.on !== appliedTerrain.on ||
    terrain.lvl !== appliedTerrain.lvl ||
    terrain.amp !== appliedTerrain.amp ||
    terrain.slope !== appliedTerrain.slope ||
    terrain.n !== appliedTerrain.n;

  const COMPARE_TABS = ['sim2sim-rapier', 'sim2sim-mujoco'];
  const isCompareTab = COMPARE_TABS.includes(selected ?? '');
  const hasTerrainControls = selected === 'g1-terrain' || isCompareTab;

  const demoSrc = (name: string) => {
    const path = demos.find((d) => d.name === name)?.path;
    const knobs = `lvl=${appliedTerrain.lvl}&amp=${appliedTerrain.amp}&slope=${appliedTerrain.slope}`;
    const query =
      name === 'g1-terrain'
        ? `?${knobs}&n=${appliedTerrain.n}`
        : COMPARE_TABS.includes(name)
          ? appliedTerrain.on
            ? `?${knobs}&n=${appliedTerrain.n}`
            : `?n=${appliedTerrain.n}`
          : '';
    return `${rootBase}${path}${query}`;
  };

  const current = demos.find((d) => d.name === selected);

  return (
    <Layout
      title="Live Demo"
      description="The Unitree G1 humanoid simulated in realtime in your browser — nexus GPU physics in WebAssembly + WebGPU"
      noFooter
    >
      <div className={styles.container}>
        {/* ONE bar: engine tabs + the knobs they share. Two stacked bars read
            as two competing headers now that the page scrolls. */}
        <div className={styles.toolbar}>
          <div className={styles.tabs}>
            {demos.map((demo) => (
              <button
                key={demo.name}
                className={`${styles.tab} ${selected === demo.name ? styles.tabSelected : ''}`}
                onClick={() => handleSelect(demo.name)}
              >
                {demo.title}
              </button>
            ))}
          </div>

          {hasTerrainControls && (
            <div className={styles.terrainControls}>
            {isCompareTab && (
              <label className={styles.terrainControl}>
                <input
                  type="checkbox"
                  checked={terrain.on}
                  onChange={(e) => setTerrain({...terrain, on: e.target.checked})}
                />
                Terrain (off = flat ground)
              </label>
            )}
            <label className={styles.terrainControl}>
              Robots {terrain.n}
              <input
                type="range"
                min={1}
                max={20}
                value={terrain.n}
                onChange={(e) => setTerrain({...terrain, n: Number(e.target.value)})}
              />
            </label>
            <label className={styles.terrainControl}>
              Difficulty {terrain.lvl}/19
              <input
                type="range"
                min={0}
                max={19}
                value={terrain.lvl}
                onChange={(e) => setTerrain({...terrain, lvl: Number(e.target.value)})}
              />
            </label>
            <label className={styles.terrainControl}>
              Roughness {terrain.amp}%
              <input
                type="range"
                min={0}
                max={300}
                step={25}
                value={terrain.amp}
                onChange={(e) => setTerrain({...terrain, amp: Number(e.target.value)})}
              />
            </label>
            <label className={styles.terrainControl}>
              Slope {terrain.slope}°
              <input
                type="range"
                min={0}
                max={20}
                value={terrain.slope}
                onChange={(e) => setTerrain({...terrain, slope: Number(e.target.value)})}
              />
            </label>
              <button
                className={styles.terrainApply}
                disabled={!terrainDirty}
                onClick={applyTerrain}
              >
                {terrainDirty ? 'Apply' : 'Applied'}
              </button>
            </div>
          )}
        </div>

        {unsupportedBrowser && !isCompareTab && (
          <div className={styles.webgpuWarning}>
            <strong>This demo does not work on {unsupportedBrowser}.</strong>{' '}
            {unsupportedBrowser}&rsquo;s WebGPU implementation doesn&rsquo;t run
            the physics correctly — please use Chrome (or another
            Chromium-based browser) instead.
          </div>
        )}

        {!webgpuSupported && !unsupportedBrowser && !isCompareTab && (
          <div className={styles.webgpuWarning}>
            <strong>WebGPU is not available in this browser.</strong> The
            simulation runs its physics as WebGPU compute shaders and requires
            Chrome (or another Chromium-based browser); if you&rsquo;re on
            Chromium and see this, enable <code>Unsafe WebGPU Support</code> in{' '}
            <code>chrome://flags</code>. Safari and Firefox are not supported.
          </div>
        )}

        <div className={styles.viewer}>
          {activeDemo ? (
            <>
              <iframe
                ref={iframeRef}
                key={`${activeDemo}-${appliedTerrain.n}-${appliedTerrain.lvl}-${appliedTerrain.amp}-${appliedTerrain.slope}`}
                src={demoSrc(activeDemo)}
                title={activeDemo}
                className={styles.viewerFrame}
              />
              <div className={styles.viewerControls}>
                <a
                  href={current?.source}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={styles.sourceLink}
                >
                  &lt;/&gt; Source
                </a>
              </div>
            </>
          ) : isLoading ? (
            <div className={styles.placeholder}>
              Loading...
            </div>
          ) : (
            <div className={styles.placeholder}>
              Select a demo
            </div>
          )}
          {/* Plain anchor semantics, but scroll explicitly: a still-loading
              wasm iframe can swallow the browser's own anchor jump. */}
          <a
            href="#more"
            className={styles.scrollCue}
            aria-label="Learn more"
            onClick={(e) => {
              e.preventDefault();
              document.getElementById('more')?.scrollIntoView({behavior: 'smooth'});
              window.history.replaceState(null, '', '#more');
            }}
          >
            <span>What is this?</span>
            <span className={styles.scrollCueArrow}>↓</span>
          </a>
        </div>
      </div>
      <main id="more">
        <Intro />
        <Features />
      </main>
    </Layout>
  );
}

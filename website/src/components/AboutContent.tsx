import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';

import styles from '../pages/index.module.css';

/// The "what is this" story. Rendered BELOW the live demo on the front page
/// (scroll down to read) and on its own at /about.
export function Intro(): ReactNode {
  return (
    <section className={styles.codeSection}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <Heading as="h2">Robot Learning, All in Rust, All on the GPU</Heading>
            <p>
              Zealot trains humanoid locomotion policies with PPO —
              reinforcement learning where physics simulation,
              observation/reward computation, and the policy network all run
              on the GPU. Physics is{' '}
              <Link to="https://nexus.dimforge.com">nexus</Link>, dimforge's
              GPU multiphysics engine: compute shaders written in Rust via{' '}
              <Link to="https://github.com/Rust-GPU/rust-gpu">Rust-GPU</Link>,
              executed through WebGPU (or CUDA / Metal natively). No Python,
              no PyTorch — the whole training loop is Rust.
            </p>
            <p>
              Because the stack targets WebGPU, the same physics engine
              compiles to WebAssembly and runs in your browser: the demo above
              walks Unitree G1 humanoids over rough terrain in realtime —
              actual GPU rigid-body simulation, not a video — and the same
              policy runs sim2sim on rapier.js and MuJoCo for comparison.
            </p>
            <div className={styles.codeLinks}>
              <Link
                className="button button--primary button--lg"
                to="https://github.com/haixuanTao/zealot">
                GitHub
              </Link>
              <Link
                className="button button--outline button--primary button--lg"
                to="https://nexus.dimforge.com">
                nexus
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export function Features(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featureGrid}>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🦿</span>
            <h3>Velocity-Tracking Locomotion</h3>
            <p>The humanoid tracks commanded forward/lateral/turn velocities — steer it live with the command sliders in the demo.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>⚡</span>
            <h3>GPU-Vectorized Training</h3>
            <p>Thousands of parallel environments step on the GPU through nexus rigid-body physics.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🦀</span>
            <h3>All-Rust PPO</h3>
            <p>Actor-critic MLPs, GAE, and Adam implemented in Rust — checkpoints are plain safetensors.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🌐</span>
            <h3>Runs in Your Browser</h3>
            <p>The same env + policy compile to WebAssembly; the demo is the real simulation, in realtime.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🤖</span>
            <h3>Real Robot Model</h3>
            <p>The MuJoCo (MJCF) model of the Unitree G1 humanoid, with real PD gains and domain randomization.</p>
          </div>
          <div className={styles.feature}>
            <span className={styles.featureIcon}>🧪</span>
            <h3>Sim-to-Sim Validated</h3>
            <p>Policies are cross-checked between nexus, rapier (CPU), and MuJoCo before deployment.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

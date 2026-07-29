import type {ReactNode} from 'react';
import Layout from '@theme/Layout';

import {Intro, Features} from '../components/AboutContent';

export default function About(): ReactNode {
  return (
    <Layout
      title="GPU-native robot learning in Rust"
      description="Zealot trains humanoid locomotion policies with PPO on nexus GPU physics, all in Rust — and runs the result live in your browser via WebAssembly and WebGPU.">
      <main>
        <Intro />
        <Features />
      </main>
    </Layout>
  );
}

#!/usr/bin/env bash
#
# Publish website/dist to the gh-pages branch (GitHub Pages project site at
# https://haixuantao.github.io/zealot/).
#
# Run AFTER a build:  npm run deploy   (= npm run build && this script)
# The wasm demos under public/demos/ are gitignored build output — build them
# with scripts/build-demos.sh first if they changed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBSITE_DIR="$(dirname "$SCRIPT_DIR")"
DIST="$WEBSITE_DIR/dist"
BRANCH=gh-pages
REMOTE=${DEPLOY_REMOTE:-origin}

[ -d "$DIST" ] || { echo "no dist/ — run npm run build first" >&2; exit 1; }

# .nojekyll keeps Pages from eating files that start with an underscore.
touch "$DIST/.nojekyll"

# API docs ride along under /doc/: rustdoc for the library crates, with a
# tiny index redirect (there is no root crate to land on). Rebuilt on every
# deploy — it's ~1 s warm and keeps the hosted docs from silently going
# stale relative to the code the site claims to demonstrate.
ZEALOT_DIR="$(dirname "$WEBSITE_DIR")"
(cd "$ZEALOT_DIR" && cargo doc --no-deps -p zealot-env -p zealot-rl >/dev/null)
rm -rf "$DIST/doc"
cp -R "$ZEALOT_DIR/target/doc" "$DIST/doc"
cat > "$DIST/doc/index.html" <<'HTML'
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>zealot — docs</title>
<style>
  body { font: 16px/1.6 system-ui, sans-serif; color: #1c2b2d; max-width: 46rem;
         margin: 3rem auto; padding: 0 1.2rem; }
  h1 { font-size: 1.5rem; } h2 { font-size: 1.02rem; margin-top: 1.7rem;
       text-transform: uppercase; letter-spacing: 0.06em; color: #14747c; }
  a { color: #14747c; } li { margin: 0.35rem 0; }
  p.sub { color: #5a6b6d; margin-top: -0.5rem; font-size: 0.9rem; }
</style></head><body>
<h1>zealot — documentation</h1>
<p>A whole-body-control training stack for humanoid robots, in Rust.
<a href="../">Live demo</a> · <a href="https://github.com/haixuanTao/zealot">GitHub</a></p>

<h2>Getting started</h2>
<p class="sub">Learning-oriented — from the live demo to your own walking policy.</p>
<ul>
  <li><a href="https://github.com/haixuanTao/zealot/blob/master/docs/getting-started.md">Getting started</a> — watch it walk (the <a href="../">live demo</a> is step one, nothing to install), train the humanoid, load your checkpoint back into the demo</li>
</ul>

<h2>How-to guides</h2>
<p class="sub">Task-oriented — recipes for specific jobs.</p>
<ul>
  <li><a href="https://github.com/haixuanTao/zealot/blob/master/docs/development.md">Building &amp; development</a> — toolchain setup (cargo-gpu, the native-CUDA cubin chain), git hooks, running the checks</li>
  <li><a href="https://github.com/haixuanTao/zealot/blob/master/docs/benchmarks.md">Reproducing the benchmarks</a> — full methodology, tables, and repro commands</li>
  <li><a href="https://github.com/haixuanTao/zealot/blob/master/website/README.md">The demo site</a> — build/deploy flow and demo URL knobs</li>
</ul>

<h2>Reference</h2>
<p class="sub">Information-oriented — the API, generated from the code.</p>
<ul>
  <li><a href="zealot_env/index.html"><code>zealot_env</code></a> — the environment/MDP layer: observations, rewards, terminations, terrain, robot specs</li>
  <li><a href="zealot_rl/index.html"><code>zealot_rl</code></a> — CPU reference policy network, autodiff, and PPO (what the GPU kernels are verified against)</li>
</ul>

<h2>Explanation</h2>
<p class="sub">Understanding-oriented — how it's built, and why.</p>
<ul>
  <li><a href="https://github.com/haixuanTao/zealot/blob/master/docs/explanation.md">How zealot is built, and why</a> — 100% Rust on any GPU, one source / three compilers, the three layers people conflate, the GPU-resident loop, sim2sim as a trust discipline</li>
</ul>
</body></html>
HTML

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cd "$WEBSITE_DIR"
SHA=$(git rev-parse --short HEAD)

git init -q "$TMP"
cp -R "$DIST"/. "$TMP"/
cd "$TMP"
git add -A
git -c user.email=deploy@zealot -c user.name=deploy commit -qm "Deploy website — based on $SHA"
git push -q --force "$(cd "$WEBSITE_DIR" && git remote get-url "$REMOTE")" "HEAD:$BRANCH"

echo "Deployed $SHA → $BRANCH (https://haixuantao.github.io/zealot/)"

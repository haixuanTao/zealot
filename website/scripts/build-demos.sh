#!/bin/bash

# Build the website's WASM demo(s).
# Usage: ./scripts/build-demos.sh [demo_name]
#
# The browser demo is the Unitree G1 on the ZEALOT training env (nexus GPU
# physics, unified stack): the `g1_web` cargo example of the zealot workspace,
# compiled to wasm32-unknown-unknown and post-processed with wasm-bindgen
# (+ optional wasm-opt). The MJCF is embedded — no bake step.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBSITE_DIR="$(dirname "$SCRIPT_DIR")"
ZEALOT_DIR="$(dirname "$WEBSITE_DIR")"
DEMOS_DIR="$WEBSITE_DIR/public/demos"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# demo name (output dir under public/demos/) -> zealot cargo example (the
# required feature has the same name as the example).
DEMOS=(g1_web g1_terrain_web)
bin_of() {
    case "$1" in
        g1_web) echo g1_web ;;
        g1_terrain_web) echo g1_terrain_web ;;
        lerobot_biped_web) echo lerobot_biped_web ;;
        *) echo "" ;;
    esac
}

check_requirements() {
    local missing=()
    if ! command -v cargo &> /dev/null; then
        missing+=("cargo")
    fi
    if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
        missing+=("wasm32-unknown-unknown target (install with: rustup target add wasm32-unknown-unknown)")
    fi
    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}Error: Missing required tools:${NC}"
        for tool in "${missing[@]}"; do
            echo "  - $tool"
        done
        exit 1
    fi
}

# wasm-bindgen requires the CLI version to exactly match the wasm-bindgen crate
# version in the nexus Cargo.lock. If the globally installed CLI doesn't match,
# install the right version locally under zealot target/ (leaves the global
# install alone).
resolve_wasm_bindgen() {
    local required
    required=$(grep -A1 '^name = "wasm-bindgen"$' "$ZEALOT_DIR/Cargo.lock" | grep '^version' | cut -d'"' -f2)
    if [ -z "$required" ]; then
        echo -e "${RED}Could not determine wasm-bindgen version from $ZEALOT_DIR/Cargo.lock${NC}" >&2
        exit 1
    fi

    if command -v wasm-bindgen &> /dev/null && [ "$(wasm-bindgen --version | awk '{print $2}')" = "$required" ]; then
        WASM_BINDGEN=wasm-bindgen
        return
    fi

    local local_root="$ZEALOT_DIR/target/wasm-bindgen-cli/$required"
    WASM_BINDGEN="$local_root/bin/wasm-bindgen"
    if [ ! -x "$WASM_BINDGEN" ]; then
        echo -e "${BLUE}Installing wasm-bindgen-cli $required (to match Cargo.lock) into zealot target/...${NC}"
        cargo install wasm-bindgen-cli --version "$required" --root "$local_root"
    fi
}

# Tunables (override via environment):
#   WASM_OPT_FLAGS=…  wasm-opt optimization flags      (default: -O3)
#   SKIP_WASM_OPT=1   skip wasm-opt entirely (fast iteration; larger .wasm)
WASM_OPT_FLAGS="${WASM_OPT_FLAGS:--O3}"

cargo_build() {
    local demo=$1
    cargo build \
        --manifest-path "$ZEALOT_DIR/Cargo.toml" \
        --target wasm32-unknown-unknown \
        --release \
        --example "$(bin_of "$demo")" \
        --features "$(bin_of "$demo")"
}

# Post-process one already-compiled demo: wasm-bindgen + wasm-opt + index.html.
postprocess_demo() {
    local demo=$1
    local bin
    bin=$(bin_of "$demo")
    local demo_dir="$DEMOS_DIR/$demo"
    local target_dir="$ZEALOT_DIR/target/wasm32-unknown-unknown/release/examples"

    mkdir -p "$demo_dir/pkg"

    if [ ! -f "$target_dir/$bin.wasm" ]; then
        echo -e "${RED}✗${NC} $demo (no .wasm — cargo build failed?)"
        return 1
    fi

    if ! "$WASM_BINDGEN" \
        "$target_dir/$bin.wasm" \
        --out-dir "$demo_dir/pkg" \
        --out-name example \
        --target web \
        --no-typescript 2>&1; then
        echo -e "${RED}✗${NC} $demo (wasm-bindgen failed)"
        return 1
    fi

    # Optimize with wasm-opt if available (skippable for fast iteration)
    if [ -z "$SKIP_WASM_OPT" ] && command -v wasm-opt &> /dev/null; then
        wasm-opt $WASM_OPT_FLAGS "$demo_dir/pkg/example_bg.wasm" -o "$demo_dir/pkg/example_bg.wasm" 2>/dev/null || true
    fi

    write_index_html "$demo_dir"

    echo -e "${GREEN}✓${NC} $demo"
    return 0
}

write_index_html() {
    local demo_dir=$1
    # Create index.html for the demo
    cat > "$demo_dir/index.html" << 'HTMLEOF'
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>zealot demo — Unitree G1 on nexus</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body {
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #0d2b2e;
    }
    canvas {
      width: 100% !important;
      height: 100% !important;
      display: block;
    }
    .loading {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: #35c1cd;
      font-family: system-ui, sans-serif;
      font-size: 14px;
      text-align: center;
      max-width: 80%;
    }
    .loading::after {
      content: '';
      display: block;
      width: 30px;
      height: 30px;
      margin: 10px auto;
      border: 3px solid #1c4a4f;
      border-top-color: #35c1cd;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    .error {
      color: #ff6b6b;
    }
    .error::after {
      display: none;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="loading" id="loading">Loading WebAssembly (GPU physics)...</div>
  <script>
    // Mirror the first WebGPU/shader error into the tab TITLE. On Safari and
    // Firefox the devtools console can't be reached from automation, but the
    // title can (AppleScript / WebDriver) — this is how cross-browser GPU
    // failures get diagnosed at all. The wasm rewrites the title each second;
    // this setter appends the captured error to whatever it writes.
    (() => {
      let firstErr = '';
      const nat = Object.getOwnPropertyDescriptor(Document.prototype, 'title');
      Object.defineProperty(document, 'title', {
        configurable: true,
        get: () => nat.get.call(document),
        set: (v) => nat.set.call(document, firstErr ? v + ' || ' + firstErr : v),
      });
      for (const kind of ['error', 'warn']) {
        const orig = console[kind].bind(console);
        console[kind] = (...a) => {
          const s = a.map((x) => (typeof x === 'string' ? x : (x && x.message) || '')).join(' ');
          if (!firstErr && /invalid|validation|shader|pipeline|error|panic/i.test(s)) {
            firstErr = (kind + ': ' + s).replace(/\s+/g, ' ').slice(0, 150);
            // Publish straight away: if the wasm loop is what died, it will
            // never write the title again and the error would be invisible.
            nat.set.call(document, 'ERR| ' + firstErr);
          }
          orig(...a);
        };
      }
    })();
  </script>
  <script>
    // Render at a capped pixel ratio. winit sizes the drawing buffer from
    // window.devicePixelRatio, so a retina display renders 4x the pixels —
    // measured to matter a lot: the sim and the renderer share one GPU, and
    // at DPR 2 the WebGL pass eats the budget the physics needs. ?dpr=
    // overrides (100ths, e.g. 150 = 1.5).
    (() => {
      const q = new URLSearchParams(location.search);
      const dpr = Math.max(0.5, Math.min(3, (Number(q.get('dpr')) || 100) / 100));
      Object.defineProperty(window, 'devicePixelRatio', {get: () => dpr});
    })();
  </script>
  <script>
    // The page embedding this demo scrolls; the canvas would otherwise eat
    // every wheel event. Scrolling DOWN is forwarded to the parent so the
    // story below the fold stays reachable — the wheel still zooms the
    // camera OUT (scroll up), which is all that's useful since the demo
    // opens already framed on one robot.
    addEventListener('wheel', (e) => {
      if (e.deltaY > 0 && parent !== window) {
        e.preventDefault();
        e.stopPropagation();
        parent.postMessage({zealotScroll: e.deltaY}, location.origin);
      }
    }, {capture: true, passive: false});
  </script>
  <script type="module">
    import init from './pkg/example.js';

    const loading = document.getElementById('loading');

    if (!navigator.gpu) {
      loading.className = 'loading error';
      loading.textContent = 'WebGPU is not available in this browser. '
        + 'The simulation runs its physics as WebGPU compute shaders and '
        + 'requires Chrome (or another Chromium-based browser). '
        + 'Safari and Firefox are not supported.';
    } else {
      init().then(() => {
        loading.style.display = 'none';
      }).catch(err => {
        console.error('WASM Error:', err);
        loading.className = 'loading error';
        loading.textContent = 'Error: ' + err.message;
      });
    }
  </script>
</body>
</html>
HTMLEOF
}

# Main logic
check_requirements
resolve_wasm_bindgen

if [ -n "$1" ]; then
    if [ -z "$(bin_of "$1")" ]; then
        echo -e "${RED}Unknown demo '$1'.${NC} Available demos: ${DEMOS[*]}"
        exit 1
    fi
    echo -e "${BLUE}Building${NC} $1..."
    cargo_build "$1"
    postprocess_demo "$1"
else
    failed=0
    for demo in "${DEMOS[@]}"; do
        echo -e "${BLUE}Building${NC} $demo..."
        cargo_build "$demo"
        postprocess_demo "$demo" || failed=$((failed + 1))
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}Success:${NC} $(( ${#DEMOS[@]} - failed ))"
    if [ $failed -gt 0 ]; then
        echo -e "${RED}Failed:${NC} $failed"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    [ $failed -eq 0 ] || exit 1
fi

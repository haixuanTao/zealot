//! Website demo: a 10-robot Unitree G1 fleet walking over the rough-terrain
//! strips it was trained on (BIPED_TERRAIN=1 — box plateaus, terrain gets
//! harder with distance; robots spread over the three family strips,
//! family = env % 3). See `g1_web_demo.rs` for the shared implementation and
//! `g1_web.rs` for the flat fleet (native/bench use; no longer on the site).
//!
//! Native run:  `cargo run --release --example g1_terrain_web --features g1_terrain_web`
//! Smoke check: `... -- --headless-check`
//! Wasm build:  `website/scripts/build-demos.sh g1_terrain_web`

#[path = "cutile_gemm.rs"]
mod cutile_gemm;
#[path = "gpu_policy.rs"]
mod gpu_policy;
#[path = "g1_web_demo.rs"]
mod g1_web_demo;

/// URL query param `key=` parsed as an integer (wasm only).
#[allow(unused)]
fn query_int(key: &str) -> Option<i64> {
    #[cfg(target_arch = "wasm32")]
    if let Some(search) = web_sys::window().and_then(|w| w.location().search().ok()) {
        for kv in search.trim_start_matches('?').split('&') {
            if let Some(v) = kv.strip_prefix(key) {
                if let Ok(n) = v.parse::<i64>() {
                    return Some(n);
                }
            }
        }
    }
    None
}

#[kiss3d::main]
pub async fn main() {
    g1_web_demo::run(g1_web_demo::DemoCfg {
        n_robots: query_int("n=").map_or(10, |n| n.clamp(1, 60) as usize),
        terrain: true,
        // `?lvl=` picks the spawn difficulty patch (0..=19, default 4).
        terrain_level: query_int("lvl=").map_or(4, |l| l.clamp(0, 19) as u32),
        // `?amp=` scales terrain amplitude in percent (100 = training terrain).
        terrain_amp_pct: query_int("amp=").map_or(100, |a| a.clamp(0, 300) as u32),
        // `?slope=` adds an uphill grade along the strip, in degrees.
        terrain_slope_deg: query_int("slope=").map_or(0, |s| s.clamp(0, 20) as u32),
    })
    .await
}

//! Website demo: a 10-robot Unitree G1 fleet walking the released v7 policy
//! on flat ground — one batched zealot/nexus GPU env, ten independent
//! physics batches. See `g1_web_demo.rs` for the shared implementation and
//! `g1_terrain_web.rs` for the single-robot rough-terrain variant.
//!
//! Native run:  `cargo run --release --example g1_web --features g1_web`
//! Smoke check: `... --example g1_web --features g1_web -- --headless-check`
//! Wasm build:  `website/scripts/build-demos.sh g1_web`

#[path = "../../src/biped/cutile_gemm.rs"]
mod cutile_gemm;
#[path = "../../src/biped/gpu_policy.rs"]
mod gpu_policy;
#[path = "g1_web_demo.rs"]
mod g1_web_demo;

/// Robot count: default 10; on the web, override with `?n=` (clamped 1..=200)
/// — "how many robots can your GPU walk?".
fn robot_count() -> usize {
    #[cfg(target_arch = "wasm32")]
    if let Some(search) = web_sys::window().and_then(|w| w.location().search().ok()) {
        for kv in search.trim_start_matches('?').split('&') {
            if let Some(v) = kv.strip_prefix("n=") {
                if let Ok(n) = v.parse::<usize>() {
                    return n.clamp(1, 200);
                }
            }
        }
    }
    10
}

/// `--dump-terrain <out.json> [x0 x1]`: write the Boxes strip's mesh (x in
/// [x0, x1), default [0, 48) — the easy end) as JSON {v:[[x,y,z]..],
/// i:[[a,b,c]..]} for the rapier bench — SAME deterministic geometry the
/// terrain demo simulates.
#[cfg(not(target_arch = "wasm32"))]
fn dump_terrain(out: &str, x0: f32, x1: f32) {
    use zealot_env::terrain::{TerrainFamily, TerrainStrip};
    let (v, t) = TerrainStrip::generate(TerrainFamily::Boxes, 0xC0FFEE).mesh();
    let keep: Vec<[u32; 3]> = t
        .into_iter()
        .filter(|tri| tri.iter().all(|&k| (x0..x1).contains(&v[k as usize][0])))
        .collect();
    let mut s = String::from("{\"v\":[");
    for (i, p) in v.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("[{:.4},{:.4},{:.4}]", p[0], p[1], p[2]));
    }
    s.push_str("],\"i\":[");
    for (i, tri) in keep.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("[{},{},{}]", tri[0], tri[1], tri[2]));
    }
    s.push_str("]}");
    std::fs::write(out, s).expect("write terrain json");
    println!("wrote {out} ({} tris)", keep.len());
}

/// URL query param `key=` as a string (wasm only). Everything up to the next
/// `&`, still percent-encoded — the demo decodes it.
#[allow(unused)]
fn query_str(key: &str) -> Option<String> {
    #[cfg(target_arch = "wasm32")]
    if let Some(search) = web_sys::window().and_then(|w| w.location().search().ok()) {
        for kv in search.trim_start_matches('?').split('&') {
            if let Some(v) = kv.strip_prefix(key) {
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

#[kiss3d::main]
pub async fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let args: Vec<String> = std::env::args().collect();
        if let Some(i) = args.iter().position(|a| a == "--dump-terrain") {
            let f = |k: usize, d: f32| args.get(i + k).and_then(|a| a.parse().ok()).unwrap_or(d);
            dump_terrain(&args[i + 1], f(2, 0.0), f(3, 48.0));
            return;
        }
    }
    g1_web_demo::run(g1_web_demo::DemoCfg {
        n_robots: robot_count(),
        terrain: false,
        terrain_level: 4,
        terrain_amp_pct: 100,
        terrain_slope_deg: 0,
        // `?ckpt=` runs a published policy instead of the embedded one:
        // a Hugging Face `owner/repo/file.safetensors`, or a full URL.
        ckpt: query_str("ckpt="),
    })
    .await
}


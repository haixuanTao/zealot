//! TEMPORARY debug probe: compare narrow-phase contact statistics between
//! backends (KHAL_BACKEND=webgpu vs metal) on the terrain scene. Steps with
//! zero actions and dumps per-step contact count / manifold-point / depth
//! aggregates. Deterministic seeds — any backend divergence is the engine's.
//!
//!   BIPED_TERRAIN=1 KHAL_BACKEND=metal cargo run --release --bin contact_probe \
//!       --features "gpu biped_gpu" -- [num_envs] [steps]

#[path = "../biped/biped_env.rs"]
mod biped_env;
#[path = "../biped/biped_env_nexus.rs"]
mod biped_env_nexus;

use biped_env_nexus::{BipedNexusBatchEnv, default_mjcf_path};
use zealot_env::robots::NUM_JOINTS;

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let steps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");

    pollster::block_on(async {
        let mut env = BipedNexusBatchEnv::new(&xml, n, 32, 0xC0FFEE).await;
        let _ = env.initial_obs().await;
        // Spawn poses BEFORE any step: torso world position + the terrain
        // collider's world position per env (are the bodies/terrain where the
        // CPU thinks they are?).
        {
            let poses = env.slurp_poses().await;
            let cpb = poses.len() / n;
            for e in 0..n.min(8) {
                let t = &poses[e * cpb + 12]; // torso-ish link (foot=12 used in contacts)
                let ter = &poses[e * cpb + 27]; // terrain collider (cx=27 in dumps)
                println!(
                    "[p0] e={e} link12=({:.4},{:.4},{:.4}) terrain=({:.4},{:.4},{:.4})",
                    t.translation.x, t.translation.y, t.translation.z,
                    ter.translation.x, ter.translation.y, ter.translation.z,
                );
            }
        }
        let actions = vec![[0.0f32; NUM_JOINTS]; n];

        for t in 0..steps {
            let _ = env.step(&actions).await;
            let (lens, contacts) = env.dbg_contacts().await;
            let cap = contacts.len() / lens.len();
            let total: u32 = lens.iter().sum();
            let mut pts = 0u32;
            let mut depth_sum = 0f64;
            let mut depth_min = f32::INFINITY;
            let mut fric_sum = 0f64;
            for (b, &len) in lens.iter().enumerate() {
                for i in 0..(len as usize).min(cap) {
                    let c = &contacts[b * cap + i];
                    pts += c.contact.len;
                    fric_sum += c.friction as f64;
                    for k in 0..(c.contact.len as usize).min(4) {
                        let d = c.contact.points_a[k].dist;
                        depth_sum += d as f64;
                        if d < depth_min {
                            depth_min = d;
                        }
                    }
                }
            }
            println!(
                "t={t} contacts={total} pts={pts} depth_sum={depth_sum:.4} depth_min={depth_min:.4} fric_sum={fric_sum:.3} len0={}",
                lens[0]
            );
            if t == 0 {
                let poses = env.slurp_poses().await;
                let cpb = poses.len() / n;
                for e in 0..n.min(8) {
                    let t12 = &poses[e * cpb + 12];
                    println!(
                        "[p1] e={e} post-step link12=({:.4},{:.4},{:.4})",
                        t12.translation.x, t12.translation.y, t12.translation.z
                    );
                }
                env.dbg_dump_batch_indices();
                if std::env::var("RESYNC").is_ok() {
                    env.dbg_resync_collider_poses();
                }
                let cwp = env.dbg_collider_world_poses().await;
                let cpb = cwp.len() / n;
                for e in 0..n.min(8) {
                    let f = &cwp[e * cpb + 12];
                    println!(
                        "[cw] e={e} coll12=({:.4},{:.4},{:.4})",
                        f.translation.x, f.translation.y, f.translation.z
                    );
                }
                let (plens, ppairs) = env.dbg_pfm_pairs().await;
                let pcap = ppairs.len() / plens.len();
                let mut prows: Vec<String> = Vec::new();
                for (b, &len) in plens.iter().enumerate() {
                    for i in 0..(len as usize).min(pcap) {
                        let pp = &ppairs[b * pcap + i];
                        prows.push(format!(
                            "b={b} cx={} cy={} feat={} p12=({:.4},{:.4},{:.4}) th=({:.4},{:.4})",
                            pp.colliders[0], pp.colliders[1], pp.feature_id,
                            pp.pose12.translation.x, pp.pose12.translation.y, pp.pose12.translation.z,
                            pp.thickness1, pp.thickness2,
                        ));
                    }
                }
                prows.sort();
                for r in prows { println!("[pf] {r}"); }
                // Full first-step contact listing (state still identical across
                // backends): one line per record, sorted for a stable diff.
                let mut rows: Vec<String> = Vec::new();
                for (b, &len) in lens.iter().enumerate() {
                    for i in 0..(len as usize).min(cap) {
                        let c = &contacts[b * cap + i];
                        rows.push(format!(
                            "b={b} cx={} cy={} feat={} n={} d0={:.5} nrm=({:.4},{:.4},{:.4})",
                            c.colliders[0],
                            c.colliders[1],
                            c._padding[0].to_bits(),
                            c.contact.len,
                            c.contact.points_a[0].dist,
                            c.contact.normal_a[0],
                            c.contact.normal_a[1],
                            c.contact.normal_a[2],
                        ));
                    }
                }
                rows.sort();
                for r in rows {
                    println!("[c0] {r}");
                }
            }
        }
    });
}

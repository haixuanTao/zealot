//! Foot↔ground contact diagnostic for the WebGpu/Metal physics bug.
//!
//! The biped free-falls through the floor on WebGpu (works on CUDA). This probe
//! builds the env, steps a few times with zero action, and reads back the
//! narrow-phase contact manifolds (`env.dbg_contacts()`) to localize the bug:
//!   - 0 contacts            → narrow-phase isn't generating foot-ground pairs.
//!   - contacts, normal +z   → narrow-phase OK; bug is the multibody contact solve.
//!   - contacts, normal −z    → flipped normal (matches the "more iters = worse" clue).
//!
//! Run: `cargo run --release --example contact_probe --features "gpu biped_gpu" -- [steps]`

#[path = "biped_env.rs"]
mod biped_env;
#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "gpu_policy.rs"]
mod gpu_policy;

use biped_env_nexus::{default_mjcf_path, BipedNexusBatchEnv};
use zealot_env::robots::lerobot_bipedal::NUM_JOINTS;

fn main() {
    let steps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let n = 4usize;
    let xml = std::fs::read_to_string(default_mjcf_path()).expect("mjcf");

    pollster::block_on(async {
        let mut env = BipedNexusBatchEnv::new(&xml, n, 32, 0xC0FFEE).await;
        let zero = vec![[0.0f32; NUM_JOINTS]; n];
        let ground0 = env.ground_collider(0);
        println!("env0 ground collider index = {ground0}");

        for s in 0..steps {
            let outs = env.step(&zero).await;
            let torso = {
                let zs = env.torso_heights().await;
                zs.iter().sum::<f32>() / n as f32
            };
            let (plen, pairs) = env.dbg_collision_pairs().await;
            let plen_sum: u32 = plen.iter().copied().sum();
            let ground_pairs = pairs
                .iter()
                .filter(|p| p[0] == ground0 || p[1] == ground0)
                .count();
            let (len, manifolds) = env.dbg_contacts().await;
            let poses = env.snapshot().await; // collider world poses (for world-normal)
            // Rotate a local vector by a quaternion (x,y,z,w).
            let qrot = |q: (f32, f32, f32, f32), v: (f32, f32, f32)| -> (f32, f32, f32) {
                let (qx, qy, qz, qw) = q;
                let (vx, vy, vz) = v;
                // t = 2 * cross(q.xyz, v)
                let tx = 2.0 * (qy * vz - qz * vy);
                let ty = 2.0 * (qz * vx - qx * vz);
                let tz = 2.0 * (qx * vy - qy * vx);
                // v' = v + qw*t + cross(q.xyz, t)
                (
                    vx + qw * tx + (qy * tz - qz * ty),
                    vy + qw * ty + (qz * tx - qx * tz),
                    vz + qw * tz + (qx * ty - qy * tx),
                )
            };

            // Global stats: how many manifolds carry contact points, and the
            // sign distribution of the contact normal's z component.
            let mut active = 0usize;
            let (mut up, mut down, mut flat) = (0usize, 0usize, 0usize);
            for m in &manifolds {
                if m.contact.len == 0 {
                    continue;
                }
                active += 1;
                let nz = m.contact.normal_a.z;
                if nz > 0.5 {
                    up += 1;
                } else if nz < -0.5 {
                    down += 1;
                } else {
                    flat += 1;
                }
            }
            let len_sum: u32 = len.iter().copied().sum();
            println!(
                "\nstep {s}: torso_z={torso:.3} fell={}",
                outs.iter().filter(|o| o.fell).count()
            );
            println!(
                "   broad-phase: pairs_len_sum={plen_sum} total_pairs={} ground_pairs={ground_pairs}",
                pairs.iter().filter(|p| p[0] != 0 || p[1] != 0).count()
            );
            println!(
                "   narrow-phase: contacts_len_sum={len_sum} active_manifolds={active} (normal_z: +z={up} -z={down} ~0={flat})"
            );
            // Multibody contact constraints: effective inv-mass inv_lhs, rhs, impulse.
            let (cc_cnt, ccs) = env.dbg_mb_contacts().await;
            let cc_active = ccs.iter().filter(|c| c.inv_lhs != 0.0 || c.impulse != 0.0).count();
            println!(
                "   mb-contact-constraints: counts={:?} active={cc_active}",
                &cc_cnt[..cc_cnt.len().min(8)]
            );
            for c in ccs.iter().filter(|c| c.inv_lhs != 0.0 || c.impulse != 0.0).take(6) {
                println!(
                    "      mb={} link={} kind={} inv_lhs={:.3e} rhs={:.3e} impulse={:.3e} free_im={:.3} lin_jac=({:.2},{:.2},{:.2})",
                    c.multibody_id, c.link_id, c.kind, c.inv_lhs, c.rhs, c.impulse, c.free_body_im,
                    c.lin_jac.x, c.lin_jac.y, c.lin_jac.z
                );
            }
            // DEBUG markers (colliders==7777): normal_a=(capacity, pairs_len, num_wg.y)
            for m in &manifolds {
                if m.colliders.x == 7777 && m.colliders.y == 7777 {
                    let d = m.contact.normal_a;
                    println!(
                        "   [DBG] batch: contacts_batch_capacity={:.0} pairs_len={:.0} num_workgroups.y={:.0}",
                        d.x, d.y, d.z
                    );
                }
            }

            // Detail: manifolds touching env0's ground collider.
            let mut shown = 0;
            for m in &manifolds {
                if m.contact.len == 0 {
                    continue;
                }
                let (ca, cb) = (m.colliders.x, m.colliders.y);
                if ca == ground0 || cb == ground0 {
                    let nrm = m.contact.normal_a;
                    let p0 = m.contact.points_a[0];
                    // normal_a is in collider `ca`'s (colliders.x) local frame; rotate
                    // to world by ca's quaternion. World normal should be ~vertical (±z).
                    let wn = poses
                        .get(ca as usize)
                        .map(|p| {
                            let r = p.rotation;
                            qrot((r.x, r.y, r.z, r.w), (nrm.x, nrm.y, nrm.z))
                        })
                        .unwrap_or((0.0, 0.0, 0.0));
                    println!(
                        "   ground pair: ({ca},{cb}) len={} normal_local=({:.2},{:.2},{:.2}) WORLD=({:.2},{:.2},{:.2}) dist0={:.4}",
                        m.contact.len, nrm.x, nrm.y, nrm.z, wn.0, wn.1, wn.2, p0.dist
                    );
                    shown += 1;
                    if shown >= 6 {
                        break;
                    }
                }
            }
            if shown == 0 && active > 0 {
                // No ground pairs but some contacts exist — show a sample.
                for m in &manifolds {
                    if m.contact.len == 0 {
                        continue;
                    }
                    let nrm = m.contact.normal_a;
                    println!(
                        "   sample pair: colliders=({},{}) len={} normal_a=({:.3},{:.3},{:.3})",
                        m.colliders.x, m.colliders.y, m.contact.len, nrm.x, nrm.y, nrm.z
                    );
                    break;
                }
            }
        }
    });
}

//! Pin zealot's G1 actuator parametrization against WBC-AGILE's.
//!
//! The reference table is GENERATED from AGILE's own source
//! (`G1_29DOF_DELAYED_DC_MOTOR` + the Velocity-G1-History task cfg) by
//! `tools/pull_agile_actuators.py` — re-run it after updating the WBC-AGILE
//! checkout, then re-run these tests. Two guarantees:
//!
//! 1. `unitree_g1_agile()` matches AGILE exactly on every parameter zealot
//!    models (kp, kd, effort, vel limit, armature, action scale, default
//!    pose). The DC torque-speed saturation column is carried in the table
//!    but not yet asserted (nexus models a constant clamp — known gap).
//! 2. The default `unitree_g1()` (unitree_rl_gym provenance) deviates from
//!    AGILE in EXACTLY the known set of parameters — so a silent change on
//!    either side (a zealot edit or an upstream AGILE retune) fails a test
//!    instead of silently skewing cross-stack comparisons.

use zealot_env::robots::unitree_g1::{unitree_g1, unitree_g1_agile};

include!("data/agile_g1_actuators.rs");

#[test]
fn agile_actuator_parity() {
    let spec = unitree_g1_agile();
    for (i, &(name, kp, kd, effort, vel, armature, _sat, q0)) in
        AGILE_G1_ACTUATORS.iter().enumerate()
    {
        let j = &spec.joints[i];
        assert_eq!(j.name, name, "joint order mismatch at {i}");
        assert_eq!(j.kp, kp, "{name} kp");
        assert_eq!(j.kd, kd, "{name} kd");
        assert_eq!(j.effort_limit, effort, "{name} effort");
        assert_eq!(j.vel_limit, vel, "{name} vel limit");
        assert_eq!(j.armature, armature, "{name} armature");
        assert_eq!(j.action_scale, AGILE_ACTION_SCALE, "{name} action scale");
        assert_eq!(j.default_pos, q0, "{name} default pos");
    }
}

/// The default G1 (unitree_rl_gym gains) is NOT AGILE-parametrized — assert
/// the exact deviation set so drift on either side is caught loudly.
#[test]
fn default_g1_known_deviations_from_agile() {
    let spec = unitree_g1();
    for (i, &(name, kp, kd, effort, vel, armature, _sat, q0)) in
        AGILE_G1_ACTUATORS.iter().enumerate()
    {
        let j = &spec.joints[i];
        assert_eq!(j.default_pos, q0, "{name}: default pose SHOULD match");
        assert_eq!(j.action_scale, 0.25, "{name}: rl_gym action scale");
        assert_eq!(j.armature, 0.01, "{name}: rl_gym armature");
        if name.contains("hip_roll") {
            // rl_gym bins hip_roll by the URDF limit (139 @ 20); AGILE bins it
            // with the other hips (88 @ 32).
            assert_eq!((j.effort_limit, j.vel_limit), (139.0, 20.0), "{name}");
            assert_eq!((effort, vel), (88.0, 32.0), "{name} (AGILE side)");
        } else {
            assert_eq!(j.effort_limit, effort, "{name}: effort SHOULD match");
        }
        if name.contains("knee") {
            assert_eq!((j.kp, kp), (150.0, 200.0), "{name} kp deviation");
            assert_eq!((j.kd, kd), (4.0, 5.0), "{name} kd deviation");
        } else if name.contains("ankle") {
            assert_eq!((j.kp, kp), (40.0, 20.0), "{name} kp deviation");
            assert_eq!(j.kd, 2.0, "{name} rl_gym kd");
            assert!(kd <= 0.2, "{name} AGILE kd is near-zero");
        } else {
            assert_eq!(j.kp, kp, "{name}: hip kp SHOULD match");
            assert_eq!((j.kd, kd), (2.0, 2.5), "{name} kd deviation");
        }
    }
}

//! Minimal vector / quaternion helpers.
//!
//! zealot-env builds with zero external dependencies (so it compiles without the
//! `cargo-gpu` toolchain), so we can't lean on `glam`/`nalgebra` here. These are
//! the handful of operations the MDP needs: rotating a world vector into the base
//! frame and back, and reading a body axis out of a quaternion. Quaternions are
//! `(x, y, z, w)` to match nexus's `Rotation` readback.

/// 3-vector dot product.
#[inline]
pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Rotate vector `v` by unit quaternion `q = (x, y, z, w)` (body→world if `q` is
/// the body's orientation). Uses `v' = v + 2·q_xyz × (q_xyz × v + w·v)`.
#[inline]
pub fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let w = q[3];
    let t = cross(u, v);
    let t = [t[0] + w * v[0], t[1] + w * v[1], t[2] + w * v[2]];
    let tt = cross(u, t);
    [v[0] + 2.0 * tt[0], v[1] + 2.0 * tt[1], v[2] + 2.0 * tt[2]]
}

/// Rotate `v` by the inverse of `q` (world→body for a body orientation `q`).
#[inline]
pub fn quat_rotate_inv(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    quat_rotate([-q[0], -q[1], -q[2], q[3]], v)
}

/// 3-vector cross product.
#[inline]
pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_quat_is_noop() {
        let q = [0.0, 0.0, 0.0, 1.0];
        let v = [1.0, 2.0, 3.0];
        let r = quat_rotate(q, v);
        for k in 0..3 {
            assert!((r[k] - v[k]).abs() < 1e-6);
        }
    }

    #[test]
    fn rotate_then_inverse_recovers() {
        // 90° about X: (x,y,z,w) = (sin45, 0, 0, cos45).
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let q = [s, 0.0, 0.0, s];
        let v = [0.0, 1.0, 0.0];
        let r = quat_rotate(q, v);
        // +Y rotated +90° about X → +Z.
        assert!((r[0]).abs() < 1e-6 && (r[1]).abs() < 1e-6 && (r[2] - 1.0).abs() < 1e-6);
        let back = quat_rotate_inv(q, r);
        for k in 0..3 {
            assert!((back[k] - v[k]).abs() < 1e-6);
        }
    }
}

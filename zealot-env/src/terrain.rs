//! Rough-terrain generation + difficulty curriculum (`BIPED_TERRAIN=1`),
//! ported from WBC-AGILE's `MEDIUM_ROUGH_TERRAIN_CFG` +
//! `terrain_levels_vel_curriculum` so zealot trains under the same
//! stepping-forcing mechanism as NVIDIA's pipeline.
//!
//! Layout: each terrain FAMILY is one strip of [`ROWS`] 8×8 m patches laid
//! along +X starting at x = +[`PATCH`] (the origin stays flat, so the
//! as-constructed state — every robot at (0,0) — never touches terrain and
//! the flag-off world is unchanged). Row `p`'s difficulty is
//! `d = (p + U(0,1)) / ROWS` (AGILE's per-patch jitter), heights quantized to
//! 5 mm (`vertical_scale`). Families (equal env split, AGILE's proportions):
//!
//! - **Boxes**: 0.15 m checker cells, each cell's top at `U(−g, +g)`,
//!   `g = 0.04·d` (row 19 ≈ ±39 mm). AGILE builds true box meshes; we sample
//!   the same field on a 0.075 m grid, turning vertical faces into one-node
//!   steps — documented deviation (steepness bounded by grid resolution).
//!   AGILE's tiny 0.1 m center platform is not reproduced (spawns are
//!   clearance-checked instead).
//! - **Rough**: node heights `U[0.01·d, 0.2·d]` (one-sided) on a 0.4 m
//!   lattice (row 19 ≈ up to 0.195 m), cosine-interpolated to the mesh grid
//!   (AGILE uses bicubic; interpolation fidelity is a knob, bounds match).
//! - **Wave**: `h = A/2·cos(2πy/λ) + A/2·sin(2πx/λ)`, `A = 0.01 + 0.24·d`,
//!   `λ = 8/3 m` (row 19 ≈ ±0.24 m peaks), patch-local coordinates.
//!
//! The curriculum is AGILE's exactly: per env, traveled distance (chord sum
//! between command resamples) > 4 m counts a success, < 2 m a failure
//! (cumulative, not consecutive); 4 successes promote, 10 failures demote,
//! either move zeroes BOTH counters; promotion past the top row reassigns a
//! uniform random row; demotion clamps at 0; initial level ~ U{0, 1}.

use crate::rng::Lcg;

/// Patch edge length (m) — AGILE's `size=(8.0, 8.0)`.
pub const PATCH: f32 = 8.0;
/// Number of difficulty rows — AGILE's `num_rows=20`.
pub const ROWS: usize = 20;
/// Height quantization (m) — AGILE's `vertical_scale`.
pub const VERTICAL_SCALE: f32 = 0.005;
/// The strip starts here (origin patch is flat ground).
pub const STRIP_X0: f32 = PATCH;
/// Strip half-width in Y.
pub const STRIP_HALF_W: f32 = PATCH / 2.0;
/// Thickness of the closed terrain slab below z=0.
pub const SLAB_BOTTOM: f32 = -0.05;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TerrainFamily {
    Boxes,
    Rough,
    Wave,
}

impl TerrainFamily {
    /// Fixed per-env family assignment (AGILE: column families are fixed).
    pub fn of_env(env_id: usize) -> Self {
        match env_id % 3 {
            0 => TerrainFamily::Boxes,
            1 => TerrainFamily::Rough,
            _ => TerrainFamily::Wave,
        }
    }

    /// Mesh grid spacing: boxes need cell-edge fidelity (0.15/2); the smooth
    /// families have ≥0.4 m features.
    pub fn grid_spacing(self) -> f32 {
        match self {
            TerrainFamily::Boxes => 0.075,
            _ => 0.2,
        }
    }
}

/// One family's 20-patch strip: a regular height grid plus the exact
/// piecewise-linear sampler matching the emitted mesh.
pub struct TerrainStrip {
    pub family: TerrainFamily,
    hs: f32,
    /// Cells along X / Y (nodes are +1).
    nx: usize,
    ny: usize,
    /// Node heights, row-major: `heights[j * (nx+1) + i]` at
    /// `(STRIP_X0 + i·hs, −STRIP_HALF_W + j·hs)`.
    heights: Vec<f32>,
}

fn quantize(h: f32) -> f32 {
    (h / VERTICAL_SCALE).round() * VERTICAL_SCALE
}

impl TerrainStrip {
    /// World-space center of a difficulty row's patch.
    pub fn patch_center(level: u32) -> (f32, f32) {
        (STRIP_X0 + PATCH * level as f32 + PATCH / 2.0, 0.0)
    }

    pub fn generate(family: TerrainFamily, seed: u64) -> Self {
        let hs = family.grid_spacing();
        let nx = (PATCH * ROWS as f32 / hs).round() as usize;
        let ny = (2.0 * STRIP_HALF_W / hs).round() as usize;
        // Spread the seed bits — `Lcg::new` ORs bit 0, so raw adjacent seeds
        // (42 vs 43) would otherwise collide.
        let mut rng = Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x7E44_A11);
        let mut heights = vec![0.0f32; (nx + 1) * (ny + 1)];

        for p in 0..ROWS {
            // AGILE: difficulty = (row + U(0,1)) / num_rows.
            let d = (p as f32 + rng.range(0.0, 1.0)) / ROWS as f32;
            let i0 = (PATCH * p as f32 / hs).round() as usize;
            let i1 = (PATCH * (p + 1) as f32 / hs).round() as usize;
            match family {
                TerrainFamily::Boxes => {
                    // 0.15 m checker cells; each cell top ~ U(−g, +g), g = 0.04·d.
                    let g = 0.04 * d;
                    let cell = 0.15f32;
                    let cells_x = (PATCH / cell).floor() as usize; // 53
                    let cells_y = (2.0 * STRIP_HALF_W / cell).floor() as usize;
                    let mut cell_h = vec![0.0f32; cells_x * cells_y];
                    for h in cell_h.iter_mut() {
                        *h = quantize(rng.range(-g, g));
                    }
                    // Cells can be negative (U(−g,+g)); keep them — the slab
                    // bottom is below the deepest possible cell.
                    for i in i0..=i1.min(nx) {
                        let lx = (i as f32 * hs) - PATCH * p as f32;
                        let cx = ((lx / cell) as usize).min(cells_x - 1);
                        for j in 0..=ny {
                            let ly = j as f32 * hs;
                            let cy = ((ly / cell) as usize).min(cells_y - 1);
                            heights[j * (nx + 1) + i] = cell_h[cy * cells_x + cx];
                        }
                    }
                }
                TerrainFamily::Rough => {
                    // U[0.01·d, 0.2·d] (one-sided) on a 0.4 m lattice, smooth
                    // interpolation down to the mesh grid.
                    let (lo, hi) = (0.01 * d, 0.2 * d);
                    let lat = 0.4f32;
                    let ln_x = (PATCH / lat) as usize + 1; // 21 nodes
                    let ln_y = (2.0 * STRIP_HALF_W / lat) as usize + 1;
                    let mut lattice = vec![0.0f32; ln_x * ln_y];
                    // AGILE quantizes to noise_step = 0.1·d.
                    let step = (0.1 * d).max(VERTICAL_SCALE);
                    let n_steps = (((hi - lo) / step).floor() as i32).max(0);
                    for h in lattice.iter_mut() {
                        let k = (rng.range(0.0, 1.0) * (n_steps + 1) as f32) as i32;
                        *h = quantize(lo + step * k.min(n_steps) as f32);
                    }
                    let smooth = |t: f32| 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
                    for i in i0..=i1.min(nx) {
                        let lx = (i as f32 * hs) - PATCH * p as f32;
                        let fx = (lx / lat).min((ln_x - 2) as f32 + 0.999);
                        let (ix, tx) = (fx as usize, smooth(fx.fract()));
                        for j in 0..=ny {
                            let ly = j as f32 * hs;
                            let fy = (ly / lat).min((ln_y - 2) as f32 + 0.999);
                            let (iy, ty) = (fy as usize, smooth(fy.fract()));
                            let h00 = lattice[iy * ln_x + ix];
                            let h10 = lattice[iy * ln_x + ix + 1];
                            let h01 = lattice[(iy + 1) * ln_x + ix];
                            let h11 = lattice[(iy + 1) * ln_x + ix + 1];
                            let h = h00 * (1.0 - tx) * (1.0 - ty)
                                + h10 * tx * (1.0 - ty)
                                + h01 * (1.0 - tx) * ty
                                + h11 * tx * ty;
                            heights[j * (nx + 1) + i] = quantize(h);
                        }
                    }
                }
                TerrainFamily::Wave => {
                    // A/2·cos(2πy/λ) + A/2·sin(2πx/λ), A = 0.01 + 0.24·d.
                    let a = 0.01 + 0.24 * d;
                    let lambda = PATCH / 3.0;
                    let tau = std::f32::consts::TAU;
                    for i in i0..=i1.min(nx) {
                        let lx = (i as f32 * hs) - PATCH * p as f32;
                        for j in 0..=ny {
                            let ly = j as f32 * hs;
                            let h = 0.5 * a * (tau * ly / lambda).cos()
                                + 0.5 * a * (tau * lx / lambda).sin();
                            heights[j * (nx + 1) + i] = quantize(h);
                        }
                    }
                }
            }
        }

        TerrainStrip { family, hs, nx, ny, heights }
    }

    fn node(&self, i: usize, j: usize) -> f32 {
        self.heights[j * (self.nx + 1) + i]
    }

    /// Exact piecewise-linear surface height at world `(x, y)` — matches the
    /// emitted mesh's triangulation (each cell split along the (i,j)→(i+1,j+1)
    /// diagonal). Returns 0.0 (the flat backstop top) outside the strip.
    pub fn height(&self, x: f32, y: f32) -> f32 {
        let lx = x - STRIP_X0;
        let ly = y + STRIP_HALF_W;
        if lx < 0.0 || ly < 0.0 {
            return 0.0;
        }
        let (fx, fy) = (lx / self.hs, ly / self.hs);
        if fx >= self.nx as f32 || fy >= self.ny as f32 {
            return 0.0;
        }
        let (i, j) = (fx as usize, fy as usize);
        let (u, v) = (fx.fract(), fy.fract());
        let h00 = self.node(i, j);
        let h10 = self.node(i + 1, j);
        let h01 = self.node(i, j + 1);
        let h11 = self.node(i + 1, j + 1);
        // Triangles: (00, 10, 11) for u >= v, (00, 11, 01) otherwise.
        if u >= v {
            h00 + u * (h10 - h00) + v * (h11 - h10)
        } else {
            h00 + v * (h01 - h00) + u * (h11 - h01)
        }
    }

    /// Max surface height over the axis-aligned box `[x±r, y±r]` (spawn
    /// clearance): samples nodes covering the box plus the box corners.
    pub fn height_max_in(&self, x: f32, y: f32, r: f32) -> f32 {
        let mut m = f32::MIN;
        let steps = ((2.0 * r / self.hs).ceil() as usize + 1).max(2);
        for si in 0..=steps {
            let sx = x - r + 2.0 * r * si as f32 / steps as f32;
            for sj in 0..=steps {
                let sy = y - r + 2.0 * r * sj as f32 / steps as f32;
                m = m.max(self.height(sx, sy));
            }
        }
        m
    }

    /// Closed-slab triangle mesh (top surface + perimeter skirts + full-grid
    /// bottom at [`SLAB_BOTTOM`]) — watertight so parry's ORIENTED
    /// pseudo-normals exist (nexus requires them for trimesh contacts).
    pub fn mesh(&self) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let (nx, ny) = (self.nx, self.ny);
        let stride = nx + 1;
        let n_grid = stride * (ny + 1);
        let mut verts = Vec::with_capacity(2 * n_grid);
        // Top grid, then bottom grid (same layout, z = SLAB_BOTTOM).
        for j in 0..=ny {
            for i in 0..=nx {
                verts.push([
                    STRIP_X0 + i as f32 * self.hs,
                    -STRIP_HALF_W + j as f32 * self.hs,
                    self.node(i, j),
                ]);
            }
        }
        for j in 0..=ny {
            for i in 0..=nx {
                verts.push([
                    STRIP_X0 + i as f32 * self.hs,
                    -STRIP_HALF_W + j as f32 * self.hs,
                    SLAB_BOTTOM,
                ]);
            }
        }
        let top = |i: usize, j: usize| (j * stride + i) as u32;
        let bot = |i: usize, j: usize| (n_grid + j * stride + i) as u32;
        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(4 * nx * ny + 4 * (nx + ny));
        for j in 0..ny {
            for i in 0..nx {
                // Top: CCW from +Z. Bottom: CCW from −Z (reversed).
                tris.push([top(i, j), top(i + 1, j), top(i + 1, j + 1)]);
                tris.push([top(i, j), top(i + 1, j + 1), top(i, j + 1)]);
                tris.push([bot(i, j), bot(i + 1, j + 1), bot(i + 1, j)]);
                tris.push([bot(i, j), bot(i, j + 1), bot(i + 1, j + 1)]);
            }
        }
        // Skirts (outward-facing).
        for i in 0..nx {
            tris.push([top(i, 0), bot(i, 0), bot(i + 1, 0)]); // -Y side
            tris.push([top(i, 0), bot(i + 1, 0), top(i + 1, 0)]);
            tris.push([top(i, ny), bot(i + 1, ny), bot(i, ny)]); // +Y side
            tris.push([top(i, ny), top(i + 1, ny), bot(i + 1, ny)]);
        }
        for j in 0..ny {
            tris.push([top(0, j), bot(0, j + 1), bot(0, j)]); // -X side
            tris.push([top(0, j), top(0, j + 1), bot(0, j + 1)]);
            tris.push([top(nx, j), bot(nx, j), bot(nx, j + 1)]); // +X side
            tris.push([top(nx, j), bot(nx, j + 1), top(nx, j + 1)]);
        }
        (verts, tris)
    }

    /// Tiny closed slab far below the world — the geometry stand-in for the
    /// single-env spawn TEMPLATES (collider count/order parity with the main
    /// batch; never contacted, never copied by snapshot resets).
    pub fn flat_stub_mesh() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let z0 = -100.0;
        let z1 = -100.05;
        let v = vec![
            [0.0, 0.0, z0],
            [1.0, 0.0, z0],
            [1.0, 1.0, z0],
            [0.0, 1.0, z0],
            [0.0, 0.0, z1],
            [1.0, 0.0, z1],
            [1.0, 1.0, z1],
            [0.0, 1.0, z1],
        ];
        let t = vec![
            [0u32, 1, 2],
            [0, 2, 3], // top
            [4, 6, 5],
            [4, 7, 6], // bottom
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ];
        (v, t)
    }
}

/// AGILE's `terrain_levels_vel_curriculum` per-env state machine.
#[derive(Clone, Copy, Debug)]
pub struct TerrainCurriculum {
    pub level: u32,
    successes: u32,
    failures: u32,
}

/// Promote after this many episodes traveling > [`MOVE_UP_DISTANCE`].
pub const N_SUCCESSES: u32 = 4;
/// Demote after this many episodes traveling < [`MOVE_DOWN_DISTANCE`].
pub const N_FAILURES: u32 = 10;
pub const MOVE_UP_DISTANCE: f32 = 4.0;
pub const MOVE_DOWN_DISTANCE: f32 = 2.0;

impl TerrainCurriculum {
    /// Initial level ~ U{0, 1} (AGILE's `max_init_terrain_level = 1`).
    pub fn init(rng: &mut Lcg) -> Self {
        let level = if rng.range(0.0, 1.0) < 0.5 { 0 } else { 1 };
        TerrainCurriculum { level, successes: 0, failures: 0 }
    }

    /// Episode-end update (called on EVERY termination, incl. timeout —
    /// matching AGILE, which runs curriculum terms for all reset envs).
    pub fn on_episode_end(&mut self, traveled: f32, rng: &mut Lcg) {
        if traveled > MOVE_UP_DISTANCE {
            self.successes += 1;
        }
        if traveled < MOVE_DOWN_DISTANCE {
            self.failures += 1;
        }
        let up = self.successes >= N_SUCCESSES;
        let down = self.failures >= N_FAILURES;
        if !(up || down) {
            return;
        }
        self.successes = 0;
        self.failures = 0;
        if up {
            let next = self.level + 1;
            self.level = if next as usize >= ROWS {
                // Solved the hardest row: reassign to a uniform random row
                // (classic Isaac Lab / AGILE behavior).
                ((rng.range(0.0, 1.0) * ROWS as f32) as u32).min(ROWS as u32 - 1)
            } else {
                next
            };
        } else {
            self.level = self.level.saturating_sub(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(f: TerrainFamily) -> TerrainStrip {
        TerrainStrip::generate(f, 42)
    }

    /// Sample many points on a row's patch interior; return (min, max) height.
    fn row_extremes(s: &TerrainStrip, row: u32) -> (f32, f32) {
        let (cx, _) = TerrainStrip::patch_center(row);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        let mut rng = Lcg::new(7);
        for _ in 0..4000 {
            let x = cx + rng.range(-3.5, 3.5);
            let y = rng.range(-3.5, 3.5);
            let h = s.height(x, y);
            lo = lo.min(h);
            hi = hi.max(h);
        }
        (lo, hi)
    }

    #[test]
    fn boxes_amplitude_pins() {
        let s = strip(TerrainFamily::Boxes);
        // Row 0: d < 0.05 → g < 2 mm (quantized: 0 or ±5 mm edge cases).
        let (lo, hi) = row_extremes(&s, 0);
        assert!(hi <= 0.006 && lo >= -0.006, "row0 boxes {lo}..{hi}");
        // Row 19: d ∈ [0.95, 1) → g ≈ 0.038..0.04; heights within ±g.
        let (lo, hi) = row_extremes(&s, 19);
        assert!(hi <= 0.0405 && lo >= -0.0405, "row19 boxes {lo}..{hi}");
        assert!(hi > 0.02 && lo < -0.02, "row19 boxes should be rough: {lo}..{hi}");
    }

    #[test]
    fn rough_amplitude_pins() {
        let s = strip(TerrainFamily::Rough);
        // One-sided: heights ≥ 0 everywhere (interpolation preserves bounds).
        let (lo0, hi0) = row_extremes(&s, 0);
        assert!(lo0 >= -1e-6 && hi0 <= 0.2 * 0.05 + VERTICAL_SCALE, "row0 rough {lo0}..{hi0}");
        let (lo19, hi19) = row_extremes(&s, 19);
        assert!(lo19 >= -1e-6, "rough is one-sided: {lo19}");
        assert!(hi19 <= 0.2 + VERTICAL_SCALE, "row19 rough max {hi19}");
        assert!(hi19 > 0.09, "row19 rough should reach near 0.2·d: {hi19}");
    }

    #[test]
    fn wave_amplitude_pins() {
        let s = strip(TerrainFamily::Wave);
        // Row 19: A ≈ 0.238..0.25 → extremes near ±A.
        let (lo, hi) = row_extremes(&s, 19);
        assert!(hi <= 0.25 + VERTICAL_SCALE && lo >= -(0.25 + VERTICAL_SCALE), "row19 wave {lo}..{hi}");
        assert!(hi > 0.17 && lo < -0.17, "row19 wave peaks: {lo}..{hi}");
        // Row 0: A ≈ 0.01..0.022.
        let (lo, hi) = row_extremes(&s, 0);
        assert!(hi <= 0.025 && lo >= -0.025, "row0 wave {lo}..{hi}");
    }

    #[test]
    fn quantization_and_determinism() {
        let s1 = strip(TerrainFamily::Rough);
        let s2 = strip(TerrainFamily::Rough);
        assert_eq!(s1.heights, s2.heights, "same seed → same strip");
        let s3 = TerrainStrip::generate(TerrainFamily::Rough, 43);
        assert_ne!(s1.heights, s3.heights, "different seed → different strip");
        // Boxes/wave node heights are quantized to 5 mm.
        let sb = strip(TerrainFamily::Boxes);
        for &h in &sb.heights {
            let q = (h / VERTICAL_SCALE).round() * VERTICAL_SCALE;
            assert!((h - q).abs() < 1e-6, "unquantized node {h}");
        }
    }

    #[test]
    fn sampler_matches_mesh() {
        for fam in [TerrainFamily::Boxes, TerrainFamily::Rough, TerrainFamily::Wave] {
            let s = strip(fam);
            let (verts, tris) = s.mesh();
            let mut rng = Lcg::new(3);
            // Barycentric evaluation of random points inside random TOP triangles.
            let n_top = {
                // Top tris are the first 2·nx·ny? They are interleaved with
                // bottom in emission order (t,t,b,b per cell) — filter by z.
                tris.len()
            };
            let mut checked = 0;
            for _ in 0..20000 {
                let t = tris[(rng.range(0.0, 1.0) * n_top as f32) as usize % tris.len()];
                let (a, b, c) = (verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize]);
                // Skip non-top faces (skirts/bottom contain z == SLAB_BOTTOM).
                if a[2] <= SLAB_BOTTOM + 1e-6 || b[2] <= SLAB_BOTTOM + 1e-6 || c[2] <= SLAB_BOTTOM + 1e-6 {
                    continue;
                }
                let (mut u, mut v) = (rng.range(0.0, 1.0), rng.range(0.0, 1.0));
                if u + v > 1.0 {
                    u = 1.0 - u;
                    v = 1.0 - v;
                }
                // Shrink toward the centroid to dodge edge ties between triangles.
                let (cu, cv) = (1.0 / 3.0, 1.0 / 3.0);
                let (u, v) = (cu + 0.9 * (u - cu), cv + 0.9 * (v - cv));
                let w = 1.0 - u - v;
                let x = w * a[0] + u * b[0] + v * c[0];
                let y = w * a[1] + u * b[1] + v * c[1];
                let z = w * a[2] + u * b[2] + v * c[2];
                let h = s.height(x, y);
                assert!(
                    (h - z).abs() < 1e-4,
                    "{fam:?}: sampler {h} != mesh {z} at ({x},{y})"
                );
                checked += 1;
            }
            assert!(checked > 1000, "{fam:?}: too few top-face samples ({checked})");
        }
    }

    #[test]
    fn curriculum_branches() {
        let mut rng = Lcg::new(1);
        let mut c = TerrainCurriculum { level: 5, successes: 0, failures: 0 };
        // 2..4 m band: no-op forever.
        for _ in 0..100 {
            c.on_episode_end(3.0, &mut rng);
        }
        assert_eq!((c.level, c.successes, c.failures), (5, 0, 0));
        // 3 successes: no move yet; 4th promotes and zeroes counters.
        for _ in 0..3 {
            c.on_episode_end(5.0, &mut rng);
        }
        assert_eq!(c.level, 5);
        c.on_episode_end(5.0, &mut rng);
        assert_eq!((c.level, c.successes, c.failures), (6, 0, 0));
        // Mixed history: 9 failures + 3 successes → nothing; 4th success
        // promotes (cumulative counters, whichever threshold first).
        for _ in 0..9 {
            c.on_episode_end(0.5, &mut rng);
        }
        for _ in 0..3 {
            c.on_episode_end(5.0, &mut rng);
        }
        assert_eq!(c.level, 6);
        c.on_episode_end(5.0, &mut rng);
        assert_eq!((c.level, c.successes, c.failures), (7, 0, 0));
        // 10 failures demote and zero.
        for _ in 0..10 {
            c.on_episode_end(0.0, &mut rng);
        }
        assert_eq!((c.level, c.successes, c.failures), (6, 0, 0));
        // Demotion clamps at 0.
        let mut c0 = TerrainCurriculum { level: 0, successes: 0, failures: 0 };
        for _ in 0..10 {
            c0.on_episode_end(0.0, &mut rng);
        }
        assert_eq!(c0.level, 0);
        // Promotion past the top row → uniform random row in [0, ROWS).
        let mut seen_non_top = false;
        for seed in 0..50 {
            let mut rt = Lcg::new(seed);
            let mut ct = TerrainCurriculum { level: ROWS as u32 - 1, successes: 3, failures: 0 };
            ct.on_episode_end(5.0, &mut rt);
            assert!((ct.level as usize) < ROWS);
            if (ct.level as usize) < ROWS - 1 {
                seen_non_top = true;
            }
        }
        assert!(seen_non_top, "top-row wrap should randomize");
        // Init level ∈ {0, 1}.
        for seed in 0..50 {
            let mut r = Lcg::new(seed);
            assert!(TerrainCurriculum::init(&mut r).level <= 1);
        }
    }
}

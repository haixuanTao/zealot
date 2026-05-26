//! Hand-rolled MLP backward pass on vortx, verified against nalgebra analytic gradients.
//!
//! Run: `cargo run -p zealot-rl --example mlp_backward --features gpu`
//!
//! Net: 6 -> tanh(8) -> 2 (linear). Loss = ½‖out - target‖². We compute all four
//! gradients (dW1, db1, dW2, db2) on the GPU using:
//!   - GEMM with transposed views for `dz·aᵀ` and `Wᵀ·dz`,
//!   - the custom `tanh_backward` kernel for `dz1 = da1 ⊙ (1 - a1²)`,
//! and check them against a CPU reference. This validates the full training-gradient
//! path on vortx — the foundation for PPO.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use nalgebra::DMatrix;
use vortx::linalg::{Activation, Gemm, OpAssign, OpAssignVariant};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use wgpu::{Features, Limits};

const IN: usize = 6;
const HID: usize = 8;
const OUT: usize = 2;

fn max_err(gpu: &[f32], cpu: &DMatrix<f32>) -> f32 {
    let (nr, nc) = (cpu.nrows(), cpu.ncols());
    let mut e = 0f32;
    for r in 0..nr {
        for c in 0..nc {
            e = e.max((gpu[r * nc + c] - cpu[(r, c)]).abs());
        }
    }
    e
}

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let backend = GpuBackend::WebGpu(WebGpu::new(Features::default(), Limits::default()).await?);
    let gemm = Gemm::from_backend(&backend)?;
    let op = OpAssign::from_backend(&backend)?;
    let act = Activation::from_backend(&backend)?;
    let mut sh = TensorLayoutBuffers::new(&backend);

    let w1 = DMatrix::<f32>::from_fn(HID, IN, |r, c| ((r * IN + c) % 7) as f32 * 0.1 - 0.3);
    let b1 = DMatrix::<f32>::from_fn(HID, 1, |r, _| (r % 3) as f32 * 0.1 - 0.1);
    let w2 = DMatrix::<f32>::from_fn(OUT, HID, |r, c| ((r * HID + c) % 5) as f32 * 0.1 - 0.2);
    let b2 = DMatrix::<f32>::from_fn(OUT, 1, |r, _| r as f32 * 0.05);
    let x = DMatrix::<f32>::from_fn(IN, 1, |r, _| (r % 4) as f32 * 0.25 - 0.5);
    let target = DMatrix::<f32>::from_fn(OUT, 1, |r, _| 0.5 - r as f32 * 0.3);

    // ---- CPU reference (forward + analytic backward) ----
    let a1c = (&w1 * &x + &b1).map(|v| v.tanh());
    let z2c = &w2 * &a1c + &b2;
    let dz2 = &z2c - &target; // dL/dz2 for ½‖·‖²
    let dw2_c = &dz2 * a1c.transpose();
    let db2_c = dz2.clone();
    let da1_c = w2.transpose() * &dz2;
    let dz1_c = da1_c.zip_map(&a1c, |g, a| g * (1.0 - a * a));
    let dw1_c = &dz1_c * x.transpose();
    let db1_c = dz1_c.clone();

    // ---- GPU ----
    let st = BufferUsages::STORAGE;
    let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let gw1 = Tensor::matrix_from_na(&backend, &w1, st)?;
    let gb1 = Tensor::matrix_from_na(&backend, &b1, st)?;
    let gw2 = Tensor::matrix_from_na(&backend, &w2, st)?;
    let gb2 = Tensor::matrix_from_na(&backend, &b2, st)?;
    let gx = Tensor::matrix_from_na(&backend, &x, st)?;
    let gt = Tensor::matrix_from_na(&backend, &target, st)?;
    let mut z1 = Tensor::matrix_from_na(&backend, &DMatrix::zeros(HID, 1), st)?; // -> a1
    let mut z2 = Tensor::matrix_from_na(&backend, &DMatrix::zeros(OUT, 1), rw)?; // -> out -> dz2 (=db2)
    let mut gdw2 = Tensor::matrix_from_na(&backend, &DMatrix::zeros(OUT, HID), rw)?;
    let mut gda1 = Tensor::matrix_from_na(&backend, &DMatrix::zeros(HID, 1), rw)?; // -> da1 -> dz1 (=db1)
    let mut gdw1 = Tensor::matrix_from_na(&backend, &DMatrix::zeros(HID, IN), rw)?;

    let mut enc = backend.begin_encoding();
    // forward
    { let mut p = enc.begin_pass("gemm1", None); gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut z1, &gw1, &gx)?; }
    { let mut p = enc.begin_pass("bias1", None); op.launch(&backend, &mut sh, &mut p, OpAssignVariant::Add, &mut z1, &gb1)?; }
    { let mut p = enc.begin_pass("tanh1", None); act.tanh(&backend, &mut sh, &mut p, &mut z1)?; }
    { let mut p = enc.begin_pass("gemm2", None); gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut z2, &gw2, &z1)?; }
    { let mut p = enc.begin_pass("bias2", None); op.launch(&backend, &mut sh, &mut p, OpAssignVariant::Add, &mut z2, &gb2)?; }
    // backward
    { let mut p = enc.begin_pass("dz2", None); op.launch(&backend, &mut sh, &mut p, OpAssignVariant::Sub, &mut z2, &gt)?; } // z2 = dz2
    { let mut p = enc.begin_pass("dW2", None); gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut gdw2, &z2, z1.transpose_last_dims())?; } // dz2·a1ᵀ
    { let mut p = enc.begin_pass("da1", None); gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut gda1, gw2.transpose_last_dims(), &z2)?; } // W2ᵀ·dz2
    { let mut p = enc.begin_pass("dz1", None); act.tanh_backward(&backend, &mut sh, &mut p, &mut gda1, &z1)?; } // *= 1 - a1²
    { let mut p = enc.begin_pass("dW1", None); gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut gdw1, &gda1, gx.transpose_last_dims())?; } // dz1·xᵀ
    backend.submit(enc)?;
    backend.synchronize()?;

    let dw1_g = backend.slow_read_vec(gdw1.buffer()).await?;
    let db1_g = backend.slow_read_vec(gda1.buffer()).await?;
    let dw2_g = backend.slow_read_vec(gdw2.buffer()).await?;
    let db2_g = backend.slow_read_vec(z2.buffer()).await?;

    let e_dw1 = max_err(&dw1_g, &dw1_c);
    let e_db1 = max_err(&db1_g, &db1_c);
    let e_dw2 = max_err(&dw2_g, &dw2_c);
    let e_db2 = max_err(&db2_g, &db2_c);
    println!("MLP backward gradient errors (gpu vs cpu analytic):");
    println!("  dW1 {e_dw1:.3e}   db1 {e_db1:.3e}   dW2 {e_dw2:.3e}   db2 {e_db2:.3e}");
    let worst = e_dw1.max(e_db1).max(e_dw2).max(e_db2);
    anyhow::ensure!(worst < 1e-4, "GPU gradients diverged from CPU (worst {worst:.3e})");
    println!("OK — hand-rolled GPU backward matches CPU analytic gradients. Autodiff path verified.");
    Ok(())
}

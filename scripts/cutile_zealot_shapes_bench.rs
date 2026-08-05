/*
 * Zealot PPO-update GEMM shapes benchmark (local scratch, not upstream).
 *
 * Times every distinct GEMM in one biped_train_gpu minibatch step (mb = 12288
 * columns): actor [45,256,256,128,12] and critic [51,512,256,128,1] forward,
 * dgrad and wgrad, in f32 and tf32 (tensor cores). The wgrad shapes
 * (K = 12288, tiny M*N) additionally run a split-K variant: the plain tile
 * GEMM gives them only a handful of CTAs, leaving the GPU idle.
 *
 * Dims are padded up to tile multiples (45→48, 51→64, 12→16, 1→16); padded
 * FLOPs are reported (slightly overstates work = conservative projection).
 */
use cuda_async::device_operation::{value, DeviceOp};
use cuda_core::Device;
use cutile::api;
use cutile::prelude::IntoPartition;
use cutile::tile_kernel::{PartitionOp, TileKernel};
use std::sync::Arc;
use std::time::Instant;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    // Plain tiled GEMM, f32 FMA path, no optimization hints (the CGA hint
    // emits Tile IR the released 13.3 tileiras cannot parse yet).
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemm_f32<const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<f32, { [BM, BN] }> = z.load();
        for i in 0i32..(k / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }

    // Same, but tiles are converted f32 -> tf32 before the mma (tensor cores;
    // f32 accumulator). f32 storage, like zealot's buffers.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemm_tf32<const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<f32, { [BM, BN] }> = z.load();
        for i in 0i32..(k / BK) {
            let tile_x: Tile<f32, { [BM, BK] }> = part_x.load([pid.0, i]);
            let tile_y: Tile<f32, { [BK, BN] }> = part_y.load([i, pid.1]);
            let tx: Tile<tf32, { [BM, BK] }> = convert_tile(tile_x);
            let ty: Tile<tf32, { [BK, BN] }> = convert_tile(tile_y);
            tile_z = mma(tx, ty, tile_z);
        }
        z.store(tile_z);
    }

    // Split-K GEMM (f32): partial products land in z_parts = [S*M, N]; the
    // grid row-block axis covers S x (M/BM).
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemm_splitk_f32<const BM: i32, const BN: i32, const BK: i32>(
        z_parts: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
        blocks_m: i32,
        ktiles_per_chunk: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let s = pid.0 / blocks_m;
        let mb = pid.0 % blocks_m;
        let mut acc: Tile<f32, { [BM, BN] }> = z_parts.load();
        for i in 0i32..ktiles_per_chunk {
            let kt = s * ktiles_per_chunk + i;
            let tile_x = part_x.load([mb, kt]);
            let tile_y = part_y.load([kt, pid.1]);
            acc = mma(tile_x, tile_y, acc);
        }
        z_parts.store(acc);
    }

    // Split-K GEMM, tf32 tensor-core variant.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemm_splitk_tf32<const BM: i32, const BN: i32, const BK: i32>(
        z_parts: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
        blocks_m: i32,
        ktiles_per_chunk: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let s = pid.0 / blocks_m;
        let mb = pid.0 % blocks_m;
        let mut acc: Tile<f32, { [BM, BN] }> = z_parts.load();
        for i in 0i32..ktiles_per_chunk {
            let kt = s * ktiles_per_chunk + i;
            let tile_x: Tile<f32, { [BM, BK] }> = part_x.load([mb, kt]);
            let tile_y: Tile<f32, { [BK, BN] }> = part_y.load([kt, pid.1]);
            let tx: Tile<tf32, { [BM, BK] }> = convert_tile(tile_x);
            let ty: Tile<tf32, { [BK, BN] }> = convert_tile(tile_y);
            acc = mma(tx, ty, acc);
        }
        z_parts.store(acc);
    }

    // Reduce the S partial products: out[mb, nb] = sum_s parts[s*blocks_m + mb, nb].
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn reduce_splitk<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [BM, BN] }>,
        parts: &Tensor<f32, { [-1, -1] }>,
        blocks_m: i32,
        s_count: i32,
    ) {
        let part = parts.partition(const_shape![BM, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BN] }> = out.load();
        for s in 0i32..s_count {
            let t: Tile<f32, { [BM, BN] }> = part.load([s * blocks_m + pid.0, pid.1]);
            acc = acc + t;
        }
        out.store(acc);
    }
}
use kernels::*;

/// Largest tile size from `cands` that divides `dim`.
fn pick(dim: usize, cands: &[usize]) -> usize {
    *cands.iter().find(|&&c| dim % c == 0).unwrap_or(&16)
}

// (label, M, N, K, occurrences per minibatch step, is_wgrad)
const SHAPES: &[(&str, usize, usize, usize, usize, bool)] = &[
    ("actor fwd l0", 256, 12288, 48, 1, false),
    ("fwd/dgrad 256x256", 256, 12288, 256, 2, false),
    ("fwd 128<-256", 128, 12288, 256, 2, false),
    ("fwd out 16<-128", 16, 12288, 128, 2, false),
    ("critic fwd l0", 512, 12288, 64, 1, false),
    ("critic fwd l1", 256, 12288, 512, 1, false),
    ("dgrad 128<-16", 128, 12288, 16, 2, false),
    ("dgrad 256<-128", 256, 12288, 128, 2, false),
    ("critic dgrad l1", 512, 12288, 256, 1, false),
    ("wgrad 16x128", 16, 128, 12288, 2, true),
    ("wgrad 128x256", 128, 256, 12288, 2, true),
    ("wgrad 256x256", 256, 256, 12288, 1, true),
    ("wgrad 256x48", 256, 48, 12288, 1, true),
    ("wgrad 256x512", 256, 512, 12288, 1, true),
    ("wgrad 512x64", 512, 64, 12288, 1, true),
];

macro_rules! bench_dtype {
    ($fname:ident, $tname:literal, $gemm:ident, $gemm_splitk:ident) => {
        fn $fname(stream: &std::sync::Arc<cuda_core::Stream>) -> f64 {
            let iters = 50usize;
            let mut step_total_s = 0.0f64;
            println!(
                "\n=== {} ===\n{:<18} {:>5}x{:>5}x{:>5}  tiles        variant   ms/call  TFLOPS  x",
                $tname, "shape", "M", "N", "K"
            );
            for &(label, m, n, k, mult, is_wgrad) in SHAPES {
                let bm = pick(m, &[128, 64, 32, 16]);
                let bn = pick(n, &[128, 64, 32, 16]);
                let bk = pick(k, &[64, 32, 16]);
                let g = vec![bm.to_string(), bn.to_string(), bk.to_string()];
                let x = api::zeros::<f32>(&[m, k])
                    .then(|t| value(Arc::new(t)))
                    .sync_on(stream)
                    .expect("x");
                let y = api::zeros::<f32>(&[k, n])
                    .then(|t| value(Arc::new(t)))
                    .sync_on(stream)
                    .expect("y");
                // Plain kernel timing.
                let mut z = api::zeros::<f32>(&[m, n])
                    .partition([bm, bn])
                    .sync_on(stream)
                    .expect("z");
                for _ in 0..3 {
                    let (zz, _, _, _) = unsafe {
                        $gemm(z, x.clone(), y.clone(), k as i32)
                            .generics(g.clone())
                            .async_on(stream)
                            .expect("launch")
                    };
                    z = zz;
                }
                unsafe { stream.synchronize() }.expect("sync");
                let start = Instant::now();
                for _ in 0..iters {
                    let (zz, _, _, _) = unsafe {
                        $gemm(z, x.clone(), y.clone(), k as i32)
                            .generics(g.clone())
                            .async_on(stream)
                            .expect("launch")
                    };
                    z = zz;
                }
                unsafe { stream.synchronize() }.expect("sync");
                let plain_s = start.elapsed().as_secs_f64() / iters as f64;

                // Split-K for the wgrad shapes (K = 12288 with tiny M*N).
                let mut best_s = plain_s;
                let mut variant = "plain ";
                if is_wgrad {
                    let bm = pick(m, &[64, 32, 16]);
                    let bn = pick(n, &[64, 32, 16]);
                    let s_count = 32usize;
                    let ktiles = k / bk;
                    if ktiles % s_count == 0 {
                        let gsk = vec![bm.to_string(), bn.to_string(), bk.to_string()];
                        let grd = vec![bm.to_string(), bn.to_string()];
                        let blocks_m = (m / bm) as i32;
                        let kpc = (ktiles / s_count) as i32;
                        let mut parts = api::zeros::<f32>(&[s_count * m, n])
                            .partition([bm, bn])
                            .sync_on(stream)
                            .expect("parts");
                        let mut out = api::zeros::<f32>(&[m, n])
                            .partition([bm, bn])
                            .sync_on(stream)
                            .expect("out");
                        for _ in 0..3 {
                            let (p2, _, _, _, _) = unsafe {
                                $gemm_splitk(parts, x.clone(), y.clone(), blocks_m, kpc)
                                    .generics(gsk.clone())
                                    .async_on(stream)
                                    .expect("splitk")
                            };
                            let p2t = Arc::new(p2.unpartition());
                            let (o2, p2t, _, _) = unsafe {
                                reduce_splitk(out, p2t, blocks_m, s_count as i32)
                                    .generics(grd.clone())
                                    .async_on(stream)
                                    .expect("reduce")
                            };
                            out = o2;
                            parts = Arc::try_unwrap(p2t)
                                .map_err(|_| "parts still shared")
                                .unwrap()
                                .partition([bm, bn]);
                        }
                        unsafe { stream.synchronize() }.expect("sync");
                        let start = Instant::now();
                        for _ in 0..iters {
                            let (p2, _, _, _, _) = unsafe {
                                $gemm_splitk(parts, x.clone(), y.clone(), blocks_m, kpc)
                                    .generics(gsk.clone())
                                    .async_on(stream)
                                    .expect("splitk")
                            };
                            let p2t = Arc::new(p2.unpartition());
                            let (o2, p2t, _, _) = unsafe {
                                reduce_splitk(out, p2t, blocks_m, s_count as i32)
                                    .generics(grd.clone())
                                    .async_on(stream)
                                    .expect("reduce")
                            };
                            out = o2;
                            parts = Arc::try_unwrap(p2t)
                                .map_err(|_| "parts still shared")
                                .unwrap()
                                .partition([bm, bn]);
                        }
                        unsafe { stream.synchronize() }.expect("sync");
                        let sk_s = start.elapsed().as_secs_f64() / iters as f64;
                        if sk_s < best_s {
                            best_s = sk_s;
                            variant = "splitK";
                        }
                    }
                }
                let tflops = 2.0 * (m * n * k) as f64 / best_s / 1e12;
                step_total_s += best_s * mult as f64;
                println!(
                    "{label:<18} {m:>5}x{n:>5}x{k:>5}  ({bm:>3},{bn:>3},{bk:>2})  {variant}  {:>8.3}  {:>6.2}  {mult}",
                    best_s * 1e3,
                    tflops
                );
            }
            println!(
                "GEMMs per minibatch step: {:.3} ms -> projected per PPO update (x20): {:.3} s",
                step_total_s * 1e3,
                step_total_s * 20.0
            );
            step_total_s
        }
    };
}

bench_dtype!(bench_f32, "f32", gemm_f32, gemm_splitk_f32);
bench_dtype!(bench_tf32, "tf32 (f32 storage)", gemm_tf32, gemm_splitk_tf32);

fn main() {
    let device = Device::new(0).expect("gpu");
    let stream = device.new_stream().expect("stream");
    bench_f32(&stream);
    bench_tf32(&stream);
}

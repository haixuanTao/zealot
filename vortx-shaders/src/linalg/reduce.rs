//! Reduction operations for tensors (sum, product, min, max, squared norm).

use super::shape::Shape;
#[cfg(feature = "push_constants")]
use super::shape::Shapes1;
use core::ops::{Add, Mul};
use glamx::UVec3;
use khal_std::{
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};

#[cfg(feature = "subgroup_ops")]
const WORKGROUP_SIZE: usize = 32;
#[cfg(not(feature = "subgroup_ops"))]
const WORKGROUP_SIZE: usize = 128;

macro_rules! decl_reductions {
    ($T: ty, $zero: expr, $one: expr, $min: expr, $max: expr;
     $reduce_add: ident, $reduce_mul: ident, $reduce_min: ident, $reduce_max: ident) => {
        /// Sum reduction.
        #[spirv_bindgen]
        #[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
        #[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(128, 1, 1))))]
        pub fn $reduce_add(
            #[spirv(local_invocation_id)] local_id: UVec3,
            #[spirv(workgroup)] workspace: &mut [$T; WORKGROUP_SIZE],
            #[cfg(feature = "push_constants")]
            #[spirv(push_constant)]
            shapes: &Shapes1,
            #[cfg(not(feature = "push_constants"))]
            #[spirv(uniform, descriptor_set = 0, binding = 0)]
            shape: &Shape,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input: &[$T],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$T],
        ) {
            #[cfg(feature = "push_constants")]
            let shape = &shapes.shape;

            let thread_id = local_id.x as usize;
            workspace.write(thread_id, $zero);

            for i in (thread_id as u32..shape.w).step_by(WORKGROUP_SIZE) {
                // TODO: support tensors that are not just vectors.
                //       We'd reduce along the last dimension only, and use the
                //       workgroup_id to compute on each axis in parallel.
                let val_i = input.read(shape.it(0, 0, 0, i) as usize);
                workspace.write(thread_id, workspace.read(thread_id) + val_i);
            }

            khal_std::sync::workgroup_memory_barrier_with_group_sync();

            #[cfg(feature = "subgroup_ops")]
            let sum = spirv_std::arch::subgroup_f_add(workspace.read(thread_id));

            #[cfg(not(feature = "subgroup_ops"))]
            {
                // Rolled + opaque bound: see `opaque_bound`.
                let mut stride = 64usize;
                for _ in 0..opaque_bound(7) {
                    reduce_workspace_sum(thread_id, stride, workspace);
                    stride /= 2;
                }
            }

            if local_id.x == 0 {
                #[cfg(feature = "subgroup_ops")]
                {
                    output.write(0, sum);
                }
                #[cfg(not(feature = "subgroup_ops"))]
                {
                    output.write(0, workspace.read(0));
                }
            }
        }

        /// Product reduction.
        #[spirv_bindgen]
        #[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
        #[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(128, 1, 1))))]
        pub fn $reduce_mul(
            #[spirv(local_invocation_id)] local_id: UVec3,
            #[spirv(workgroup)] workspace: &mut [$T; WORKGROUP_SIZE],
            #[cfg(feature = "push_constants")]
            #[spirv(push_constant)]
            shapes: &Shapes1,
            #[cfg(not(feature = "push_constants"))]
            #[spirv(uniform, descriptor_set = 0, binding = 0)]
            shape: &Shape,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input: &[$T],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$T],
        ) {
            #[cfg(feature = "push_constants")]
            let shape = &shapes.shape;

            let thread_id = local_id.x as usize;
            workspace.write(thread_id, $one);

            for i in (thread_id as u32..shape.w).step_by(WORKGROUP_SIZE) {
                // TODO: support tensors that are not just vectors.
                //       We'd reduce along the last dimension only, and use the
                //       workgroup_id to compute on each axis in parallel.
                let val_i = input.read(shape.it(0, 0, 0, i) as usize);
                workspace.write(thread_id, workspace.read(thread_id) * val_i);
            }

            khal_std::sync::workgroup_memory_barrier_with_group_sync();

            #[cfg(feature = "subgroup_ops")]
            let prod = spirv_std::arch::subgroup_f_mul(workspace.read(thread_id));

            #[cfg(not(feature = "subgroup_ops"))]
            {
                // Rolled + opaque bound: see `opaque_bound`.
                let mut stride = 64usize;
                for _ in 0..opaque_bound(7) {
                    reduce_workspace_mul(thread_id, stride, workspace);
                    stride /= 2;
                }
            }

            if local_id.x == 0 {
                #[cfg(feature = "subgroup_ops")]
                {
                    output.write(0, prod);
                }
                #[cfg(not(feature = "subgroup_ops"))]
                {
                    output.write(0, workspace.read(0));
                }
            }
        }

        /// Minimum reduction.
        #[spirv_bindgen]
        #[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
        #[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(128, 1, 1))))]
        pub fn $reduce_min(
            #[spirv(global_invocation_id)] _global_id: UVec3,
            #[spirv(local_invocation_id)] local_id: UVec3,
            #[spirv(workgroup)] workspace: &mut [$T; WORKGROUP_SIZE],
            #[cfg(feature = "push_constants")]
            #[spirv(push_constant)]
            shapes: &Shapes1,
            #[cfg(not(feature = "push_constants"))]
            #[spirv(uniform, descriptor_set = 0, binding = 0)]
            shape: &Shape,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input: &[$T],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$T],
        ) {
            #[cfg(feature = "push_constants")]
            let shape = &shapes.shape;

            let thread_id = local_id.x as usize;
            workspace.write(thread_id, $max);

            for i in (thread_id as u32..shape.w).step_by(WORKGROUP_SIZE) {
                // TODO: support tensors that are not just vectors.
                //       We'd reduce along the last dimension only, and use the
                //       workgroup_id to compute on each axis in parallel.
                let val_i = input.read(shape.it(0, 0, 0, i) as usize);
                workspace.write(thread_id, workspace.read(thread_id).min(val_i));
            }

            khal_std::sync::workgroup_memory_barrier_with_group_sync();

            #[cfg(feature = "subgroup_ops")]
            let min = spirv_std::arch::subgroup_f_min(workspace.read(thread_id));

            #[cfg(not(feature = "subgroup_ops"))]
            {
                #[inline(always)]
                fn reduce_workspace_min(
                    thread_id: usize,
                    stride: usize,
                    workspace: &mut impl MaybeIndexUnchecked<$T>,
                ) {
                    if thread_id < stride {
                        workspace.write(
                            thread_id,
                            workspace
                                .read(thread_id)
                                .min(workspace.read(thread_id + stride)),
                        );
                    }
                    khal_std::sync::workgroup_memory_barrier_with_group_sync();
                }

                // Rolled + opaque bound: see `opaque_bound`.
                let mut stride = 64usize;
                for _ in 0..opaque_bound(7) {
                    reduce_workspace_min(thread_id, stride, workspace);
                    stride /= 2;
                }
            }

            if local_id.x == 0 {
                #[cfg(feature = "subgroup_ops")]
                {
                    output.write(0, min);
                }
                #[cfg(not(feature = "subgroup_ops"))]
                {
                    output.write(0, workspace.read(0));
                }
            }
        }

        /// Maximum reduction.
        #[spirv_bindgen]
        #[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
        #[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(128, 1, 1))))]
        pub fn $reduce_max(
            #[spirv(local_invocation_id)] local_id: UVec3,
            #[spirv(workgroup)] workspace: &mut [$T; WORKGROUP_SIZE],
            #[cfg(feature = "push_constants")]
            #[spirv(push_constant)]
            shapes: &Shapes1,
            #[cfg(not(feature = "push_constants"))]
            #[spirv(uniform, descriptor_set = 0, binding = 0)]
            shape: &Shape,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input: &[$T],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$T],
        ) {
            #[cfg(feature = "push_constants")]
            let shape = &shapes.shape;

            let thread_id = local_id.x as usize;
            workspace.write(thread_id, $min);

            for i in (thread_id as u32..shape.w).step_by(WORKGROUP_SIZE) {
                // TODO: support tensors that are not just vectors.
                //       We'd reduce along the last dimension only, and use the
                //       workgroup_id to compute on each axis in parallel.
                //       The output would have shape [n, c, h, 1].
                let val_i = input.read(shape.it(0, 0, 0, i) as usize);
                workspace.write(thread_id, workspace.read(thread_id).max(val_i));
            }

            khal_std::sync::workgroup_memory_barrier_with_group_sync();

            #[cfg(feature = "subgroup_ops")]
            let max = spirv_std::arch::subgroup_f_max(workspace.read(thread_id));

            #[cfg(not(feature = "subgroup_ops"))]
            {
                #[inline(always)]
                fn reduce_workspace_max(
                    thread_id: usize,
                    stride: usize,
                    workspace: &mut impl MaybeIndexUnchecked<$T>,
                ) {
                    if thread_id < stride {
                        workspace.write(
                            thread_id,
                            workspace
                                .read(thread_id)
                                .max(workspace.read(thread_id + stride)),
                        );
                    }
                    khal_std::sync::workgroup_memory_barrier_with_group_sync();
                }

                // Rolled + opaque bound: see `opaque_bound`.
                let mut stride = 64usize;
                for _ in 0..opaque_bound(7) {
                    reduce_workspace_max(thread_id, stride, workspace);
                    stride /= 2;
                }
            }

            if local_id.x == 0 {
                #[cfg(feature = "subgroup_ops")]
                {
                    output.write(0, max);
                }
                #[cfg(not(feature = "subgroup_ops"))]
                {
                    output.write(0, workspace.read(0));
                }
            }
        }
    };
}

// NOTE: not the actual max, but some platforms (like safari) don’t accept larger values.
const MAX_F32: f32 = 1.0e18;
// NOTE: not the actual max, but some platforms (like safari) don’t accept larger values.
const MIN_F32: f32 = -1.0e18;

decl_reductions!(
    f32, 0.0, 1.0, MIN_F32, MAX_F32;
    reduce_add_f32, reduce_mul_f32, reduce_min_f32, reduce_max_f32
);
decl_reductions!(
    u32, 0, 1, u32::MIN, u32::MAX;
    reduce_add_u32, reduce_mul_u32, reduce_min_u32, reduce_max_u32
);
decl_reductions!(
    i32, 0, 1, i32::MIN, i32::MAX;
    reduce_add_i32, reduce_mul_i32, reduce_min_i32, reduce_max_i32
);

/// Squared norm reduction.
#[spirv_bindgen]
#[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
#[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(128, 1, 1))))]
pub fn reduce_sq_norm(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup)] workspace: &mut [f32; WORKGROUP_SIZE],
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 0)]
    shape: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape = &shapes.shape;

    let thread_id = local_id.x as usize;
    workspace.write(thread_id, 0.0);

    for i in (thread_id as u32..shape.w).step_by(WORKGROUP_SIZE) {
        // TODO: support tensors that are not just vectors.
        //       We'd reduce along the last dimension only, and use the
        //       workgroup_id to compute on each axis in parallel.
        //       The output would have shape [n, c, h, 1].
        let val_i = input.read(shape.it(0, 0, 0, i) as usize);
        workspace.write(thread_id, workspace.read(thread_id) + val_i * val_i);
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    #[cfg(feature = "subgroup_ops")]
    let sum = spirv_std::arch::subgroup_f_add(workspace.read(thread_id));

    #[cfg(not(feature = "subgroup_ops"))]
    {
        // Rolled + opaque bound: see `opaque_bound`.
        let mut stride = 64usize;
        for _ in 0..opaque_bound(7) {
            reduce_workspace_sum(thread_id, stride, workspace);
            stride /= 2;
        }
    }

    if local_id.x == 0 {
        #[cfg(feature = "subgroup_ops")]
        {
            output.write(0, sum);
        }
        #[cfg(not(feature = "subgroup_ops"))]
        {
            output.write(0, workspace.read(0));
        }
    }
}


/// An optimization-opaque loop bound for the barriered reduction chains.
/// rustc_codegen_nvvm fully unrolls/inlines constant sequences and its
/// structurizer then sinks `bar.sync` into the divergent `thread_id < stride`
/// guards — mismatched barrier counts deadlock the block on sm_90+. Routing
/// the bound through `black_box` keeps the loop rolled (the barrier stays at
/// the uniform loop tail). Plain constant on SPIR-V (rust-gpu has no
/// `black_box`; naga's uniformity analysis keeps barriers sound there).
#[inline(always)]
fn opaque_bound(n: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    {
        n
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        core::hint::black_box(n)
    }
}

#[inline(always)]
fn reduce_workspace_sum<T>(thread_id: usize, stride: usize, workspace: &mut impl MaybeIndexUnchecked<T>)
where
    T: Copy + Add<T, Output = T>,
{
    if thread_id < stride {
        workspace.write(
            thread_id,
            workspace.read(thread_id) + workspace.read(thread_id + stride),
        );
    }
    khal_std::sync::workgroup_memory_barrier_with_group_sync();
}

#[inline(always)]
fn reduce_workspace_mul<T>(thread_id: usize, stride: usize, workspace: &mut impl MaybeIndexUnchecked<T>)
where
    T: Copy + Mul<T, Output = T>,
{
    if thread_id < stride {
        workspace.write(
            thread_id,
            workspace.read(thread_id) * workspace.read(thread_id + stride),
        );
    }
    khal_std::sync::workgroup_memory_barrier_with_group_sync();
}

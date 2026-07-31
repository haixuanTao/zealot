use khal_builder::KhalBuilder;
use std::path::PathBuf;

fn main() {
    let output_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR not set by cargo"))
        .join("shaders-spirv");
    KhalBuilder::from_dependency("zealot_obs_shaders", true)
        .feature("unsafe_remove_boundchecks")
        .build(output_dir);
}

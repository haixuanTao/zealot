// Re-exports this crate's source location to host crates that build the
// shaders.
fn main() {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set by cargo");
    println!("cargo::metadata=manifest_dir={manifest_dir}");
    println!("cargo:rerun-if-changed=build.rs");
}

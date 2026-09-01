//! Link search path for the optional cuBLAS reference GEMM (`cutile` feature,
//! enabled at runtime with `BIPED_CUBLAS_GEMM=1` — see `src/biped/cublas.rs`).
//!
//! Uses `CUDA_TOOLKIT_PATH` when set (the same variable the cuTile build
//! already needs), else the usual install location.
fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");
    if std::env::var_os("CARGO_FEATURE_CUTILE").is_none() {
        return;
    }
    // cuBLAS lives in the versioned toolkit; prefer an explicit path, then the
    // `cuda` symlink. Only the *library* is needed, not the compiler.
    let mut roots: Vec<String> = vec![
        std::env::var("CUBLAS_PATH").unwrap_or_default(),
        std::env::var("CUDA_TOOLKIT_PATH").unwrap_or_default(),
        "/usr/local/cuda".into(),
    ];
    // The toolkit that provides nvcc is not necessarily the one that ships
    // cuBLAS (a CUDA 13 nvcc install can repoint /usr/local/cuda while cuBLAS
    // only exists under an older versioned tree), so scan the versioned dirs.
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        let mut versioned: Vec<String> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("cuda-"))
            })
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        versioned.sort();
        versioned.reverse(); // newest first
        roots.extend(versioned);
    }
    for r in roots.iter().filter(|r| !r.is_empty()) {
        let lib = format!("{r}/lib64");
        if std::path::Path::new(&format!("{lib}/libcublas.so")).exists() {
            println!("cargo:rustc-link-search=native={lib}");
            return;
        }
    }
    println!("cargo:warning=libcublas not found; BIPED_CUBLAS_GEMM will not link");
}

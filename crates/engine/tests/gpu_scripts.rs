//! GPU regression tests: each test runs a script from `tests/scripts/` in a headless engine
//! with Vulkan validation on (see `src/engine/script.rs` for the script language).
//!
//! They need a GPU with Vulkan 1.3+ and mesh shaders, so they are ignored by default:
//!
//! ```text
//! cargo test --release -p voxel-engine -- --ignored
//! UPDATE_GOLDEN=1 cargo test --release -p voxel-engine -- --ignored render   # accept new images
//! ```
//!
//! Set `VOXEL_TEST_NO_VALIDATE=1` on machines without the validation layers.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// One engine at a time: they share the GPU and `assets/prefabs`.
static GPU: Mutex<()> = Mutex::new(());

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn run(script: &str) {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let root = root();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_voxel-engine"));
    cmd.current_dir(&root).args(["--size", "1280x720", "--script"]).arg(format!("crates/engine/tests/scripts/{script}.vxs"));
    if std::env::var_os("VOXEL_TEST_NO_VALIDATE").is_none() {
        cmd.arg("--validate");
    }
    let status = cmd.status().expect("failed to start the engine");
    // selection.vxs saves a prefab; never leave it in the asset folder
    let _ = std::fs::remove_file(root.join("assets/prefabs/zz_script_test.ron"));
    assert!(status.success(), "script {script} failed (see the FAILED lines above)");
}

#[test]
#[ignore = "needs a Vulkan GPU with mesh shaders"]
fn ui_capture() {
    run("ui_capture");
}

#[test]
#[ignore = "needs a Vulkan GPU with mesh shaders"]
fn block_tool() {
    run("block_tool");
}

#[test]
#[ignore = "needs a Vulkan GPU with mesh shaders"]
fn brushes() {
    run("brushes");
}

#[test]
#[ignore = "needs a Vulkan GPU with mesh shaders"]
fn selection() {
    run("selection");
}

#[test]
#[ignore = "needs a Vulkan GPU with mesh shaders"]
fn render() {
    run("render");
}

#[test]
#[ignore = "needs a Vulkan GPU with mesh shaders"]
fn apron() {
    run("apron");
}

#[test]
#[ignore = "needs a Vulkan GPU with mesh shaders"]
fn layers() {
    run("layers");
}

#[test]
#[ignore = "needs a Vulkan GPU with mesh shaders"]
fn world_save() {
    run("world_save");
}

#[test]
#[ignore = "needs a Vulkan GPU with mesh shaders"]
fn water() {
    run("water");
}

#[test]
#[ignore = "needs a Vulkan GPU with mesh shaders"]
fn blocks() {
    run("blocks");
}

#[test]
#[ignore = "needs a Vulkan GPU with mesh shaders"]
fn props() {
    run("props");
}

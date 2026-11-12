//! World save files (`saves/NAME.vxw`, RON text): the seed, camera, time of day and every edited
//! block. The world itself is regenerated from the generator graphs, so a save only holds what
//! the generator can't reproduce.
//!
//! Blocks are stored by name (`name`, or `name:state` for oriented blocks) through a palette, so
//! saves survive changes to the block list. Each chunk's edits are runs of consecutive cells (in
//! local index order: x fastest, then z, then y) that hold the same block, so filled boxes stay
//! small. Files from before this format (`seed`, `camera`, raw block ids) still load.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use glam::IVec3;
use serde::{Deserialize, Serialize};

use crate::blocks::BlockRegistry;
use crate::world::edits::EditOverlay;

pub const EXTENSION: &str = "vxw";
pub const DIR: &str = "saves";
const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraState {
    pub pos: [f64; 3],
    /// radians
    pub yaw: f32,
    pub pitch: f32,
}

/// What a world file holds besides the edits.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldState {
    pub seed: u32,
    pub camera: CameraState,
    pub time_of_day: f32,
}

/// Edits of one cell run: (first local index, length, palette index).
type Run = (u16, u16, u16);

#[derive(Serialize, Deserialize)]
struct WorldFile {
    version: u32,
    seed: u32,
    camera: CameraState,
    time_of_day: f32,
    /// block names, `name` or `name:state`
    palette: Vec<String>,
    /// chunk -> runs of cells
    chunks: Vec<([i32; 3], Vec<Run>)>,
}

/// The format before version 1: block values were raw registry ids.
#[derive(Deserialize)]
struct LegacyFile {
    seed: u32,
    camera: [f64; 3],
    overlay: EditOverlay,
}

/// `saves/NAME.vxw` for a name typed in the UI (anything but letters, digits, `-` and `_`
/// becomes `_`).
pub fn path_for(name: &str) -> Option<PathBuf> {
    let name: String = name.trim().chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    (!name.is_empty()).then(|| Path::new(DIR).join(format!("{name}.{EXTENSION}")))
}

/// The world's display name: the file name without its extension.
pub fn name_of(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Saved worlds in `saves/`, newest first: (path, modified, size in bytes).
pub fn list() -> Vec<(PathBuf, std::time::SystemTime, u64)> {
    let mut out: Vec<_> = std::fs::read_dir(DIR)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == EXTENSION))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((e.path(), m.modified().ok()?, m.len()))
        })
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

pub fn to_string(state: &WorldState, overlay: &EditOverlay, reg: &BlockRegistry) -> Result<String> {
    let mut palette: Vec<String> = Vec::new();
    let mut index: HashMap<u16, u16> = HashMap::new();
    let mut keys: Vec<&IVec3> = overlay.chunks.keys().collect();
    keys.sort_by_key(|c| (c.y, c.z, c.x));
    let mut chunks = Vec::with_capacity(keys.len());
    for c in keys {
        let cells = &overlay.chunks[c];
        let mut sorted: Vec<(u16, u16)> = cells.iter().map(|(&i, &b)| (i, b)).collect();
        sorted.sort_unstable();
        let mut runs: Vec<Run> = Vec::new();
        for (i, b) in sorted {
            let p = *index.entry(b).or_insert_with(|| {
                palette.push(reg.value_name(b));
                (palette.len() - 1) as u16
            });
            match runs.last_mut() {
                Some((start, len, q)) if *q == p && *start as u32 + *len as u32 == i as u32 => *len += 1,
                _ => runs.push((i, 1, p)),
            }
        }
        chunks.push((c.to_array(), runs));
    }
    let file = WorldFile { version: VERSION, seed: state.seed, camera: state.camera, time_of_day: state.time_of_day, palette, chunks };
    Ok(ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::new().depth_limit(2))?)
}

/// Parse a world file. Returns the state, the edits and the names of blocks this build doesn't
/// know (their cells are left to the generator).
pub fn from_str(s: &str, reg: &BlockRegistry) -> Result<(WorldState, EditOverlay, Vec<String>)> {
    let file: WorldFile = match ron::from_str(s) {
        Ok(f) => f,
        Err(e) => {
            // the format before version 1
            let Ok(old) = ron::from_str::<LegacyFile>(s) else { return Err(e).context("not a world file") };
            let state = WorldState { seed: old.seed, camera: CameraState { pos: old.camera, yaw: 0.0, pitch: 0.0 }, time_of_day: 0.3 };
            return Ok((state, old.overlay, Vec::new()));
        }
    };
    if file.version > VERSION {
        bail!("world file version {} is newer than this build reads ({VERSION})", file.version);
    }
    let mut unknown = Vec::new();
    let values: Vec<Option<u16>> = file
        .palette
        .iter()
        .map(|n| {
            let v = reg.parse_value(n);
            if v.is_none() {
                unknown.push(n.clone());
            }
            v
        })
        .collect();
    let mut overlay = EditOverlay::default();
    for (c, runs) in file.chunks {
        let mut cells = HashMap::new();
        for (start, len, p) in runs {
            let Some(&Some(v)) = values.get(p as usize) else { continue };
            for i in start as u32..(start as u32 + len as u32).min(32768) {
                cells.insert(i as u16, v);
            }
        }
        if !cells.is_empty() {
            overlay.chunks.insert(IVec3::from(c), cells);
        }
    }
    let state = WorldState { seed: file.seed, camera: file.camera, time_of_day: file.time_of_day };
    Ok((state, overlay, unknown))
}

/// Write atomically: a crash mid-save leaves the previous file intact.
pub fn save(path: &Path, state: &WorldState, overlay: &EditOverlay, reg: &BlockRegistry) -> Result<()> {
    if let Some(d) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension(format!("{EXTENSION}.tmp"));
    std::fs::write(&tmp, to_string(state, overlay, reg)?).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub fn load(path: &Path, reg: &BlockRegistry) -> Result<(WorldState, EditOverlay, Vec<String>)> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    from_str(&s, reg).with_context(|| format!("loading {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::STATE_SHIFT;

    fn state() -> WorldState {
        WorldState { seed: 7, camera: CameraState { pos: [1.5, 80.25, -3.0], yaw: 0.5, pitch: -0.25 }, time_of_day: 0.42 }
    }

    #[test]
    fn round_trip() {
        let reg = BlockRegistry::new();
        let mut o = EditOverlay::default();
        let stone = reg.id("stone");
        let log_x = reg.id("oak_log") | (1 << STATE_SHIFT);
        let mut writes = Vec::new();
        for x in -3..5 {
            writes.push((IVec3::new(x, 70, 10), stone)); // crosses a chunk border
        }
        writes.push((IVec3::new(2, 71, 10), log_x));
        writes.push((IVec3::new(100, -40, 7), 0)); // broken block (air)
        o.apply("test", &writes);
        let s = to_string(&state(), &o, &reg).unwrap();
        assert!(s.contains("\"oak_log:1\"") && s.contains("\"stone\""), "{s}");
        let (st, back, unknown) = from_str(&s, &reg).unwrap();
        assert_eq!(st, state());
        assert!(unknown.is_empty());
        assert_eq!(back.chunks, o.chunks);
        // the row of stone inside chunk x = 0 is one run
        let file: WorldFile = ron::from_str(&s).unwrap();
        let (_, runs) = file.chunks.iter().find(|(c, _)| *c == [0, 2, 0]).unwrap();
        assert_eq!(runs.len(), 2, "{runs:?}"); // 5 stone, then the log one row up
    }

    #[test]
    fn unknown_blocks_are_skipped() {
        let reg = BlockRegistry::new();
        let s = r#"(version: 1, seed: 1, camera: (pos: (0, 0, 0), yaw: 0, pitch: 0), time_of_day: 0.5,
            palette: ["stone", "unobtainium"], chunks: [((0, 0, 0), [(0, 2, 0), (2, 3, 1)])])"#;
        let (_, o, unknown) = from_str(s, &reg).unwrap();
        assert_eq!(unknown, ["unobtainium"]);
        assert_eq!(o.chunks[&IVec3::ZERO].len(), 2);
    }

    #[test]
    fn legacy_files_load() {
        let reg = BlockRegistry::new();
        let s = "(seed: 3, camera: (1.0, 2.0, 3.0), overlay: (chunks: {(0, 1, 0): {5: 1}}))";
        let (st, o, _) = from_str(s, &reg).unwrap();
        assert_eq!((st.seed, st.camera.pos), (3, [1.0, 2.0, 3.0]));
        assert_eq!(o.get(IVec3::new(5, 32, 0)), Some(1));
    }

    #[test]
    fn names_become_paths() {
        assert_eq!(path_for(" my world! "), Some(PathBuf::from("saves/my_world_.vxw")));
        assert_eq!(path_for("  "), None);
        assert_eq!(name_of(Path::new("saves/castle.vxw")), "castle");
    }
}

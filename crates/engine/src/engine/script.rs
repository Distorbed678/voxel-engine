//! Scripted input playback for headless runs (`--script FILE`).
//!
//! A script drives the camera, the editing tools and synthetic mouse/keyboard input one frame at
//! a time, takes screenshots and checks results. The GPU regression tests
//! (`crates/engine/tests/gpu_scripts.rs`) and the README media are scripts. Scripted runs use a
//! fixed 1/60 s time step, fixed generation batches and the default UI layout, so a script gives
//! the same frames every time on a given GPU and driver.
//!
//! One command per line; `#` starts a comment; angles are in degrees; positions are blocks.
//!
//! | command | effect |
//! |---|---|
//! | `pos X Y Z` / `look YAW PITCH` | place the camera |
//! | `move X Y Z YAW PITCH FRAMES` | glide the camera there (smoothstep) |
//! | `waypoint X Y Z YAW PITCH` ... `fly_path FRAMES` | fly a smooth spline from the camera through the waypoints |
//! | `find_biome NAME [HEIGHT]` | survey the generator, go to the nearest large patch of a biome, HEIGHT blocks above its highest block |
//! | `anim T\|run` | freeze the animation clock (wind, water, clouds) at T seconds / let it run |
//! | `find_cave [MIN_HEIGHT]` | step around from the camera until a column has an air pocket at least MIN_HEIGHT tall below sea level, and stand in it |
//! | `time T` / `view N` / `ui on\|off` / `fly on\|off` | time of day, debug view, UI, fly mode |
//! | `clouds on\|off` / `fog on\|off` / `shadows on\|off` / `taa on\|off` / `bloom on\|off` / `block_light on\|off` | render toggles |
//! | `wait N` / `settle [MAX]` | wait N frames / until streaming, meshing and shader builds are idle |
//! | `screenshot PATH` / `record DIR\|off` | save this frame / save every frame as `DIR/frame_NNNNN.png` |
//! | `cursor X Y` | move the mouse to a pixel (picking follows it; waits for the pick) |
//! | `click left\|right\|middle [shift]` / `hold BUTTON FRAMES [shift]` / `scroll DY` | mouse input |
//! | `drag X1 Y1 X2 Y2 FRAMES [BUTTON]` | press at one pixel, move to another over FRAMES, release |
//! | `key [ctrl+][shift+]NAME` | press a key for one frame, e.g. `key ctrl+z` |
//! | `tool block\|brush\|select` / `block NAME` | choose the tool / selected block |
//! | `brush add\|remove\|paint\|smooth [RADIUS] [sphere\|cube\|cylinder]` | brush settings |
//! | `select X1 Y1 Z1 X2 Y2 Z2` | set the box selection |
//! | `op fill\|clear\|copy\|paste\|rotate\|replace FROM\|save_prefab NAME` | selection operations |
//! | `expect_pick NAME\|none` | block under the cursor (from the GPU pick) |
//! | `expect_block X Y Z NAME` | actual block at a position (GPU region read: generation + edits) |
//! | `expect_capture true\|false` | whether the UI owns the mouse |
//! | `expect_undo N [LABEL]` | undo history length and the label of the newest entry |
//! | `expect_prefab NAME` | a prefab of that name is loaded (e.g. after `op save_prefab`) |
//! | `expect_selected NAME` | the block selected for placing (e.g. after a middle-click pick) |
//! | `expect_clipboard SX SY SZ` / `expect_selection X1 Y1 Z1 X2 Y2 Z2` | tool state |
//! | `expect_image GOLDEN [TOL]` | compare this frame against a reference PNG (created if missing, `UPDATE_GOLDEN=1` rewrites) |
//! | `check_apron CX CY CZ` | (debugging) compare a LOD0 chunk's generated apron cells with its neighbours' blocks |
//! | `log TEXT` | print a line |

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use glam::{DVec3, IVec3};
use winit::dpi::PhysicalPosition;
use winit::event::{DeviceId, ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::keyboard::KeyCode;

use super::{Engine, FIF};
use crate::probe::{MAP_N, ProbeResult};
use crate::tools::{BrushMode, Shape, ToolKind};
use crate::world::RegionRead;

/// Height range searched for the ground after `find_biome`.
const GROUND_MIN: i32 = -64;
const GROUND_MAX: i32 = 447;

/// Where `expect_image` writes its screenshots and diff images.
const OUT_DIR: &str = "target/script-out";

#[derive(Clone, Debug)]
enum Op {
    Fill,
    Clear,
    Copy,
    Paste,
    Rotate,
    Replace(String),
    SavePrefab(String),
}

#[derive(Clone, Debug)]
enum Cmd {
    Pos(DVec3),
    Look(f32, f32),
    Move { pos: DVec3, yaw: f32, pitch: f32, frames: u32 },
    Waypoint(DVec3, f32, f32),
    FlyPath(u32),
    FindBiome { name: String, height: f64 },
    FindCave(i32),
    Anim(Option<f32>),
    Time(f32),
    View(u32),
    Ui(bool),
    Fly(bool),
    Clouds(bool),
    Fog(bool),
    Shadows(bool),
    Taa(bool),
    Bloom(bool),
    BlockLight(bool),
    Wait(u32),
    Settle(u32),
    Screenshot(PathBuf),
    Record(Option<PathBuf>),
    Cursor(f64, f64),
    Click { button: MouseButton, shift: bool },
    Hold { button: MouseButton, frames: u32, shift: bool },
    Scroll(f32),
    Drag { from: (f64, f64), to: (f64, f64), frames: u32, button: MouseButton },
    Key { code: KeyCode, ctrl: bool, shift: bool },
    Tool(ToolKind),
    Block(String),
    Brush { mode: BrushMode, radius: Option<i32>, shape: Option<Shape> },
    Select(IVec3, IVec3),
    Op(Op),
    ExpectPick(Option<String>),
    ExpectBlock(IVec3, String),
    ExpectCapture(bool),
    ExpectSelected(String),
    ExpectPrefab(String),
    ExpectUndo(usize, Option<String>),
    ExpectClipboard(IVec3),
    ExpectSelection(IVec3, IVec3),
    ExpectImage { golden: PathBuf, tol: f32 },
    CheckApron(IVec3),
    Log(String),
}

/// A command that spans several frames.
enum Busy {
    Frames(u32),
    Settle { left: u32, quiet: u32 },
    Survey { biome: u32, height: f64, spacing: f64, sent: bool, left: u32 },
    Move { from: (DVec3, f32, f32), to: (DVec3, f32, f32), i: u32, n: u32 },
    /// Catmull-Rom through camera keys (position, yaw, pitch), eased over the whole path.
    Path { keys: Vec<(DVec3, f32, f32)>, i: u32, n: u32 },
    /// Mouse drag: 2 frames of hover, press, `n` frames of motion, release.
    Drag { from: (f64, f64), to: (f64, f64), button: MouseButton, i: u32, n: u32 },
    Region { id: u64, pos: IVec3, want: u16, left: u32 },
    /// Column read under the camera after `find_biome`: lands HEIGHT above the top block.
    Ground { id: u64, height: f64, left: u32 },
    /// Region read for `check_apron`.
    Apron { id: u64, chunk: IVec3, left: u32 },
    /// Column read for `find_cave`.
    Cave { id: u64, min_height: i32, tries: u32, left: u32 },
    Image { shot: PathBuf, golden: PathBuf, tol: f32, left: u32 },
}

pub struct Script {
    path: PathBuf,
    cmds: Vec<(usize, Cmd)>,
    pc: usize,
    line: usize,
    busy: Option<Busy>,
    record: Option<(PathBuf, u32)>,
    /// Waypoints collected for the next `fly_path`.
    waypoints: Vec<(DVec3, f32, f32)>,
    /// Synthetic mouse buttons to release: (button, frames left).
    release_mouse: Vec<(MouseButton, u32)>,
    /// Synthetic keys to release: (key, frames left).
    release_keys: Vec<(KeyCode, u32)>,
    pub failures: Vec<String>,
    pub done: bool,
    started: bool,
}

impl Script {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading script {}", path.display()))?;
        let mut cmds = Vec::new();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let cmd = parse(line).with_context(|| format!("{}:{}: `{line}`", path.display(), i + 1))?;
            cmds.push((i + 1, cmd));
        }
        Ok(Self {
            path: path.to_path_buf(),
            cmds,
            pc: 0,
            line: 0,
            busy: None,
            record: None,
            waypoints: Vec::new(),
            release_mouse: Vec::new(),
            release_keys: Vec::new(),
            failures: Vec::new(),
            done: false,
            started: false,
        })
    }

    fn fail(&mut self, line: usize, msg: impl Into<String>) {
        let m = format!("{}:{line}: {}", self.path.display(), msg.into());
        log::error!("script: {m}");
        self.failures.push(m);
    }
}

// ---------------------------------------------------------------------------------------------
// parsing
// ---------------------------------------------------------------------------------------------

fn num<T: std::str::FromStr>(t: Option<&&str>, what: &str) -> Result<T> {
    let t = t.with_context(|| format!("missing {what}"))?;
    t.parse().map_err(|_| anyhow::anyhow!("bad {what} `{t}`"))
}

fn ivec(t: &[&str], what: &str) -> Result<IVec3> {
    Ok(IVec3::new(num(t.first(), what)?, num(t.get(1), what)?, num(t.get(2), what)?))
}

fn on_off(t: Option<&&str>) -> Result<bool> {
    match t.copied() {
        Some("on" | "true" | "1") => Ok(true),
        Some("off" | "false" | "0") => Ok(false),
        other => bail!("expected on/off, got {other:?}"),
    }
}

fn button(t: Option<&&str>) -> Result<MouseButton> {
    match t.copied() {
        Some("left") => Ok(MouseButton::Left),
        Some("right") => Ok(MouseButton::Right),
        Some("middle") => Ok(MouseButton::Middle),
        other => bail!("expected left/right/middle, got {other:?}"),
    }
}

fn key_code(name: &str) -> Option<KeyCode> {
    use KeyCode::*;
    const LETTERS: [KeyCode; 26] = [
        KeyA, KeyB, KeyC, KeyD, KeyE, KeyF, KeyG, KeyH, KeyI, KeyJ, KeyK, KeyL, KeyM, KeyN, KeyO, KeyP, KeyQ, KeyR, KeyS,
        KeyT, KeyU, KeyV, KeyW, KeyX, KeyY, KeyZ,
    ];
    const DIGITS: [KeyCode; 10] = [Digit0, Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9];
    const FKEYS: [KeyCode; 12] = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12];
    let n = name.to_ascii_lowercase();
    let b = n.as_bytes();
    Some(match n.as_str() {
        "tab" => Tab,
        "esc" | "escape" => Escape,
        "space" => Space,
        "enter" => Enter,
        "del" | "delete" => Delete,
        _ if b.len() == 1 && b[0].is_ascii_lowercase() => LETTERS[(b[0] - b'a') as usize],
        _ if b.len() == 1 && b[0].is_ascii_digit() => DIGITS[(b[0] - b'0') as usize],
        _ if b.len() >= 2 && b[0] == b'f' => *FKEYS.get(n[1..].parse::<usize>().ok()?.checked_sub(1)?)?,
        _ => return None,
    })
}

fn parse(line: &str) -> Result<Cmd> {
    let t: Vec<&str> = line.split_whitespace().collect();
    let a = &t[1..];
    Ok(match t[0] {
        "pos" => Cmd::Pos(DVec3::new(num(a.first(), "x")?, num(a.get(1), "y")?, num(a.get(2), "z")?)),
        "look" => Cmd::Look(num(a.first(), "yaw")?, num(a.get(1), "pitch")?),
        "move" => Cmd::Move {
            pos: DVec3::new(num(a.first(), "x")?, num(a.get(1), "y")?, num(a.get(2), "z")?),
            yaw: num(a.get(3), "yaw")?,
            pitch: num(a.get(4), "pitch")?,
            frames: num(a.get(5), "frames")?,
        },
        "waypoint" => Cmd::Waypoint(
            DVec3::new(num(a.first(), "x")?, num(a.get(1), "y")?, num(a.get(2), "z")?),
            num(a.get(3), "yaw")?,
            num(a.get(4), "pitch")?,
        ),
        "fly_path" => Cmd::FlyPath(num(a.first(), "frames")?),
        "find_cave" => Cmd::FindCave(a.first().map(|h| h.parse()).transpose()?.unwrap_or(8)),
        "find_biome" => Cmd::FindBiome {
            name: a.first().context("missing biome name")?.to_string(),
            height: a.get(1).map(|h| h.parse()).transpose()?.unwrap_or(40.0),
        },
        "anim" => Cmd::Anim(match a.first().copied() {
            Some("run") => None,
            _ => Some(num(a.first(), "seconds")?),
        }),
        "time" => Cmd::Time(num(a.first(), "time")?),
        "view" => Cmd::View(num(a.first(), "view")?),
        "ui" => Cmd::Ui(on_off(a.first())?),
        "fly" => Cmd::Fly(on_off(a.first())?),
        "clouds" => Cmd::Clouds(on_off(a.first())?),
        "fog" => Cmd::Fog(on_off(a.first())?),
        "shadows" => Cmd::Shadows(on_off(a.first())?),
        "taa" => Cmd::Taa(on_off(a.first())?),
        "bloom" => Cmd::Bloom(on_off(a.first())?),
        "block_light" => Cmd::BlockLight(on_off(a.first())?),
        "wait" => Cmd::Wait(num(a.first(), "frames")?),
        "settle" => Cmd::Settle(a.first().map(|f| f.parse()).transpose()?.unwrap_or(3000)),
        "screenshot" => Cmd::Screenshot(a.first().context("missing path")?.into()),
        "record" => match a.first().copied() {
            Some("off") => Cmd::Record(None),
            Some(dir) => Cmd::Record(Some(dir.into())),
            None => bail!("missing directory"),
        },
        "cursor" => Cmd::Cursor(num(a.first(), "x")?, num(a.get(1), "y")?),
        "click" => Cmd::Click { button: button(a.first())?, shift: a.get(1) == Some(&"shift") },
        "hold" => Cmd::Hold { button: button(a.first())?, frames: num(a.get(1), "frames")?, shift: a.get(2) == Some(&"shift") },
        "scroll" => Cmd::Scroll(num(a.first(), "amount")?),
        "drag" => Cmd::Drag {
            from: (num(a.first(), "x1")?, num(a.get(1), "y1")?),
            to: (num(a.get(2), "x2")?, num(a.get(3), "y2")?),
            frames: num(a.get(4), "frames")?,
            button: if a.len() > 5 { button(a.get(5))? } else { MouseButton::Left },
        },
        "key" => {
            let spec = a.first().context("missing key")?.to_ascii_lowercase();
            let parts: Vec<&str> = spec.split('+').collect();
            let (mods, name) = parts.split_at(parts.len() - 1);
            Cmd::Key {
                code: key_code(name[0]).with_context(|| format!("unknown key `{}`", name[0]))?,
                ctrl: mods.contains(&"ctrl"),
                shift: mods.contains(&"shift"),
            }
        }
        "tool" => Cmd::Tool(match a.first().copied() {
            Some("block") => ToolKind::Block,
            Some("brush") => ToolKind::Brush,
            Some("select") => ToolKind::Select,
            other => bail!("unknown tool {other:?}"),
        }),
        "block" => Cmd::Block(a.first().context("missing block name")?.to_string()),
        "brush" => {
            let mode = match a.first().copied() {
                Some("add") => BrushMode::Add,
                Some("remove") => BrushMode::Remove,
                Some("paint") => BrushMode::Paint,
                Some("smooth") => BrushMode::Smooth,
                other => bail!("unknown brush mode {other:?}"),
            };
            let mut radius = None;
            let mut shape = None;
            for w in &a[1..] {
                match *w {
                    "sphere" => shape = Some(Shape::Sphere),
                    "cube" => shape = Some(Shape::Cube),
                    "cylinder" => shape = Some(Shape::Cylinder),
                    r => radius = Some(r.parse().map_err(|_| anyhow::anyhow!("bad radius `{r}`"))?),
                }
            }
            Cmd::Brush { mode, radius, shape }
        }
        "select" => Cmd::Select(ivec(a, "corner")?, ivec(a.get(3..).unwrap_or(&[]), "corner")?),
        "op" => Cmd::Op(match a.first().copied() {
            Some("fill") => Op::Fill,
            Some("clear") => Op::Clear,
            Some("copy") => Op::Copy,
            Some("paste") => Op::Paste,
            Some("rotate") => Op::Rotate,
            Some("replace") => Op::Replace(a.get(1).context("missing block to replace")?.to_string()),
            Some("save_prefab") => Op::SavePrefab(a.get(1).context("missing prefab name")?.to_string()),
            other => bail!("unknown op {other:?}"),
        }),
        "expect_pick" => Cmd::ExpectPick(match a.first().copied() {
            Some("none") => None,
            Some(n) => Some(n.to_string()),
            None => bail!("missing block name"),
        }),
        "expect_block" => Cmd::ExpectBlock(ivec(a, "position")?, a.get(3).context("missing block name")?.to_string()),
        "expect_capture" => Cmd::ExpectCapture(on_off(a.first())?),
        "expect_undo" => Cmd::ExpectUndo(num(a.first(), "count")?, (a.len() > 1).then(|| a[1..].join(" "))),
        "expect_prefab" => Cmd::ExpectPrefab(a.first().context("missing prefab name")?.to_string()),
        "expect_selected" => Cmd::ExpectSelected(a.first().context("missing block name")?.to_string()),
        "expect_clipboard" => Cmd::ExpectClipboard(ivec(a, "size")?),
        "expect_selection" => Cmd::ExpectSelection(ivec(a, "corner")?, ivec(a.get(3..).unwrap_or(&[]), "corner")?),
        "expect_image" => Cmd::ExpectImage {
            golden: a.first().context("missing golden path")?.into(),
            tol: a.get(1).map(|t| t.parse()).transpose()?.unwrap_or(1.5),
        },
        "check_apron" => Cmd::CheckApron(ivec(a, "chunk")?),
        "log" => Cmd::Log(a.join(" ")),
        other => bail!("unknown command `{other}`"),
    })
}

// ---------------------------------------------------------------------------------------------
// execution
// ---------------------------------------------------------------------------------------------

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Catmull-Rom interpolation between p1 and p2.
fn catmull<T>(p0: T, p1: T, p2: T, p3: T, t: f64) -> T
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<f64, Output = T>,
{
    let (t2, t3) = (t * t, t * t * t);
    (p1 * 2.0 + (p2 - p0) * t + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2 + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * t3) * 0.5
}

/// Sample a camera path at u in [0, 1].
fn sample_path(keys: &[(DVec3, f32, f32)], u: f64) -> (DVec3, f32, f32) {
    let segs = keys.len() - 1;
    let x = (u * segs as f64).clamp(0.0, segs as f64);
    let k = (x.floor() as usize).min(segs - 1);
    let t = x - k as f64;
    let at = |i: isize| keys[i.clamp(0, keys.len() as isize - 1) as usize];
    let (a, b, c, d) = (at(k as isize - 1), at(k as isize), at(k as isize + 1), at(k as isize + 2));
    let ang = |f: fn(&(DVec3, f32, f32)) -> f32| catmull(DVec3::splat(f(&a) as f64), DVec3::splat(f(&b) as f64), DVec3::splat(f(&c) as f64), DVec3::splat(f(&d) as f64), t).x as f32;
    (catmull(a.0, b.0, c.0, d.0, t), ang(|k| k.1), ang(|k| k.2))
}

/// The cell of a survey grid deepest inside `biome`, preferring cells near the center.
/// `strict` requires the 5×5 neighbourhood to be mostly that biome.
fn best_cell(cells: &[ProbeResult], biome: u32, strict: bool) -> Option<usize> {
    let n = MAP_N as i32;
    let mut best: Option<((i32, i32), usize)> = None;
    for j in 0..n {
        for i in 0..n {
            if cells[(j * n + i) as usize].biome != biome {
                continue;
            }
            let mut score = 0;
            for dj in -2..=2 {
                for di in -2..=2 {
                    let (x, y) = (i + di, j + dj);
                    if x >= 0 && y >= 0 && x < n && y < n && cells[(y * n + x) as usize].biome == biome {
                        score += 1;
                    }
                }
            }
            let key = (score, -((i - n / 2).pow(2) + (j - n / 2).pow(2)));
            if best.is_none_or(|b| key > b.0) {
                best = Some((key, (j * n + i) as usize));
            }
        }
    }
    best.filter(|b| !strict || b.0.0 >= 15).map(|b| b.1)
}

/// Compare `shot` against `golden`. Returns a failure message, or None on a match. Goldens are
/// stored at half resolution, which keeps them small and ignores sub-pixel noise.
fn compare_images(shot: &Path, golden: &Path, tol: f32) -> Result<Option<String>> {
    let full = image::open(shot)?.to_rgb8();
    let a = image::imageops::resize(&full, (full.width() / 2).max(1), (full.height() / 2).max(1), image::imageops::FilterType::Triangle);
    let existed = golden.exists();
    if !existed || std::env::var_os("UPDATE_GOLDEN").is_some() {
        if let Some(d) = golden.parent() {
            std::fs::create_dir_all(d)?;
        }
        a.save(golden)?;
        log::warn!("script: golden image {} {}", golden.display(), if existed { "updated" } else { "created" });
        return Ok(None);
    }
    let b = image::open(golden)?.to_rgb8();
    if a.dimensions() != b.dimensions() {
        return Ok(Some(format!("{}: size {:?} differs from golden {:?}", shot.display(), a.dimensions(), b.dimensions())));
    }
    let mut diff = image::RgbImage::new(a.width(), a.height());
    let (mut sum, mut bad) = (0u64, 0u64);
    for ((pa, pb), pd) in a.pixels().zip(b.pixels()).zip(diff.pixels_mut()) {
        let d: [u8; 3] = std::array::from_fn(|c| pa.0[c].abs_diff(pb.0[c]));
        sum += d.iter().map(|v| *v as u64).sum::<u64>();
        let m = *d.iter().max().unwrap();
        if m > 24 {
            bad += 1;
        }
        *pd = image::Rgb([m.saturating_mul(4), m.saturating_mul(4), m.saturating_mul(4)]);
    }
    let px = (a.width() * a.height()) as f64;
    let mean = sum as f64 / (px * 3.0);
    let frac = bad as f64 / px;
    if mean > tol as f64 || frac > 0.02 {
        let dpath = shot.with_extension("diff.png");
        diff.save(&dpath)?;
        return Ok(Some(format!(
            "{} differs from {}: mean diff {mean:.2} (tolerance {tol}), {:.2}% of pixels off; diff in {}",
            shot.display(),
            golden.display(),
            frac * 100.0,
            dpath.display()
        )));
    }
    log::info!("script: {} matches golden (mean diff {mean:.2})", shot.display());
    Ok(None)
}

fn mouse_event(button: MouseButton, pressed: bool) -> WindowEvent {
    WindowEvent::MouseInput {
        device_id: DeviceId::dummy(),
        state: if pressed { ElementState::Pressed } else { ElementState::Released },
        button,
    }
}

impl Engine {
    /// Run this frame's share of the script (called before input handling).
    pub(super) fn script_step(&mut self) -> Result<()> {
        let Some(mut s) = self.script.take() else { return Ok(()) };
        let r = self.run_script(&mut s);
        self.script = Some(s);
        r
    }

    /// Region reads requested by `expect_block` end here; other reads are handed back.
    pub(super) fn script_take_region(&mut self, r: RegionRead) -> Option<RegionRead> {
        let Some(s) = &mut self.script else { return Some(r) };
        if let Some(Busy::Ground { id, height, .. }) = &s.busy {
            if *id != r.id {
                return Some(r);
            }
            let top = (0..r.size.y).rev().find(|&y| r.get(r.min + IVec3::new(0, y, 0)) != 0).map(|y| r.min.y + y + 1);
            self.camera.pos.y = top.unwrap_or(64) as f64 + *height;
            log::info!("script: biome found at {:.0} {:.0} {:.0}", self.camera.pos.x, self.camera.pos.y, self.camera.pos.z);
            s.busy = None;
            return None;
        }
        if let Some(Busy::Apron { id, chunk, .. }) = &s.busy {
            if *id != r.id {
                return Some(r);
            }
            let chunk = *chunk;
            let line = s.line;
            match self.world.raw_slots.remove(&chunk) {
                None => s.fail(line, "raw slot not captured"),
                Some(raw) => {
                    let origin = chunk * crate::world::CHUNK;
                    let ca = crate::world::CA;
                    let mut bad = 0;
                    for y in 0..ca {
                        for z in 0..ca {
                            for x in 0..ca {
                                let l = IVec3::new(x, y, z) - IVec3::ONE;
                                let interior = l.cmpge(IVec3::ZERO).all() && l.cmplt(IVec3::splat(crate::world::CHUNK)).all();
                                if interior {
                                    continue;
                                }
                                let got = raw[((y * ca + z) * ca + x) as usize];
                                let want = r.get(origin + l);
                                if got != want {
                                    if bad < 8 {
                                        let name = |b: u16| self.blocks.blocks.get(b as usize).map(|d| d.name).unwrap_or("?");
                                        log::warn!("script: apron {} has {} but the world has {}", origin + l, name(got), name(want));
                                    }
                                    bad += 1;
                                }
                            }
                        }
                    }
                    if bad > 0 {
                        s.fail(line, format!("{bad} apron cells of chunk {chunk} differ from their neighbours"));
                    } else {
                        log::info!("script: apron of chunk {chunk} matches");
                    }
                }
            }
            s.busy = None;
            return None;
        }
        if let Some(Busy::Cave { id, min_height, tries, .. }) = &s.busy {
            if *id != r.id {
                return Some(r);
            }
            let (min_height, tries) = (*min_height, *tries);
            // below the top block, the highest air run at least `min_height` tall
            let col: Vec<u16> = (0..r.size.y).map(|y| r.get(r.min + IVec3::new(0, y, 0))).collect();
            let top = col.iter().rposition(|&b| b != 0);
            let mut found = None;
            if let Some(top) = top {
                let mut y = top as i32;
                while y >= 0 && found.is_none() {
                    if col[y as usize] == 0 {
                        let end = y;
                        while y >= 0 && col[y as usize] == 0 {
                            y -= 1;
                        }
                        // below sea level: air under trees or floating islands isn't a cave
                        if end - y >= min_height && r.min.y + end < 56 {
                            found = Some(r.min.y + y + 1 + (end - y) / 3);
                        }
                    }
                    y -= 1;
                }
            }
            let line = s.line;
            match found {
                Some(cy) => {
                    self.camera.pos.y = cy as f64 + 0.5;
                    log::info!("script: cave found at {:.0} {:.0} {:.0}", self.camera.pos.x, self.camera.pos.y, self.camera.pos.z);
                    s.busy = None;
                }
                None if tries == 0 => {
                    s.fail(line, "no cave found");
                    s.busy = None;
                }
                None => {
                    self.camera.pos.x += 53.0;
                    self.camera.pos.z += 31.0;
                    let (x, z) = (self.camera.pos.x.floor() as i32, self.camera.pos.z.floor() as i32);
                    let id = self.world.request_region(IVec3::new(x, GROUND_MIN, z), IVec3::new(x, 200, z));
                    s.busy = Some(Busy::Cave { id, min_height, tries: tries - 1, left: 600 });
                }
            }
            return None;
        }
        let Some(Busy::Region { id, pos, want, .. }) = &s.busy else { return Some(r) };
        if *id != r.id {
            return Some(r);
        }
        let (pos, want) = (*pos, *want);
        let got = r.get(pos);
        if got != want {
            let name = |b: u16| self.blocks.blocks.get(b as usize).map(|d| d.name).unwrap_or("?");
            let msg = format!("block at {pos}: expected {}, got {}", name(want), name(got));
            let line = s.line;
            s.fail(line, msg);
        }
        s.busy = None;
        None
    }

    fn world_idle(&self) -> bool {
        let st = &self.world.stats;
        st.pending == 0
            && st.in_flight == 0
            && st.retiring == 0
            && self.world.region_reads.is_empty()
            && self.world.copies.is_empty()
            && self.world.record_updates.is_empty()
            && self.wgpu.busy_slots() == 0
            && self.tools.pending_ops() == 0
            && !self.wg.is_building()
            && !self.wg.is_optimizing()
    }

    fn run_script(&mut self, s: &mut Script) -> Result<()> {
        if s.done {
            return Ok(());
        }
        if !s.started {
            // the mouse starts in the middle of the screen
            s.started = true;
            let e = self.renderer.extent;
            let (x, y) = (e.width as f64 * 0.5, e.height as f64 * 0.5);
            self.handle_window_event(&WindowEvent::CursorMoved { device_id: DeviceId::dummy(), position: PhysicalPosition::new(x, y) });
        }
        s.release_keys.retain_mut(|(k, left)| {
            if *left == 0 {
                self.input.held.remove(k);
                false
            } else {
                *left -= 1;
                true
            }
        });
        let mut released = Vec::new();
        s.release_mouse.retain_mut(|(b, left)| {
            if *left == 0 {
                released.push(*b);
                false
            } else {
                *left -= 1;
                true
            }
        });
        for b in released {
            self.handle_window_event(&mouse_event(b, false));
        }
        if let Some((dir, n)) = &mut s.record
            && self.screenshot_request.is_none() {
                self.screenshot_request = Some(dir.join(format!("frame_{n:05}.png")));
                *n += 1;
            }
        if let Some(b) = s.busy.take() {
            s.busy = self.advance(s, b)?;
            if s.busy.is_some() {
                return Ok(());
            }
        }
        while s.pc < s.cmds.len() {
            let (line, cmd) = s.cmds[s.pc].clone();
            s.pc += 1;
            s.line = line;
            s.busy = self.exec(s, line, cmd)?;
            if s.busy.is_some() {
                return Ok(());
            }
        }
        s.done = true;
        // let the last frames (and their screenshots) finish, then quit
        self.opts.frames = Some(self.frame_counter + 1);
        if s.failures.is_empty() {
            log::info!("script {}: all checks passed", s.path.display());
        } else {
            log::error!("script {}: {} check(s) failed", s.path.display(), s.failures.len());
        }
        Ok(())
    }

    fn advance(&mut self, s: &mut Script, b: Busy) -> Result<Option<Busy>> {
        let line = s.line;
        Ok(match b {
            Busy::Frames(n) => (n > 1).then_some(Busy::Frames(n - 1)),
            Busy::Settle { left, quiet } => {
                let quiet = if self.world_idle() { quiet + 1 } else { 0 };
                if quiet >= 4 {
                    // restart the TAA jitter sequence and history: what follows doesn't depend on
                    // how many frames streaming took
                    self.taa_frame = 0;
                    self.taa.reset = true;
                    None
                } else if left == 0 {
                    s.fail(line, "settle timed out (world still streaming)");
                    None
                } else {
                    Some(Busy::Settle { left: left - 1, quiet })
                }
            }
            Busy::Survey { biome, height, spacing, sent, left } => {
                if !sent {
                    self.probes.survey = None;
                    self.probes.survey_request = Some((self.camera.pos.x, self.camera.pos.z, spacing));
                    return Ok(Some(Busy::Survey { biome, height, spacing, sent: true, left }));
                }
                let Some(((ox, oz, sp), cells)) = self.probes.survey.take() else {
                    if left == 0 {
                        s.fail(line, "biome survey timed out");
                        return Ok(None);
                    }
                    return Ok(Some(Busy::Survey { biome, height, spacing, sent, left: left - 1 }));
                };
                let last = spacing >= 1024.0;
                match best_cell(&cells, biome, !last) {
                    Some(c) => {
                        let (i, j) = (c % MAP_N, c / MAP_N);
                        self.camera.pos.x = ox + (i as f64 + 0.5) * sp;
                        self.camera.pos.z = oz + (j as f64 + 0.5) * sp;
                        // the probe height ignores 3D terrain: read the real column
                        let (x, z) = (self.camera.pos.x.floor() as i32, self.camera.pos.z.floor() as i32);
                        let id = self.world.request_region(IVec3::new(x, GROUND_MIN, z), IVec3::new(x, GROUND_MAX, z));
                        Some(Busy::Ground { id, height, left: 600 })
                    }
                    None if last => {
                        s.fail(line, "biome not found within 32 km");
                        None
                    }
                    None => Some(Busy::Survey { biome, height, spacing: spacing * 2.0, sent: false, left: 60 }),
                }
            }
            Busy::Move { from, to, i, n } => {
                let i = i + 1;
                let t = smoothstep((i as f32 / n.max(1) as f32).min(1.0));
                self.camera.pos = from.0.lerp(to.0, t as f64);
                let mut dyaw = (to.1 - from.1) % std::f32::consts::TAU;
                if dyaw > std::f32::consts::PI {
                    dyaw -= std::f32::consts::TAU;
                } else if dyaw < -std::f32::consts::PI {
                    dyaw += std::f32::consts::TAU;
                }
                self.camera.yaw = from.1 + dyaw * t;
                self.camera.pitch = from.2 + (to.2 - from.2) * t;
                (i < n).then_some(Busy::Move { from, to, i, n })
            }
            Busy::Path { keys, i, n } => {
                let i = i + 1;
                let u = smoothstep((i as f32 / n.max(1) as f32).min(1.0)) as f64;
                let (p, yaw, pitch) = sample_path(&keys, u);
                self.camera.pos = p;
                self.camera.yaw = yaw;
                self.camera.pitch = pitch.clamp(-1.55, 1.55);
                (i < n).then_some(Busy::Path { keys, i, n })
            }
            Busy::Drag { from, to, button, i, n } => {
                // i: 0,1 hover at `from`; 2 press; 3..=n+2 move; n+3 release
                let i = i + 1;
                if i == 2 {
                    self.handle_window_event(&mouse_event(button, true));
                } else if i > 2 && i <= n + 2 {
                    let t = (i - 2) as f64 / n.max(1) as f64;
                    let p = (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t);
                    self.handle_window_event(&WindowEvent::CursorMoved { device_id: DeviceId::dummy(), position: PhysicalPosition::new(p.0, p.1) });
                } else if i == n + 3 {
                    self.handle_window_event(&mouse_event(button, false));
                }
                (i < n + 5).then_some(Busy::Drag { from, to, button, i, n })
            }
            Busy::Apron { id, chunk, left } => {
                if left == 0 {
                    s.fail(line, "apron read timed out");
                    None
                } else {
                    Some(Busy::Apron { id, chunk, left: left - 1 })
                }
            }
            Busy::Cave { id, min_height, tries, left } => {
                if left == 0 {
                    s.fail(line, "cave search timed out");
                    None
                } else {
                    Some(Busy::Cave { id, min_height, tries, left: left - 1 })
                }
            }
            Busy::Ground { id, height, left } => {
                if left == 0 {
                    s.fail(line, "ground read timed out");
                    None
                } else {
                    Some(Busy::Ground { id, height, left: left - 1 })
                }
            }
            Busy::Region { id, pos, want, left } => {
                if left == 0 {
                    s.fail(line, format!("region read at {pos} timed out"));
                    None
                } else {
                    Some(Busy::Region { id, pos, want, left: left - 1 })
                }
            }
            Busy::Image { shot, golden, tol, left } => {
                if left > 0 {
                    return Ok(Some(Busy::Image { shot, golden, tol, left: left - 1 }));
                }
                if !shot.exists() {
                    s.fail(line, format!("screenshot {} was not written", shot.display()));
                } else if let Some(msg) = compare_images(&shot, &golden, tol)? {
                    s.fail(line, msg);
                }
                None
            }
        })
    }

    fn exec(&mut self, s: &mut Script, line: usize, cmd: Cmd) -> Result<Option<Busy>> {
        let block_id = |e: &Engine, name: &str| e.blocks.try_id(name);
        match cmd {
            Cmd::Pos(p) => self.camera.pos = p,
            Cmd::Look(yaw, pitch) => {
                self.camera.yaw = yaw.to_radians();
                self.camera.pitch = pitch.to_radians().clamp(-1.55, 1.55);
            }
            Cmd::Move { pos, yaw, pitch, frames } => {
                let from = (self.camera.pos, self.camera.yaw, self.camera.pitch);
                return Ok(Some(Busy::Move { from, to: (pos, yaw.to_radians(), pitch.to_radians().clamp(-1.55, 1.55)), i: 0, n: frames }));
            }
            Cmd::Waypoint(p, yaw, pitch) => s.waypoints.push((p, yaw.to_radians(), pitch.to_radians())),
            Cmd::FlyPath(frames) => {
                let mut keys = vec![(self.camera.pos, self.camera.yaw, self.camera.pitch)];
                keys.append(&mut s.waypoints);
                // unwrap yaw so the path turns the short way round
                for k in 1..keys.len() {
                    let prev = keys[k - 1].1;
                    let mut y = keys[k].1;
                    while y - prev > std::f32::consts::PI {
                        y -= std::f32::consts::TAU;
                    }
                    while y - prev < -std::f32::consts::PI {
                        y += std::f32::consts::TAU;
                    }
                    keys[k].1 = y;
                }
                if keys.len() < 2 {
                    s.fail(line, "fly_path needs at least one waypoint");
                    return Ok(None);
                }
                return Ok(Some(Busy::Path { keys, i: 0, n: frames }));
            }
            Cmd::FindCave(min_height) => {
                let (x, z) = (self.camera.pos.x.floor() as i32, self.camera.pos.z.floor() as i32);
                let id = self.world.request_region(IVec3::new(x, GROUND_MIN, z), IVec3::new(x, 200, z));
                return Ok(Some(Busy::Cave { id, min_height, tries: 200, left: 600 }));
            }
            Cmd::FindBiome { name, height } => {
                let Some(biome) = self.wg.compiled.biomes.iter().position(|b| b.name == name) else {
                    let known: Vec<&str> = self.wg.compiled.biomes.iter().map(|b| b.name.as_str()).collect();
                    s.fail(line, format!("unknown biome `{name}` (known: {})", known.join(", ")));
                    return Ok(None);
                };
                return Ok(Some(Busy::Survey { biome: biome as u32, height, spacing: 32.0, sent: false, left: 60 }));
            }
            Cmd::Anim(t) => {
                self.anim_frozen = t.is_some();
                if let Some(t) = t {
                    self.anim_time = t;
                }
            }
            Cmd::Time(t) => self.atmosphere.time_of_day = t.rem_euclid(1.0),
            Cmd::View(v) => self.ui.debug_view = v,
            Cmd::Ui(on) => self.ui.show_ui = on,
            Cmd::Fly(on) => self.set_fly_mode(on),
            Cmd::Clouds(on) => self.ui.clouds.enabled = on,
            Cmd::Fog(on) => self.ui.fog = on,
            Cmd::Shadows(on) => self.ui.shadows = on,
            Cmd::Taa(on) => self.ui.taa = on,
            Cmd::Bloom(on) => self.ui.bloom = on,
            Cmd::BlockLight(on) => self.ui.block_light = on,
            Cmd::Wait(n) => return Ok((n > 0).then_some(Busy::Frames(n))),
            Cmd::Settle(max) => return Ok(Some(Busy::Settle { left: max, quiet: 0 })),
            Cmd::Screenshot(p) => {
                self.screenshot_request = Some(p);
                return Ok(Some(Busy::Frames(1)));
            }
            Cmd::Record(Some(dir)) => {
                std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
                s.record = Some((dir, 0));
            }
            Cmd::Record(None) => s.record = None,
            Cmd::Cursor(x, y) => {
                self.handle_window_event(&WindowEvent::CursorMoved { device_id: DeviceId::dummy(), position: PhysicalPosition::new(x, y) });
                // the GPU pick and ImGui hover follow a few frames later
                return Ok(Some(Busy::Frames(FIF as u32 + 2)));
            }
            Cmd::Click { button, shift } => {
                if shift {
                    self.input.held.insert(KeyCode::ShiftLeft);
                    s.release_keys.push((KeyCode::ShiftLeft, 0));
                }
                self.handle_window_event(&mouse_event(button, true));
                s.release_mouse.push((button, 0));
                return Ok(Some(Busy::Frames(1)));
            }
            Cmd::Hold { button, frames, shift } => {
                let held = frames.saturating_sub(1);
                if shift {
                    self.input.held.insert(KeyCode::ShiftLeft);
                    s.release_keys.push((KeyCode::ShiftLeft, held));
                }
                self.handle_window_event(&mouse_event(button, true));
                s.release_mouse.push((button, held));
                return Ok(Some(Busy::Frames(frames + 1)));
            }
            Cmd::Scroll(dy) => {
                self.handle_window_event(&WindowEvent::MouseWheel {
                    device_id: DeviceId::dummy(),
                    delta: MouseScrollDelta::LineDelta(0.0, dy),
                    phase: TouchPhase::Moved,
                });
                return Ok(Some(Busy::Frames(1)));
            }
            Cmd::Drag { from, to, frames, button } => {
                self.handle_window_event(&WindowEvent::CursorMoved { device_id: DeviceId::dummy(), position: PhysicalPosition::new(from.0, from.1) });
                return Ok(Some(Busy::Drag { from, to, button, i: 0, n: frames }));
            }
            Cmd::Key { code, ctrl, shift } => {
                for (on, k) in [(ctrl, KeyCode::ControlLeft), (shift, KeyCode::ShiftLeft), (true, code)] {
                    if on {
                        self.input.held.insert(k);
                        s.release_keys.push((k, 0));
                    }
                }
                self.input.pressed.insert(code);
                return Ok(Some(Busy::Frames(1)));
            }
            Cmd::Tool(k) => self.tools.kind = k,
            Cmd::Block(name) => match block_id(self, &name) {
                Some(id) => self.ui.selected_block = id,
                None => s.fail(line, format!("unknown block `{name}`")),
            },
            Cmd::Brush { mode, radius, shape } => {
                self.tools.kind = ToolKind::Brush;
                self.tools.mode = mode;
                if let Some(r) = radius {
                    self.tools.radius = r.clamp(1, 32);
                }
                if let Some(sh) = shape {
                    self.tools.shape = sh;
                }
            }
            Cmd::Select(a, b) => {
                self.tools.sel_a = Some(a);
                self.tools.sel_b = Some(b);
            }
            Cmd::Op(op) => match op {
                Op::Fill => self.tools.fill_selection(&mut self.world, self.ui.selected_block),
                Op::Clear => self.tools.fill_selection(&mut self.world, 0),
                Op::Copy => self.tools.copy_selection(&mut self.world),
                Op::Paste => self.tools.paste_at_pick(&mut self.world, self.pick),
                Op::Rotate => self.tools.rotate_clipboard(),
                Op::Replace(name) => match block_id(self, &name) {
                    Some(from) => self.tools.replace_in_selection(&mut self.world, from, self.ui.selected_block),
                    None => s.fail(line, format!("unknown block `{name}`")),
                },
                Op::SavePrefab(name) => self.tools.save_selection_as_prefab(&mut self.world, &name),
            },
            Cmd::ExpectPick(want) => {
                let got = self.pick.map(|h| self.blocks.blocks[h.block as usize].name.to_string());
                if got != want {
                    s.fail(line, format!("expected pick {want:?}, got {got:?}"));
                }
            }
            Cmd::ExpectBlock(pos, name) => match block_id(self, &name) {
                Some(want) => {
                    let id = self.world.request_region(pos, pos);
                    return Ok(Some(Busy::Region { id, pos, want, left: 600 }));
                }
                None => s.fail(line, format!("unknown block `{name}`")),
            },
            Cmd::ExpectCapture(want) => {
                let got = self.imgui.ctx.io().want_capture_mouse();
                if got != want {
                    s.fail(line, format!("expected UI mouse capture {want}, got {got}"));
                }
            }
            Cmd::ExpectUndo(n, label) => {
                let undo = &self.world.overlay.undo;
                let last = undo.last().map(|o| o.label.clone());
                if undo.len() != n || (label.is_some() && last != label) {
                    s.fail(line, format!("expected {n} undo entries (newest {label:?}), got {} (newest {last:?})", undo.len()));
                }
            }
            Cmd::ExpectPrefab(name) => {
                if !self.tools.prefabs.prefabs.iter().any(|p| p.name == name) {
                    s.fail(line, format!("prefab `{name}` is not loaded"));
                }
            }
            Cmd::ExpectSelected(want) => {
                let got = self.blocks.blocks[self.ui.selected_block as usize].name;
                if got != want {
                    s.fail(line, format!("expected selected block {want}, got {got}"));
                }
            }
            Cmd::ExpectClipboard(size) => {
                let got = self.tools.clipboard.as_ref().map(|c| c.size);
                if got != Some(size) {
                    s.fail(line, format!("expected clipboard {size}, got {got:?}"));
                }
            }
            Cmd::ExpectSelection(a, b) => {
                let got = self.tools.selection();
                if got != Some((a.min(b), a.max(b))) {
                    s.fail(line, format!("expected selection {a} {b}, got {got:?}"));
                }
            }
            Cmd::ExpectImage { golden, tol } => {
                let stem = golden.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "shot".into());
                let shot = Path::new(OUT_DIR).join(format!("{stem}.png"));
                let _ = std::fs::remove_file(&shot);
                self.screenshot_request = Some(shot.clone());
                return Ok(Some(Busy::Image { shot, golden, tol, left: FIF as u32 + 2 }));
            }
            Cmd::CheckApron(chunk) => {
                let o = chunk * crate::world::CHUNK;
                self.world.capture_raw.insert(chunk);
                let id = self.world.request_region(o - IVec3::ONE, o + IVec3::splat(crate::world::CHUNK));
                return Ok(Some(Busy::Apron { id, chunk, left: 600 }));
            }
            Cmd::Log(text) => log::info!("script: {text}"),
        }
        Ok(None)
    }
}

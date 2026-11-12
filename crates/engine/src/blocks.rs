//! Block registry: block definitions, their textures, the GPU block table and the GLSL header the
//! world generator includes.
//!
//! Blocks are data: `assets/blocks/NAME.ron`, one file per block (the file name is the block
//! name). The built-in list below is only the default set, written out by
//! `--export-block-defaults DIR` or when `assets/blocks` is empty. `assets/blocks/_order.ron` lists
//! the names in id order so ids stay the same between sessions where it matters (old raw-id
//! saves); everything persistent (graphs, prefabs, world files) refers to blocks by name.
//!
//! The block editor creates, edits and deletes blocks at runtime. Ids stay stable within a
//! session: a deleted block leaves a tombstone (it acts as air) until the next start. Changes are
//! collected in `BlockChanges`, which the engine applies once per frame (GPU table, texture
//! layers, generator header, remeshing).

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};

use crate::textures::{TexAsset, TextureStore};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u32)]
pub enum BlockKind {
    Air = 0,
    #[default]
    Opaque = 1,
    Cutout = 2,
    Transparent = 3,
    Model = 4,
}

impl BlockKind {
    pub const ALL: [BlockKind; 5] = [BlockKind::Air, BlockKind::Opaque, BlockKind::Cutout, BlockKind::Transparent, BlockKind::Model];
    pub fn label(self) -> &'static str {
        match self {
            BlockKind::Air => "Air (invisible)",
            BlockKind::Opaque => "Opaque cube",
            BlockKind::Cutout => "Cutout cube (leaves)",
            BlockKind::Transparent => "Transparent cube (glass, water)",
            BlockKind::Model => "Model",
        }
    }
}

/// Biome colour applied to the block's green parts (water: the whole texture).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u32)]
pub enum Tint {
    #[default]
    None = 0,
    Grass = 1,
    Foliage = 2,
    Water = 3,
}

impl Tint {
    pub const ALL: [Tint; 4] = [Tint::None, Tint::Grass, Tint::Foliage, Tint::Water];
    pub fn label(self) -> &'static str {
        match self {
            Tint::None => "None",
            Tint::Grass => "Grass (biome)",
            Tint::Foliage => "Foliage (biome)",
            Tint::Water => "Water (biome, water shading)",
        }
    }
}

/// Small models drawn by the model mesh shader (`shaders/render/models.glsl`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u16)]
pub enum Model {
    #[default]
    None = 0,
    /// two crossed quads (plants)
    Cross = 1,
    /// bottom half block
    Slab = 2,
    /// stairs (the state holds the facing)
    Stairs = 3,
    /// small hanging box
    Lantern = 4,
    /// small flat stones
    Pebbles = 5,
    /// stem + cap
    Mushroom = 6,
    /// crossed quads, slightly taller and wider
    TallCross = 7,
}

impl Model {
    pub const ALL: [Model; 7] = [Model::Cross, Model::TallCross, Model::Slab, Model::Stairs, Model::Lantern, Model::Pebbles, Model::Mushroom];
    pub fn label(self) -> &'static str {
        match self {
            Model::None => "None",
            Model::Cross => "Cross (plant)",
            Model::TallCross => "Tall cross (shrub)",
            Model::Slab => "Slab",
            Model::Stairs => "Stairs",
            Model::Lantern => "Lantern",
            Model::Pebbles => "Pebbles",
            Model::Mushroom => "Mushroom",
        }
    }
}

/// What a block's state means, which decides how it is oriented when placed and how it turns
/// with a rotated selection or prefab.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum Rotation {
    /// no state
    #[default]
    None = 0,
    /// the axis: 0 = Y, 1 = X, 2 = Z (logs: the top/bottom textures go on the ends)
    Column = 1,
    /// the facing: 0 = automatic, 1..4 = model rotations 0..3 (stairs)
    Facing = 2,
}

impl Rotation {
    pub const ALL: [Rotation; 3] = [Rotation::None, Rotation::Column, Rotation::Facing];
    pub fn label(self) -> &'static str {
        match self {
            Rotation::None => "None",
            Rotation::Column => "Column (axis, like logs)",
            Rotation::Facing => "Facing (like stairs)",
        }
    }
}

/// Face order everywhere: +X, -X, +Y, -Y, +Z, -Z.
pub const FACE_NAMES: [&str; 6] = ["east (+X)", "west (-X)", "top (+Y)", "bottom (-Y)", "south (+Z)", "north (-Z)"];

#[derive(Clone, Debug, PartialEq)]
pub struct BlockDef {
    /// Stable name: the file name, the `B_NAME` define, what graphs, prefabs and saves store.
    pub name: String,
    pub display: String,
    pub category: String,
    pub kind: BlockKind,
    /// Texture name for each face: +X, -X, +Y, -Y, +Z, -Z
    pub faces: [String; 6],
    pub tint: Tint,
    /// Self-illumination of the block's own faces (0..255).
    pub emission: u8,
    /// Colour of the light the block casts on its surroundings.
    pub light_color: [u8; 3],
    /// Reach of that light in blocks (0..15, 0 = no light).
    pub light_level: u8,
    /// Extra light falloff for light passing through this block (water, leaves, ice), 0..15.
    pub light_att: u8,
    pub model: Model,
    pub rotation: Rotation,
    /// Flowing fluid replaces it (plants, pebbles, small mushrooms).
    pub washes_away: bool,
    /// A fluid: its state holds a flow level and its surface sits lower (water).
    pub fluid: bool,
    /// Deleted this session: a tombstone that acts as air, so ids stay stable.
    pub deleted: bool,
}

/// Voxel values: block id in the low 12 bits, state in the high 4 (`shaders/voxel.glsl`).
pub const STATE_SHIFT: u16 = 12;
/// Block ids are 12 bits.
pub const MAX_BLOCKS: usize = 4096;
/// Blocks the generator and the engine fall back to: they can be edited but not deleted.
pub const CORE_BLOCKS: [&str; 9] = ["air", "stone", "dirt", "grass", "snow", "gravel", "water", "lava", "ice"];

pub fn block_id(v: u16) -> u16 {
    v & 0x0FFF
}

/// Block and texture names: lower-case letters, digits and `_`, starting with a letter (they
/// become GLSL identifiers and file names).
pub fn valid_name(s: &str) -> bool {
    let mut ch = s.chars();
    matches!(ch.next(), Some(c) if c.is_ascii_lowercase()) && ch.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') && s.len() <= 48
}

/// "oak_log" -> "Oak Log"
pub fn display_from_name(name: &str) -> String {
    name.split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_ascii_uppercase().to_string() + c.as_str()).unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl BlockDef {
    pub fn new(name: &str, tex: &str) -> Self {
        Self {
            name: name.to_string(),
            display: display_from_name(name),
            category: "Custom".into(),
            kind: BlockKind::Opaque,
            faces: std::array::from_fn(|_| tex.to_string()),
            tint: Tint::None,
            emission: 0,
            light_color: [0; 3],
            light_level: 0,
            light_att: 0,
            model: Model::None,
            rotation: Rotation::None,
            washes_away: false,
            fluid: false,
            deleted: false,
        }
    }

    /// A cube with one texture on top/bottom and another on the sides (logs, basalt, cactus).
    fn looks_like_column(&self) -> bool {
        let f = &self.faces;
        self.kind == BlockKind::Opaque && f[2] == f[3] && f[0] == f[1] && f[0] == f[4] && f[0] == f[5] && f[0] != f[2]
    }

    /// The light the block casts, as the GPU stores it (255 = 15 blocks of reach).
    pub fn gpu_light(&self) -> [u8; 3] {
        self.light_color.map(|c| (c as u32 * self.light_level.min(15) as u32 / 15) as u8)
    }

    pub fn is_core(&self) -> bool {
        CORE_BLOCKS.contains(&self.name.as_str())
    }
}

// ---- built-in defaults ------------------------------------------------------------------------

fn cube(name: &str, display: &str, tex: &str, cat: &str) -> BlockDef {
    BlockDef { display: display.into(), category: cat.into(), ..BlockDef::new(name, tex) }
}

fn column(name: &str, display: &str, top: &str, side: &str, cat: &str) -> BlockDef {
    let s = side.to_string();
    let t = top.to_string();
    BlockDef { faces: [s.clone(), s.clone(), t.clone(), t, s.clone(), s], rotation: Rotation::Column, ..cube(name, display, side, cat) }
}

fn tbs(name: &str, display: &str, top: &str, side: &str, bottom: &str, cat: &str) -> BlockDef {
    let s = side.to_string();
    BlockDef { faces: [s.clone(), s.clone(), top.into(), bottom.into(), s.clone(), s], ..cube(name, display, side, cat) }
}

fn with_kind(mut b: BlockDef, kind: BlockKind) -> BlockDef {
    b.kind = kind;
    b
}

fn glow(mut b: BlockDef, e: u8) -> BlockDef {
    b.emission = e;
    b
}

/// Light source: `rgb` colour with a reach of `level` blocks (0..15).
fn lit(mut b: BlockDef, rgb: [u8; 3], level: u8) -> BlockDef {
    b.light_color = rgb;
    b.light_level = level.min(15);
    b
}

fn attenuate(mut b: BlockDef, att: u8) -> BlockDef {
    b.light_att = att;
    b
}

fn model(name: &str, display: &str, tex: &str, model: Model, cat: &str) -> BlockDef {
    let washes_away = matches!(model, Model::Cross | Model::TallCross | Model::Pebbles | Model::Mushroom);
    let rotation = if model == Model::Stairs { Rotation::Facing } else { Rotation::None };
    BlockDef { kind: BlockKind::Model, model, washes_away, rotation, ..cube(name, display, tex, cat) }
}

/// The built-in block list (the defaults `assets/blocks` starts from). Index = id.
pub fn builtin_defs() -> Vec<BlockDef> {
    use BlockKind::{Air, Cutout, Opaque, Transparent};
    let defs = vec![
        BlockDef { kind: Air, category: String::new(), display: "Air".into(), ..BlockDef::new("air", "stone") },
        // --- terrain ---
        cube("stone", "Stone", "stone", "Terrain"),
        cube("dirt", "Dirt", "dirt", "Terrain"),
        BlockDef { tint: Tint::Grass, ..tbs("grass", "Grass", "grass_top", "grass_side", "dirt", "Terrain") },
        cube("sand", "Sand", "sand", "Terrain"),
        cube("sandstone", "Sandstone", "sandstone", "Terrain"),
        cube("gravel", "Gravel", "gravel", "Terrain"),
        cube("clay", "Clay", "clay", "Terrain"),
        cube("snow", "Snow", "snow", "Terrain"),
        tbs("snowy_grass", "Snowy Grass", "snow", "snowy_grass_side", "dirt", "Terrain"),
        attenuate(with_kind(cube("ice", "Ice", "ice", "Terrain"), Transparent), 1),
        cube("packed_ice", "Packed Ice", "packed_ice", "Terrain"),
        BlockDef { tint: Tint::Water, fluid: true, ..attenuate(with_kind(cube("water", "Water", "water", "Fluids"), Transparent), 2) },
        lit(glow(cube("lava", "Lava", "lava", "Fluids"), 220), [255, 140, 50], 15),
        cube("red_sand", "Red Sand", "red_sand", "Terrain"),
        cube("terracotta", "Terracotta", "terracotta", "Terrain"),
        cube("terracotta_orange", "Orange Terracotta", "terracotta_orange", "Terrain"),
        cube("terracotta_red", "Red Terracotta", "terracotta_red", "Terrain"),
        cube("terracotta_yellow", "Yellow Terracotta", "terracotta_yellow", "Terrain"),
        cube("terracotta_brown", "Brown Terracotta", "terracotta_brown", "Terrain"),
        cube("terracotta_white", "White Terracotta", "terracotta_white", "Terrain"),
        column("basalt", "Basalt", "basalt_top", "basalt_side", "Terrain"),
        cube("ash", "Ash", "ash", "Terrain"),
        lit(glow(cube("magma", "Magma", "magma", "Terrain"), 120), [255, 100, 40], 11),
        cube("obsidian", "Obsidian", "obsidian", "Terrain"),
        cube("deepslate", "Deepslate", "deepslate", "Terrain"),
        cube("granite", "Granite", "granite", "Terrain"),
        cube("andesite", "Andesite", "andesite", "Terrain"),
        cube("calcite", "Calcite", "calcite", "Terrain"),
        cube("moss", "Moss", "moss", "Terrain"),
        tbs("podzol", "Podzol", "podzol_top", "podzol_side", "dirt", "Terrain"),
        BlockDef { tint: Tint::Grass, ..tbs("dry_grass", "Dry Grass", "dry_grass_top", "dry_grass_side", "dirt", "Terrain") },
        cube("coarse_dirt", "Coarse Dirt", "coarse_dirt", "Terrain"),
        cube("mud", "Mud", "mud", "Terrain"),
        // --- ores & crystals ---
        cube("coal_ore", "Coal Ore", "coal_ore", "Ores"),
        cube("iron_ore", "Iron Ore", "iron_ore", "Ores"),
        cube("gold_ore", "Gold Ore", "gold_ore", "Ores"),
        cube("copper_ore", "Copper Ore", "copper_ore", "Ores"),
        lit(glow(cube("crystal", "Crystal", "crystal", "Ores"), 90), [170, 120, 255], 9),
        cube("crystal_shell", "Crystal Shell", "crystal_shell", "Ores"),
        // --- trees ---
        column("oak_log", "Oak Log", "oak_log_top", "oak_log", "Trees"),
        BlockDef { tint: Tint::Foliage, ..with_kind(cube("oak_leaves", "Oak Leaves", "oak_leaves", "Trees"), Cutout) },
        column("birch_log", "Birch Log", "birch_log_top", "birch_log", "Trees"),
        BlockDef { tint: Tint::Foliage, ..with_kind(cube("birch_leaves", "Birch Leaves", "birch_leaves", "Trees"), Cutout) },
        column("spruce_log", "Spruce Log", "spruce_log_top", "spruce_log", "Trees"),
        with_kind(cube("spruce_leaves", "Spruce Leaves", "spruce_leaves", "Trees"), Cutout),
        column("acacia_log", "Acacia Log", "acacia_log_top", "acacia_log", "Trees"),
        with_kind(cube("acacia_leaves", "Acacia Leaves", "acacia_leaves", "Trees"), Cutout),
        column("charred_log", "Charred Log", "charred_log_top", "charred_log", "Trees"),
        column("cactus", "Cactus", "cactus_top", "cactus_side", "Trees"),
        column("palm_log", "Palm Log", "palm_log_top", "palm_log", "Trees"),
        with_kind(cube("palm_leaves", "Palm Leaves", "palm_leaves", "Trees"), Cutout),
        column("mushroom_stem", "Mushroom Stem", "mushroom_stem_top", "mushroom_stem", "Trees"),
        cube("mushroom_cap_red", "Red Mushroom Cap", "mushroom_cap_red", "Trees"),
        lit(glow(cube("mushroom_cap_glow", "Glow Mushroom Cap", "mushroom_cap_glow", "Trees"), 150), [90, 200, 255], 12),
        // --- building ---
        cube("cobblestone", "Cobblestone", "cobblestone", "Building"),
        cube("mossy_cobblestone", "Mossy Cobblestone", "mossy_cobblestone", "Building"),
        cube("planks", "Oak Planks", "planks", "Building"),
        cube("bricks", "Bricks", "bricks", "Building"),
        cube("stone_bricks", "Stone Bricks", "stone_bricks", "Building"),
        with_kind(cube("glass", "Glass", "glass", "Building"), Transparent),
        lit(glow(cube("lamp", "Lamp", "lamp", "Building"), 240), [255, 225, 170], 15),
        // --- models ---
        BlockDef { tint: Tint::Grass, ..model("tall_grass", "Tall Grass", "tall_grass", Model::Cross, "Plants") },
        BlockDef { tint: Tint::Grass, ..model("fern", "Fern", "fern", Model::Cross, "Plants") },
        model("flower_red", "Red Flower", "flower_red", Model::Cross, "Plants"),
        model("flower_yellow", "Yellow Flower", "flower_yellow", Model::Cross, "Plants"),
        model("flower_blue", "Blue Flower", "flower_blue", Model::Cross, "Plants"),
        model("flower_white", "White Flower", "flower_white", Model::Cross, "Plants"),
        model("dead_bush", "Dead Bush", "dead_bush", Model::Cross, "Plants"),
        model("dry_shrub", "Dry Shrub", "dry_shrub", Model::TallCross, "Plants"),
        model("mushroom_red", "Red Mushroom", "mushroom_cap_red", Model::Mushroom, "Plants"),
        lit(glow(model("mushroom_glow", "Glow Mushroom", "mushroom_cap_glow", Model::Mushroom, "Plants"), 120), [90, 200, 255], 10),
        lit(glow(model("crystal_cluster", "Crystal Cluster", "crystal", Model::TallCross, "Plants"), 110), [170, 120, 255], 10),
        model("pebbles", "Pebbles", "cobblestone", Model::Pebbles, "Plants"),
        model("snow_shrub", "Frosted Shrub", "snow_shrub", Model::Cross, "Plants"),
        model("ember_weed", "Ember Weed", "ember_weed", Model::Cross, "Plants"),
        model("stone_slab", "Stone Slab", "stone_bricks", Model::Slab, "Building"),
        model("cobble_slab", "Cobblestone Slab", "cobblestone", Model::Slab, "Building"),
        model("planks_slab", "Plank Slab", "planks", Model::Slab, "Building"),
        model("stone_stairs", "Stone Stairs", "stone_bricks", Model::Stairs, "Building"),
        model("planks_stairs", "Plank Stairs", "planks", Model::Stairs, "Building"),
        lit(glow(model("lantern", "Lantern", "lantern", Model::Lantern, "Building"), 230), [255, 180, 90], 15),
    ];
    // leaves scatter light a little
    defs.into_iter()
        .map(|d| if d.kind == Cutout && d.light_att == 0 { attenuate(d, 1) } else { d })
        .inspect(|d| debug_assert!(d.kind != Opaque || d.looks_like_column() == (d.rotation == Rotation::Column), "{}", d.name))
        .collect()
}

// ---- file format ------------------------------------------------------------------------------

fn is_default<T: Default + PartialEq>(v: &T) -> bool {
    *v == T::default()
}

/// Face textures in a block file: `all`, then `side` (the four sides), `top`/`bottom`, and single
/// faces (`east` +X, `west` -X, `south` +Z, `north` -Z) override what comes before.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FaceFile {
    #[serde(skip_serializing_if = "String::is_empty")]
    pub all: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub top: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub bottom: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub side: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub east: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub west: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub south: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub north: String,
}

impl FaceFile {
    pub fn from_faces(f: &[String; 6]) -> Self {
        if f.iter().all(|t| *t == f[0]) {
            return Self { all: f[0].clone(), ..Default::default() };
        }
        let mut out = Self { top: f[2].clone(), bottom: f[3].clone(), ..Default::default() };
        if f[0] == f[1] && f[0] == f[4] && f[0] == f[5] {
            out.side = f[0].clone();
        } else {
            out.east = f[0].clone();
            out.west = f[1].clone();
            out.south = f[4].clone();
            out.north = f[5].clone();
        }
        out
    }

    pub fn to_faces(&self) -> Result<[String; 6]> {
        let mut f: [String; 6] = std::array::from_fn(|_| self.all.clone());
        let set = |f: &mut [String; 6], i: &[usize], t: &str| {
            if !t.is_empty() {
                for &i in i {
                    f[i] = t.to_string();
                }
            }
        };
        set(&mut f, &[0, 1, 4, 5], &self.side);
        set(&mut f, &[2], &self.top);
        set(&mut f, &[3], &self.bottom);
        set(&mut f, &[0], &self.east);
        set(&mut f, &[1], &self.west);
        set(&mut f, &[4], &self.south);
        set(&mut f, &[5], &self.north);
        if let Some(i) = f.iter().position(|t| t.is_empty()) {
            bail!("no texture for the {} face", FACE_NAMES[i]);
        }
        Ok(f)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LightFile {
    pub color: [u8; 3],
    /// reach in blocks, 0..15
    pub level: u8,
}

fn no_light(l: &LightFile) -> bool {
    l.level == 0
}

/// One `assets/blocks/NAME.ron` file.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BlockFile {
    pub display: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub category: String,
    pub kind: BlockKind,
    #[serde(skip_serializing_if = "is_default")]
    pub model: Model,
    pub textures: FaceFile,
    #[serde(skip_serializing_if = "is_default")]
    pub tint: Tint,
    /// self-glow of the block's faces, 0..255
    #[serde(skip_serializing_if = "is_default")]
    pub emission: u8,
    /// light cast on the surroundings
    #[serde(skip_serializing_if = "no_light")]
    pub light: LightFile,
    /// extra falloff of light passing through, 0..15
    #[serde(skip_serializing_if = "is_default")]
    pub attenuation: u8,
    #[serde(skip_serializing_if = "is_default")]
    pub rotation: Rotation,
    #[serde(skip_serializing_if = "is_default")]
    pub washes_away: bool,
    #[serde(skip_serializing_if = "is_default")]
    pub fluid: bool,
}

impl BlockFile {
    pub fn from_def(d: &BlockDef) -> Self {
        Self {
            display: d.display.clone(),
            category: d.category.clone(),
            kind: d.kind,
            model: if d.kind == BlockKind::Model { d.model } else { Model::None },
            textures: FaceFile::from_faces(&d.faces),
            tint: d.tint,
            emission: d.emission,
            light: LightFile { color: d.light_color, level: d.light_level },
            attenuation: d.light_att,
            rotation: d.rotation,
            washes_away: d.washes_away,
            fluid: d.fluid,
        }
    }

    pub fn to_def(&self, name: &str) -> Result<BlockDef> {
        if self.kind == BlockKind::Model && self.model == Model::None {
            bail!("kind Model needs a `model`");
        }
        Ok(BlockDef {
            name: name.to_string(),
            display: if self.display.is_empty() { display_from_name(name) } else { self.display.clone() },
            category: self.category.clone(),
            kind: self.kind,
            faces: self.textures.to_faces()?,
            tint: self.tint,
            emission: self.emission,
            light_color: self.light.color,
            light_level: self.light.level.min(15),
            light_att: self.attenuation.min(15),
            model: if self.kind == BlockKind::Model { self.model } else { Model::None },
            rotation: self.rotation,
            washes_away: self.washes_away,
            fluid: self.fluid,
            deleted: false,
        })
    }
}

fn ron_config() -> ron::ser::PrettyConfig {
    ron::ser::PrettyConfig::new().depth_limit(3).struct_names(false)
}

pub fn def_to_ron(d: &BlockDef) -> Result<String> {
    Ok(ron::ser::to_string_pretty(&BlockFile::from_def(d), ron_config())? + "\n")
}

pub fn def_from_ron(name: &str, s: &str) -> Result<BlockDef> {
    let f: BlockFile = ron::from_str(s)?;
    f.to_def(name)
}

/// The id order file in the blocks folder.
const ORDER_FILE: &str = "_order.ron";

// ---- GPU data -----------------------------------------------------------------------------------

/// `BlockInfo` in `shaders/voxel.glsl`.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBlockInfo {
    /// 6 x u16 texture layers (the first frame of animated textures)
    pub faces: [u32; 3],
    /// kind | tint << 4 | emission << 8 | model << 16 | fluid << 24 | rotation << 25
    pub flags: u32,
    /// emitted light r | g << 8 | b << 16 | attenuation << 24 | blocks-light << 28
    pub light: u32,
    /// 6 x u16 animation per face: frames - 1 (6 bits) | frames per second * 8 (10 bits); 0 = still
    pub anim: [u32; 3],
}

pub const FLAG_FLUID: u32 = 1 << 24;

/// Where a texture lives in the texture array: `frames` consecutive layers from `base`.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureSlot {
    pub name: String,
    pub base: u16,
    pub frames: u16,
    pub fps: f32,
}

impl TextureSlot {
    /// The animation word of `GpuBlockInfo::anim` (0 for still textures).
    pub fn anim_bits(&self) -> u32 {
        if self.frames <= 1 {
            return 0;
        }
        let frames = (self.frames.min(64) - 1) as u32;
        let fps = (self.fps * 8.0).round().clamp(1.0, 1023.0) as u32;
        frames | (fps << 6)
    }
}

/// What changed since the engine last synced the GPU (`BlockRegistry::take_changes`).
#[derive(Default, Debug)]
pub struct BlockChanges {
    /// the GPU block table must be rewritten
    pub table: bool,
    /// texture layer layout changed: re-upload every layer
    pub layout: bool,
    /// texture layers whose pixels changed
    pub layers: BTreeSet<u16>,
    /// chunks must be regenerated and remeshed (kind, model, light, ...)
    pub remesh: bool,
    /// ids deleted this session (scrub them from the edit overlay)
    pub deleted: Vec<u16>,
}

impl BlockChanges {
    pub fn any(&self) -> bool {
        self.table || self.layout || !self.layers.is_empty() || self.remesh || !self.deleted.is_empty()
    }
}

// ---- registry -----------------------------------------------------------------------------------

pub struct BlockRegistry {
    /// index = id; deleted blocks stay as tombstones (`deleted`, acting as air)
    pub blocks: Vec<BlockDef>,
    pub by_name: HashMap<String, u16>,
    /// Texture layout of the array textures (only textures some block uses).
    pub textures: Vec<TextureSlot>,
    pub texture_index: HashMap<String, usize>,
    /// Texture pixels (procedural built-ins, PNGs, painted).
    pub store: TextureStore,
    /// Where block files are read from and saved to (None: built-in, never saved).
    pub dir: Option<PathBuf>,
    pub changes: BlockChanges,
    /// Blocks changed since the last save.
    unsaved: BTreeSet<String>,
    /// Block files to delete on the next save.
    removed: BTreeSet<String>,
    order_dirty: bool,
    pub autosave: bool,
    last_change: Instant,
    /// Bumped on every change (UI caches).
    pub revision: u64,
    /// Problems found while loading (shown once in the UI).
    pub load_warnings: Vec<String>,
}

impl BlockRegistry {
    /// The built-in blocks with procedural textures, nothing on disk (tests, tools).
    pub fn new() -> Self {
        Self::from_defs(builtin_defs(), TextureStore::new(None), None)
    }

    fn from_defs(blocks: Vec<BlockDef>, store: TextureStore, dir: Option<PathBuf>) -> Self {
        let mut r = Self {
            blocks,
            by_name: HashMap::new(),
            textures: Vec::new(),
            texture_index: HashMap::new(),
            store,
            dir,
            changes: BlockChanges::default(),
            unsaved: BTreeSet::new(),
            removed: BTreeSet::new(),
            order_dirty: false,
            autosave: true,
            last_change: Instant::now(),
            revision: 0,
            load_warnings: Vec::new(),
        };
        r.reindex();
        r.relayout();
        r.changes = BlockChanges::default();
        r
    }

    /// Load `dir/*.ron` (textures from `tex_dir`). An empty or missing folder gets the built-in
    /// defaults written into it.
    pub fn load(dir: &Path, tex_dir: &Path) -> Result<Self> {
        let store = TextureStore::new(Some(tex_dir.to_path_buf()));
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "ron")).collect())
            .unwrap_or_default();
        files.retain(|p| p.file_name().is_some_and(|n| n != ORDER_FILE));
        if files.is_empty() {
            log::info!("no block files in {}: writing the built-in blocks there", dir.display());
            let mut r = Self::from_defs(builtin_defs(), store, Some(dir.to_path_buf()));
            r.unsaved = r.blocks.iter().skip(1).map(|b| b.name.clone()).collect();
            r.order_dirty = true;
            if let Err(e) = r.save() {
                log::warn!("could not write the default blocks: {e:#}");
            }
            return Ok(r);
        }
        let mut warnings = Vec::new();
        let mut defs: HashMap<String, BlockDef> = HashMap::new();
        for p in &files {
            let name = p.file_stem().unwrap_or_default().to_string_lossy().to_string();
            if name == "air" {
                continue;
            }
            if !valid_name(&name) {
                warnings.push(format!("{}: not a valid block name (a-z, 0-9, _)", p.display()));
                continue;
            }
            match std::fs::read_to_string(p).map_err(anyhow::Error::from).and_then(|s| def_from_ron(&name, &s)) {
                Ok(d) => {
                    defs.insert(name, d);
                }
                Err(e) => warnings.push(format!("{}: {e:#}", p.display())),
            }
        }
        // ids: the order file first, then the rest by name
        let order: Vec<String> = std::fs::read_to_string(dir.join(ORDER_FILE)).ok().and_then(|s| ron::from_str(&s).ok()).unwrap_or_default();
        let mut blocks = vec![builtin_defs().swap_remove(0)];
        for n in &order {
            if let Some(d) = defs.remove(n) {
                blocks.push(d);
            }
        }
        let mut rest: Vec<BlockDef> = defs.into_values().collect();
        rest.sort_by(|a, b| a.name.cmp(&b.name));
        let order_dirty = !rest.is_empty() || order.len() + 1 != blocks.len();
        blocks.extend(rest);
        if blocks.len() > MAX_BLOCKS {
            warnings.push(format!("{} blocks: only the first {MAX_BLOCKS} are used", blocks.len()));
            blocks.truncate(MAX_BLOCKS);
        }
        // the generator and the engine rely on these
        for core in CORE_BLOCKS {
            if !blocks.iter().any(|b| b.name == core)
                && let Some(d) = builtin_defs().into_iter().find(|d| d.name == core)
            {
                warnings.push(format!("core block `{core}` missing: using the built-in one"));
                blocks.push(d);
            }
        }
        for w in &warnings {
            log::warn!("blocks: {w}");
        }
        let mut r = Self::from_defs(blocks, store, Some(dir.to_path_buf()));
        r.order_dirty = order_dirty;
        r.load_warnings = warnings;
        log::info!("loaded {} blocks and {} textures from {}", r.blocks.len() - 1, r.textures.len(), dir.display());
        Ok(r)
    }

    fn reindex(&mut self) {
        self.by_name = self.blocks.iter().enumerate().filter(|(_, d)| !d.deleted).map(|(i, d)| (d.name.clone(), i as u16)).collect();
    }

    /// Recompute the texture layer layout from the faces of every block. Returns whether it
    /// changed (then every layer must be uploaded again).
    fn relayout(&mut self) -> bool {
        let mut slots = Vec::new();
        let mut index = HashMap::new();
        let mut base = 0u32;
        for d in self.blocks.iter().filter(|d| !d.deleted) {
            for t in &d.faces {
                if index.contains_key(t) {
                    continue;
                }
                let a = self.store.get(t);
                let frames = a.frames.len().clamp(1, 64) as u16;
                index.insert(t.clone(), slots.len());
                slots.push(TextureSlot { name: t.clone(), base: base as u16, frames, fps: a.fps });
                base += frames as u32;
            }
        }
        let layout_changed = slots.iter().map(|s| (&s.name, s.base, s.frames)).ne(self.textures.iter().map(|s| (&s.name, s.base, s.frames)));
        let fps_changed = slots.iter().map(|s| s.fps).ne(self.textures.iter().map(|s| s.fps));
        self.textures = slots;
        self.texture_index = index;
        if layout_changed {
            self.changes.layout = true;
        }
        if layout_changed || fps_changed {
            self.changes.table = true;
        }
        layout_changed
    }

    /// Total texture array layers.
    pub fn layer_count(&self) -> u32 {
        self.textures.last().map(|s| s.base as u32 + s.frames as u32).unwrap_or(0)
    }

    /// Pixels of every layer, in layer order: (layer, image).
    pub fn layer_images(&mut self) -> Vec<(u16, crate::textures::TexImage)> {
        let mut out = Vec::new();
        for s in self.textures.clone() {
            let a = self.store.get(&s.name);
            for f in 0..s.frames {
                out.push((s.base + f, a.frames[(f as usize).min(a.frames.len() - 1)].clone()));
            }
        }
        out
    }

    /// The image of one layer.
    pub fn layer_image(&mut self, layer: u16) -> Option<crate::textures::TexImage> {
        let s = self.textures.iter().find(|s| layer >= s.base && layer < s.base + s.frames)?.clone();
        let a = self.store.get(&s.name);
        a.frames.get((layer - s.base) as usize).cloned()
    }

    pub fn id(&self, name: &str) -> u16 {
        *self.by_name.get(name).unwrap_or_else(|| panic!("unknown block {name}"))
    }

    pub fn try_id(&self, name: &str) -> Option<u16> {
        self.by_name.get(name).copied()
    }

    /// The definition of a voxel value (the state bits are ignored; unknown ids give air).
    pub fn get(&self, v: u16) -> &BlockDef {
        self.blocks.get(block_id(v) as usize).unwrap_or(&self.blocks[0])
    }

    /// Live blocks (not air, not deleted) with their ids.
    pub fn live(&self) -> impl Iterator<Item = (u16, &BlockDef)> {
        self.blocks.iter().enumerate().skip(1).filter(|(_, d)| !d.deleted).map(|(i, d)| (i as u16, d))
    }

    /// name -> id of every live block (the world generator compiles block names to ids).
    pub fn id_map(&self) -> HashMap<String, u32> {
        self.by_name.iter().map(|(n, i)| (n.clone(), *i as u32)).collect()
    }

    /// `v` with the state it gets when placed against a face with `normal`, the camera looking
    /// along `forward`: column blocks (logs) take the face's axis, stairs face away from the camera.
    pub fn oriented(&self, v: u16, normal: glam::IVec3, forward: glam::Vec3) -> u16 {
        let state = match self.get(v).rotation {
            Rotation::Column => {
                if normal.x != 0 {
                    1
                } else if normal.z != 0 {
                    2
                } else {
                    0
                }
            }
            Rotation::Facing => {
                // facing 1..4 = rotations 0..3 of the stairs model (back step towards +Z, -X, -Z, +X)
                if forward.z.abs() >= forward.x.abs() {
                    if forward.z >= 0.0 { 1 } else { 3 }
                } else if forward.x < 0.0 {
                    2
                } else {
                    4
                }
            }
            Rotation::None => return v,
        };
        block_id(v) | (state << STATE_SHIFT)
    }

    /// `v` turned `steps` quarter turns around Y, +X towards +Z (as `Prefab::rotated` turns
    /// positions; `rotate_value` in the generated block header is the GPU copy).
    pub fn rotated_y(&self, v: u16, steps: u32) -> u16 {
        let st = v >> STATE_SHIFT;
        let st = match self.get(v).rotation {
            Rotation::Column if steps % 2 == 1 && (st == 1 || st == 2) => 3 - st,
            Rotation::Facing if st != 0 => ((st - 1 + steps as u16) & 3) + 1,
            _ => return v,
        };
        block_id(v) | (st << STATE_SHIFT)
    }

    /// `name`, or `name:state` for a block with a state (prefab files, test scripts).
    pub fn value_name(&self, v: u16) -> String {
        match v >> STATE_SHIFT {
            0 => self.get(v).name.clone(),
            st => format!("{}:{st}", self.get(v).name),
        }
    }

    /// The inverse of `value_name`.
    pub fn parse_value(&self, s: &str) -> Option<u16> {
        let (name, state) = s.split_once(':').unwrap_or((s, "0"));
        let state = state.parse::<u16>().ok().filter(|&st| st < 16)?;
        Some(self.try_id(name)? | (state << STATE_SHIFT))
    }

    /// The GPU block table, `MAX_BLOCKS` entries.
    pub fn gpu_table(&self) -> Vec<GpuBlockInfo> {
        let mut out = vec![GpuBlockInfo::default(); MAX_BLOCKS];
        for (o, d) in out.iter_mut().zip(&self.blocks) {
            if d.deleted || d.kind == BlockKind::Air {
                continue;
            }
            let slot = |i: usize| &self.textures[self.texture_index[&d.faces[i]]];
            let layer = |i: usize| slot(i).base as u32;
            let anim = |i: usize| slot(i).anim_bits();
            let l = d.gpu_light();
            *o = GpuBlockInfo {
                faces: [layer(0) | (layer(1) << 16), layer(2) | (layer(3) << 16), layer(4) | (layer(5) << 16)],
                flags: d.kind as u32
                    | ((d.tint as u32) << 4)
                    | ((d.emission as u32) << 8)
                    | (((d.model as u32) & 0xFF) << 16)
                    | if d.fluid { FLAG_FLUID } else { 0 }
                    | ((d.rotation as u32) << 25),
                light: l[0] as u32
                    | ((l[1] as u32) << 8)
                    | ((l[2] as u32) << 16)
                    | ((d.light_att.min(15) as u32) << 24)
                    | (((d.kind == BlockKind::Opaque) as u32) << 28),
                anim: [anim(0) | (anim(1) << 16), anim(2) | (anim(3) << 16), anim(4) | (anim(5) << 16)],
            };
        }
        out
    }

    /// GLSL for the world generator: the ids of the core blocks the compiler falls back to
    /// (`#define B_STONE 1u` etc.). Kinds and rotations are read from the GPU block table, and graphs
    /// compile block names to ids, so block edits don't change the generator's source (only a
    /// graph that names a block does). Core blocks can't be deleted, so this stays the same for a
    /// session.
    pub fn glsl_header(&self) -> String {
        let mut s = String::from("// ---- core blocks (blocks.rs) ----\n");
        for name in CORE_BLOCKS {
            s.push_str(&format!("#define B_{} {}u\n", name.to_uppercase(), self.try_id(name).unwrap_or(0)));
        }
        s
    }

    // ---- editing ------------------------------------------------------------------------------

    fn touched(&mut self, name: &str) {
        self.unsaved.insert(name.to_string());
        self.last_change = Instant::now();
        self.revision += 1;
    }

    /// Unsaved block or texture changes.
    pub fn has_unsaved(&self) -> bool {
        !self.unsaved.is_empty() || !self.removed.is_empty() || self.order_dirty || self.store.has_dirty()
    }

    pub fn take_changes(&mut self) -> BlockChanges {
        std::mem::take(&mut self.changes)
    }

    /// Create a block named `name`, a copy of `from` (or a plain stone-textured cube).
    pub fn create(&mut self, name: &str, from: Option<u16>) -> Result<u16> {
        if !valid_name(name) {
            bail!("`{name}` is not a valid name (a-z, 0-9 and _, starting with a letter)");
        }
        if self.by_name.contains_key(name) || name == "air" {
            bail!("a block named `{name}` exists already");
        }
        if self.blocks.len() >= MAX_BLOCKS {
            bail!("no free block ids ({MAX_BLOCKS} max)");
        }
        let mut d = match from {
            Some(id) if !self.get(id).deleted && id != 0 => {
                let src = self.get(id).clone();
                BlockDef { name: name.to_string(), display: display_from_name(name), ..src }
            }
            _ => BlockDef::new(name, "stone"),
        };
        if d.category.is_empty() {
            d.category = "Custom".into();
        }
        // a name freed by a deletion this session may come back: it gets a new id
        self.removed.remove(name);
        let id = self.blocks.len() as u16;
        self.blocks.push(d);
        self.reindex();
        self.relayout();
        // nothing in the world uses it yet: no regeneration
        self.changes.table = true;
        self.order_dirty = true;
        self.touched(name);
        Ok(id)
    }

    /// Replace a block's definition. Classifies the change: render-only properties (textures,
    /// tint, emission) just rewrite the GPU table; the rest also regenerates the world.
    pub fn update(&mut self, id: u16, mut def: BlockDef) -> Result<()> {
        let Some(old) = self.blocks.get(id as usize).cloned() else { bail!("no block with id {id}") };
        if id == 0 || old.deleted {
            bail!("block {id} can't be edited");
        }
        def.name = old.name.clone();
        def.deleted = false;
        if def.kind == BlockKind::Model && def.model == Model::None {
            def.model = Model::Cross;
        }
        if def.kind != BlockKind::Model {
            def.model = Model::None;
        }
        if def == old {
            return Ok(());
        }
        for t in &def.faces {
            if !valid_name(t) {
                bail!("`{t}` is not a valid texture name");
            }
        }
        let render_only = BlockDef { faces: old.faces.clone(), tint: old.tint, emission: old.emission, display: old.display.clone(), category: old.category.clone(), ..def.clone() } == old;
        let faces_changed = def.faces != old.faces;
        self.blocks[id as usize] = def;
        if faces_changed {
            self.relayout();
        }
        self.changes.table = true;
        if !render_only {
            self.changes.remesh = true;
        }
        let name = old.name.clone();
        self.touched(&name);
        Ok(())
    }

    /// Delete a block (core blocks can't be deleted). Its id becomes a tombstone for the rest of
    /// the session; blocks of this kind in the world turn into air.
    pub fn delete(&mut self, id: u16) -> Result<()> {
        let d = self.blocks.get(id as usize).cloned().context("no such block")?;
        if d.deleted || id == 0 {
            bail!("block {id} is already gone");
        }
        if d.is_core() {
            bail!("`{}` is a core block (the generator falls back to it): edit it instead", d.name);
        }
        self.blocks[id as usize] = BlockDef { deleted: true, kind: BlockKind::Air, ..d.clone() };
        self.reindex();
        self.relayout();
        self.unsaved.remove(&d.name);
        self.removed.insert(d.name.clone());
        self.order_dirty = true;
        self.changes.table = true;
        self.changes.remesh = true;
        self.changes.deleted.push(id);
        self.last_change = Instant::now();
        self.revision += 1;
        Ok(())
    }

    /// Replace the frames (and speed) of a texture, e.g. from the texture editor. Same frame
    /// count: only those layers are uploaded again; otherwise the layout changes.
    pub fn set_texture(&mut self, name: &str, asset: TexAsset) {
        let old_frames = self.store.get(name).frames.len();
        let fps_changed = self.store.get(name).fps != asset.fps;
        let changed: Vec<usize> = {
            let old = self.store.get(name);
            (0..asset.frames.len()).filter(|&i| old.frames.get(i).map(|f| f.color != asset.frames[i].color || f.height != asset.frames[i].height).unwrap_or(true)).collect()
        };
        let frames = asset.frames.len();
        self.store.insert(name, asset);
        self.last_change = Instant::now();
        self.revision += 1;
        if frames != old_frames {
            self.relayout();
            return;
        }
        if let Some(&i) = self.texture_index.get(name) {
            let base = self.textures[i].base;
            for f in changed {
                self.changes.layers.insert(base + f as u16);
            }
            if fps_changed {
                self.textures[i].fps = self.store.get(name).fps;
                self.changes.table = true;
            }
        }
    }

    /// Every texture name the editors can offer: built-ins, files and textures made this session.
    pub fn all_texture_names(&self) -> Vec<String> {
        let mut v: BTreeSet<String> = builtin_defs().iter().flat_map(|d| d.faces.clone()).collect();
        v.extend(self.store.known_names());
        v.extend(self.textures.iter().map(|s| s.name.clone()));
        v.into_iter().collect()
    }

    /// Write changed block files, the order file and changed textures.
    pub fn save(&mut self) -> Result<()> {
        let Some(dir) = self.dir.clone() else {
            self.unsaved.clear();
            self.removed.clear();
            self.order_dirty = false;
            return Ok(());
        };
        std::fs::create_dir_all(&dir)?;
        for name in std::mem::take(&mut self.removed) {
            let p = dir.join(format!("{name}.ron"));
            if p.exists() {
                std::fs::remove_file(&p).with_context(|| format!("removing {}", p.display()))?;
            }
        }
        for name in std::mem::take(&mut self.unsaved) {
            let Some(&id) = self.by_name.get(&name) else { continue };
            let p = dir.join(format!("{name}.ron"));
            std::fs::write(&p, def_to_ron(&self.blocks[id as usize])?).with_context(|| format!("writing {}", p.display()))?;
        }
        if std::mem::take(&mut self.order_dirty) {
            let order: Vec<&str> = self.blocks.iter().skip(1).filter(|d| !d.deleted).map(|d| d.name.as_str()).collect();
            let s = ron::ser::to_string_pretty(&order, ron::ser::PrettyConfig::new())?;
            std::fs::write(dir.join(ORDER_FILE), s + "\n")?;
        }
        self.store.save_dirty()?;
        Ok(())
    }

    /// Save 2 s after the last change (block editor, texture editor).
    pub fn autosave_tick(&mut self) -> Option<Result<()>> {
        if self.autosave && self.has_unsaved() && self.last_change.elapsed().as_secs_f32() > 2.0 {
            return Some(self.save());
        }
        None
    }

    /// Write the built-in blocks as files (`--export-block-defaults`).
    pub fn export_builtin(dir: &Path) -> Result<usize> {
        std::fs::create_dir_all(dir)?;
        let defs = builtin_defs();
        for d in defs.iter().skip(1) {
            std::fs::write(dir.join(format!("{}.ron", d.name)), def_to_ron(d)?)?;
        }
        let order: Vec<&str> = defs.iter().skip(1).map(|d| d.name.as_str()).collect();
        std::fs::write(dir.join(ORDER_FILE), ron::ser::to_string_pretty(&order, ron::ser::PrettyConfig::new())? + "\n")?;
        Ok(defs.len() - 1)
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{IVec3, Vec3};

    #[test]
    fn placing_orients_logs_and_stairs() {
        let r = BlockRegistry::new();
        let log = r.id("oak_log");
        // logs take the axis of the face they're placed against
        assert_eq!(r.oriented(log, IVec3::Y, Vec3::Z), log);
        assert_eq!(r.oriented(log, IVec3::X, Vec3::Z), log | (1 << STATE_SHIFT));
        assert_eq!(r.oriented(log, -IVec3::Z, Vec3::Z), log | (2 << STATE_SHIFT));
        // stairs face away from the camera
        let st = r.id("stone_stairs");
        assert_eq!(r.oriented(st, IVec3::Y, Vec3::new(0.1, -0.5, 0.9)) >> STATE_SHIFT, 1);
        assert_eq!(r.oriented(st, IVec3::Y, Vec3::new(-0.9, -0.5, 0.1)) >> STATE_SHIFT, 2);
        assert_eq!(r.oriented(st, IVec3::Y, Vec3::new(0.0, -0.5, -1.0)) >> STATE_SHIFT, 3);
        assert_eq!(r.oriented(st, IVec3::Y, Vec3::new(1.0, -0.5, 0.0)) >> STATE_SHIFT, 4);
        // other blocks are placed as they are, and lookups ignore the state
        let stone = r.id("stone");
        assert_eq!(r.oriented(stone, IVec3::X, Vec3::Z), stone);
        assert_eq!(r.get(log | (2 << STATE_SHIFT)).name, "oak_log");
        assert!(r.get(log).rotation == Rotation::Column && r.get(stone).rotation == Rotation::None);
    }

    #[test]
    fn rotating_turns_states() {
        let r = BlockRegistry::new();
        let (log, st, stone) = (r.id("oak_log"), r.id("stone_stairs"), r.id("stone"));
        let s = |v: u16, k: u16| v | (k << STATE_SHIFT);
        assert_eq!(r.rotated_y(s(log, 1), 1), s(log, 2));
        assert_eq!(r.rotated_y(s(log, 2), 3), s(log, 1));
        assert_eq!(r.rotated_y(s(log, 1), 2), s(log, 1));
        assert_eq!(r.rotated_y(log, 1), log);
        // stairs: back step +Z -> -X -> -Z -> +X
        assert_eq!(r.rotated_y(s(st, 1), 1), s(st, 2));
        assert_eq!(r.rotated_y(s(st, 4), 1), s(st, 1));
        assert_eq!(r.rotated_y(s(st, 2), 3), s(st, 1));
        assert_eq!(r.rotated_y(st, 1), st); // automatic stays automatic
        assert_eq!(r.rotated_y(stone, 1), stone);
        for v in [stone, s(log, 2), s(st, 3)] {
            assert_eq!(r.parse_value(&r.value_name(v)), Some(v));
        }
        assert_eq!(r.parse_value("oak_log:16"), None);
    }

    #[test]
    fn block_files_round_trip() {
        for d in builtin_defs().iter().skip(1) {
            let s = def_to_ron(d).unwrap();
            let back = def_from_ron(&d.name, &s).unwrap_or_else(|e| panic!("{}: {e:#}\n{s}", d.name));
            assert_eq!(&back, d, "{}:\n{s}", d.name);
        }
        // short forms and overrides
        let d = def_from_ron("x", "(kind: Opaque, textures: (all: \"stone\", top: \"grass_top\", north: \"dirt\"))").unwrap();
        assert_eq!(d.faces, ["stone", "stone", "grass_top", "stone", "stone", "dirt"].map(String::from));
        assert_eq!(d.display, "X");
        assert!(def_from_ron("x", "(kind: Opaque, textures: (top: \"a\"))").is_err(), "missing faces");
        assert!(def_from_ron("x", "(kind: Opaque, textures: (all: \"a\"), colour: 3)").is_err(), "typos are errors");
        assert!(def_from_ron("x", "(kind: Model, textures: (all: \"a\"))").is_err(), "a model needs a model");
    }

    #[test]
    fn names_and_ids_survive_a_save_and_load() {
        let dir = std::env::temp_dir().join(format!("ve_blocks_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let tex = dir.join("textures");
        // an empty folder gets the defaults, with the built-in ids
        let mut r = BlockRegistry::load(&dir.join("blocks"), &tex).unwrap();
        let builtin = BlockRegistry::new();
        for (i, d) in builtin.blocks.iter().enumerate() {
            assert_eq!(r.blocks[i], *d, "id {i}");
        }
        // create, edit and delete, then save and load again
        let lamp = r.id("lamp");
        let id = r.create("blue_lamp", Some(lamp)).unwrap();
        let mut d = r.blocks[id as usize].clone();
        d.light_color = [80, 120, 255];
        d.display = "Blue Lamp".into();
        r.update(id, d).unwrap();
        let clay = r.id("clay");
        r.delete(clay).unwrap();
        assert!(r.delete(r.id("water")).is_err(), "core blocks stay");
        assert!(r.create("blue_lamp", None).is_err() && r.create("Bad Name", None).is_err());
        assert_eq!(r.try_id("clay"), None);
        assert_eq!(r.get(clay).kind, BlockKind::Air, "tombstone");
        r.save().unwrap();
        let r2 = BlockRegistry::load(&dir.join("blocks"), &tex).unwrap();
        assert_eq!(r2.try_id("clay"), None);
        let b = r2.get(r2.id("blue_lamp"));
        assert_eq!((b.light_color, b.display.as_str(), b.light_level), ([80, 120, 255], "Blue Lamp", 15));
        // ids stay in order: everything before clay keeps its id, the rest moves up one
        for n in ["stone", "sand", "gravel"] {
            assert_eq!(r2.id(n), builtin.id(n), "{n}");
        }
        for n in ["snow", "water", "lantern"] {
            assert_eq!(r2.id(n), builtin.id(n) - 1, "{n}");
        }
        assert!(r2.id("blue_lamp") > r2.id("lantern"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gpu_table_and_header() {
        let r = BlockRegistry::new();
        let t = r.gpu_table();
        assert_eq!(t.len(), MAX_BLOCKS);
        assert_eq!(std::mem::size_of::<GpuBlockInfo>(), 32);
        let water = r.id("water") as usize;
        assert_eq!(t[water].flags & 15, BlockKind::Transparent as u32);
        assert_ne!(t[water].flags & FLAG_FLUID, 0);
        // lava is animated: its faces carry frames and speed
        let lava = r.get(r.id("lava"));
        let slot = &r.textures[r.texture_index[&lava.faces[2]]];
        assert!(slot.frames > 1 && t[r.id("lava") as usize].anim[1] & 0xFFFF == slot.anim_bits());
        let h = r.glsl_header();
        assert!(h.contains("#define B_STONE 1u") && h.contains("#define B_WATER 12u"));
        // rotations are in the flags (the generator turns prefab blocks with them)
        assert_eq!((t[r.id("oak_log") as usize].flags >> 25) & 3, 1);
        assert_eq!((t[r.id("stone_stairs") as usize].flags >> 25) & 3, 2);
    }
}

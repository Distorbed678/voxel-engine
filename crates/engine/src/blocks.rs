//! Block registry: ids, render kinds, per-face textures. Also emits the GLSL id header used by
//! the world generator and the GPU block table used by the mesher/renderer.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum BlockKind {
    Air = 0,
    Opaque = 1,
    Cutout = 2,
    Transparent = 3,
    Model = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Tint {
    None = 0,
    Grass = 1,
    Foliage = 2,
    Water = 3,
}

#[derive(Clone, Debug)]
pub struct BlockDef {
    pub name: &'static str,
    pub display: &'static str,
    pub kind: BlockKind,
    /// Texture name for each face: +X, -X, +Y, -Y, +Z, -Z
    pub faces: [&'static str; 6],
    /// Self-illumination of the block's own faces (0..255).
    pub emission: u8,
    /// Light the block casts on its surroundings (RGB, 255 = 15-block reach). See `lit`.
    pub light: [u8; 3],
    /// Extra light falloff for light passing through this block (water, leaves, ice).
    pub light_att: u8,
    pub tint: Tint,
    pub model: u16,
    pub category: &'static str,
}

/// Voxel values: block id in the low 12 bits, state in the high 4 (`shaders/voxel.glsl`).
pub const STATE_SHIFT: u16 = 12;

pub fn block_id(v: u16) -> u16 {
    v & 0x0FFF
}

/// What a block's state means, which decides how it turns with a rotated selection or prefab.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rotation {
    /// no state
    None = 0,
    /// the axis: 0 = Y, 1 = X, 2 = Z
    Column = 1,
    /// the facing: 0 = automatic, 1..4 = model rotations 0..3
    Facing = 2,
}

impl BlockDef {
    /// A cube with one texture on top/bottom and another on the sides (logs, basalt, cactus):
    /// it can lie along X or Z.
    pub fn is_column(&self) -> bool {
        let f = &self.faces;
        self.kind == BlockKind::Opaque && f[2] == f[3] && f[0] == f[1] && f[0] == f[4] && f[0] == f[5] && f[0] != f[2]
    }

    /// Flowing water replaces it (plants, pebbles, small mushrooms).
    pub fn washes_away(&self) -> bool {
        self.kind == BlockKind::Model && matches!(self.model, models::CROSS | models::TALL_CROSS | models::PEBBLES | models::MUSHROOM)
    }

    pub fn rotation(&self) -> Rotation {
        if self.is_column() {
            Rotation::Column
        } else if self.kind == BlockKind::Model && self.model == models::STAIRS {
            Rotation::Facing
        } else {
            Rotation::None
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBlockInfo {
    pub faces: [u32; 3],
    pub flags: u32,
    /// emitted light r | g << 8 | b << 16 | attenuation << 24 | blocks-light << 28
    pub light: u32,
}

pub struct BlockRegistry {
    pub blocks: Vec<BlockDef>,
    pub by_name: HashMap<&'static str, u16>,
    /// Texture layer names (index = array layer)
    pub textures: Vec<&'static str>,
    pub texture_index: HashMap<&'static str, u16>,
}

fn cube(name: &'static str, display: &'static str, tex: &'static str, cat: &'static str) -> BlockDef {
    BlockDef {
        name,
        display,
        kind: BlockKind::Opaque,
        faces: [tex; 6],
        emission: 0,
        light: [0; 3],
        light_att: 0,
        tint: Tint::None,
        model: 0,
        category: cat,
    }
}

fn column(name: &'static str, display: &'static str, top: &'static str, side: &'static str, cat: &'static str) -> BlockDef {
    BlockDef { faces: [side, side, top, top, side, side], ..cube(name, display, side, cat) }
}

fn tbs(name: &'static str, display: &'static str, top: &'static str, side: &'static str, bottom: &'static str, cat: &'static str) -> BlockDef {
    BlockDef { faces: [side, side, top, bottom, side, side], ..cube(name, display, side, cat) }
}

fn with_kind(mut b: BlockDef, kind: BlockKind) -> BlockDef {
    b.kind = kind;
    b
}

fn glow(mut b: BlockDef, e: u8) -> BlockDef {
    b.emission = e;
    b
}

/// Light source: `rgb` colour at strength `level` (blocks of reach, 0..15).
fn lit(mut b: BlockDef, rgb: [u8; 3], level: u32) -> BlockDef {
    b.light = rgb.map(|c| (c as u32 * level.min(15) / 15) as u8);
    b
}

fn attenuate(mut b: BlockDef, att: u8) -> BlockDef {
    b.light_att = att;
    b
}

fn model(name: &'static str, display: &'static str, tex: &'static str, model: u16, cat: &'static str) -> BlockDef {
    BlockDef { kind: BlockKind::Model, model, ..cube(name, display, tex, cat) }
}

/// Model ids understood by the model mesh shader (shaders/render/models.glsl).
pub mod models {
    pub const CROSS: u16 = 1; // two crossed quads (plants)
    pub const SLAB: u16 = 2; // bottom half block
    pub const STAIRS: u16 = 3; // stairs facing -Z
    pub const LANTERN: u16 = 4; // small hanging box
    pub const PEBBLES: u16 = 5; // small flat stones
    pub const MUSHROOM: u16 = 6; // stem + cap
    pub const TALL_CROSS: u16 = 7; // crossed quads, slightly taller/wider
}

impl BlockRegistry {
    pub fn new() -> Self {
        use BlockKind::*;
        let defs = vec![
            BlockDef { kind: Air, ..cube("air", "Air", "stone", "") },
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
            BlockDef { tint: Tint::Water, ..attenuate(with_kind(cube("water", "Water", "water", "Fluids"), Transparent), 2) },
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
            BlockDef { tint: Tint::Grass, ..model("tall_grass", "Tall Grass", "tall_grass", models::CROSS, "Plants") },
            BlockDef { tint: Tint::Grass, ..model("fern", "Fern", "fern", models::CROSS, "Plants") },
            model("flower_red", "Red Flower", "flower_red", models::CROSS, "Plants"),
            model("flower_yellow", "Yellow Flower", "flower_yellow", models::CROSS, "Plants"),
            model("flower_blue", "Blue Flower", "flower_blue", models::CROSS, "Plants"),
            model("flower_white", "White Flower", "flower_white", models::CROSS, "Plants"),
            model("dead_bush", "Dead Bush", "dead_bush", models::CROSS, "Plants"),
            model("dry_shrub", "Dry Shrub", "dry_shrub", models::TALL_CROSS, "Plants"),
            model("mushroom_red", "Red Mushroom", "mushroom_cap_red", models::MUSHROOM, "Plants"),
            lit(glow(model("mushroom_glow", "Glow Mushroom", "mushroom_cap_glow", models::MUSHROOM, "Plants"), 120), [90, 200, 255], 10),
            lit(glow(model("crystal_cluster", "Crystal Cluster", "crystal", models::TALL_CROSS, "Plants"), 110), [170, 120, 255], 10),
            model("pebbles", "Pebbles", "cobblestone", models::PEBBLES, "Plants"),
            model("snow_shrub", "Frosted Shrub", "snow_shrub", models::CROSS, "Plants"),
            model("ember_weed", "Ember Weed", "ember_weed", models::CROSS, "Plants"),
            model("stone_slab", "Stone Slab", "stone_bricks", models::SLAB, "Building"),
            model("cobble_slab", "Cobblestone Slab", "cobblestone", models::SLAB, "Building"),
            model("planks_slab", "Plank Slab", "planks", models::SLAB, "Building"),
            model("stone_stairs", "Stone Stairs", "stone_bricks", models::STAIRS, "Building"),
            model("planks_stairs", "Plank Stairs", "planks", models::STAIRS, "Building"),
            lit(glow(model("lantern", "Lantern", "lantern", models::LANTERN, "Building"), 230), [255, 180, 90], 15),
        ];
        // leaves scatter light a little
        let defs: Vec<BlockDef> =
            defs.into_iter().map(|d| if d.kind == Cutout && d.light_att == 0 { attenuate(d, 1) } else { d }).collect();
        let mut textures = Vec::new();
        let mut texture_index = HashMap::new();
        for d in &defs {
            for t in d.faces {
                if !texture_index.contains_key(t) {
                    texture_index.insert(t, textures.len() as u16);
                    textures.push(t);
                }
            }
        }
        let by_name = defs.iter().enumerate().map(|(i, d)| (d.name, i as u16)).collect();
        Self { blocks: defs, by_name, textures, texture_index }
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

    /// `v` with the state it gets when placed against a face with `normal`, the camera looking
    /// along `forward`: column blocks (logs) take the face's axis, stairs face away from the camera.
    pub fn oriented(&self, v: u16, normal: glam::IVec3, forward: glam::Vec3) -> u16 {
        let state = match self.get(v).rotation() {
            Rotation::Column => {
                if normal.x != 0 { 1 } else if normal.z != 0 { 2 } else { 0 }
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
    /// positions; `rotate_value` in `shaders/worldgen/prelude.glsl` is the GPU copy).
    pub fn rotated_y(&self, v: u16, steps: u32) -> u16 {
        let st = v >> STATE_SHIFT;
        let st = match self.get(v).rotation() {
            Rotation::Column if steps % 2 == 1 && (st == 1 || st == 2) => 3 - st,
            Rotation::Facing if st != 0 => ((st - 1 + steps as u16) & 3) + 1,
            _ => return v,
        };
        block_id(v) | (st << STATE_SHIFT)
    }

    /// `name`, or `name:state` for a block with a state (prefab files, test scripts).
    pub fn value_name(&self, v: u16) -> String {
        match v >> STATE_SHIFT {
            0 => self.get(v).name.to_string(),
            st => format!("{}:{st}", self.get(v).name),
        }
    }

    /// The inverse of `value_name`.
    pub fn parse_value(&self, s: &str) -> Option<u16> {
        let (name, state) = s.split_once(':').unwrap_or((s, "0"));
        let state = state.parse::<u16>().ok().filter(|&st| st < 16)?;
        Some(self.try_id(name)? | (state << STATE_SHIFT))
    }

    pub fn gpu_table(&self) -> Vec<GpuBlockInfo> {
        self.blocks
            .iter()
            .map(|d| {
                let l: Vec<u32> = d.faces.iter().map(|t| self.texture_index[t] as u32).collect();
                GpuBlockInfo {
                    faces: [l[0] | (l[1] << 16), l[2] | (l[3] << 16), l[4] | (l[5] << 16)],
                    flags: d.kind as u32
                        | ((d.tint as u32) << 4)
                        | ((d.emission as u32) << 8)
                        | ((d.model as u32) << 16),
                    light: d.light[0] as u32
                        | ((d.light[1] as u32) << 8)
                        | ((d.light[2] as u32) << 16)
                        | ((d.light_att.min(15) as u32) << 24)
                        | (((d.kind == BlockKind::Opaque) as u32) << 28),
                }
            })
            .collect()
    }

    /// `#define B_STONE 1u` etc. for the world generator.
    pub fn glsl_header(&self) -> String {
        let mut s = String::from("// Generated by blocks.rs - do not edit\n#ifndef BLOCKS_GLSL\n#define BLOCKS_GLSL\n");
        for (i, d) in self.blocks.iter().enumerate() {
            s.push_str(&format!("#define B_{} {}u\n", d.name.to_uppercase(), i));
        }
        s.push_str(&format!("#define NUM_BLOCKS {}u\n", self.blocks.len()));
        let table = |f: &dyn Fn(&BlockDef) -> u32| self.blocks.iter().map(|d| format!("{}u", f(d))).collect::<Vec<_>>().join(", ");
        let n = self.blocks.len();
        s.push_str(&format!("const uint BLOCK_KIND[{n}] = uint[]({});\n", table(&|d| d.kind as u32)));
        s.push_str(&format!("const uint BLOCK_ROT[{n}] = uint[]({});\n#endif\n", table(&|d| d.rotation() as u32)));
        s
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
        assert!(r.get(log).is_column() && !r.get(stone).is_column());
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
}

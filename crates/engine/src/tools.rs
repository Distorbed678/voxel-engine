//! Editing tools: single block, brushes (add / remove / paint / smooth), box selection with
//! fill / clear / replace / copy / paste / rotate, prefabs and the undo history panel.
//! Operations that need the existing blocks (paint, smooth, replace, copy) run on an async
//! GPU region read and finish a few frames later.

use std::collections::HashMap;

use dear_imgui_rs::{Condition, Ui};
use glam::{IVec3, Vec4};

use crate::blocks::{BlockKind, BlockRegistry};
use crate::camera::Camera;
use crate::engine::PickHit;
use crate::input::Input;
use crate::prefab::{Prefab, PrefabLibrary};
use crate::ui::UiState;
use crate::world::{RegionRead, World};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolKind {
    Block,
    Brush,
    Select,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    Sphere,
    Cube,
    Cylinder,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BrushMode {
    Add,
    Remove,
    Paint,
    Smooth,
}

enum Pending {
    Paint { positions: Vec<IVec3>, block: u16 },
    Smooth { positions: Vec<IVec3> },
    Replace { from: u16, to: u16 },
    Copy,
    SavePrefab(String),
}

pub struct Tools {
    pub kind: ToolKind,
    pub shape: Shape,
    pub mode: BrushMode,
    pub radius: i32,
    pub sel_a: Option<IVec3>,
    pub sel_b: Option<IVec3>,
    pub clipboard: Option<Prefab>,
    pub paste_air: bool,
    pub replace_from: u16,
    pending: HashMap<u64, Pending>,
    /// Seconds since the brush last applied (hold-to-repeat).
    since_apply: f32,
    pub prefab_name: String,
    pub prefabs: PrefabLibrary,
    pub show_tools: bool,
    pub show_history: bool,
    pub status: String,
}

const MAX_SELECTION: i64 = 256 * 256 * 256;

fn shape_positions(shape: Shape, center: IVec3, r: i32) -> Vec<IVec3> {
    let mut v = Vec::new();
    let rf = r as f32 + 0.5;
    for dy in -r..=r {
        for dz in -r..=r {
            for dx in -r..=r {
                let inside = match shape {
                    Shape::Sphere => ((dx * dx + dy * dy + dz * dz) as f32) < rf * rf,
                    Shape::Cube => true,
                    Shape::Cylinder => ((dx * dx + dz * dz) as f32) < rf * rf,
                };
                if inside {
                    v.push(center + IVec3::new(dx, dy, dz));
                }
            }
        }
    }
    v
}

impl Tools {
    pub fn new(prefabs: PrefabLibrary) -> Self {
        Self {
            kind: ToolKind::Block,
            shape: Shape::Sphere,
            mode: BrushMode::Add,
            radius: 3,
            sel_a: None,
            sel_b: None,
            clipboard: None,
            paste_air: false,
            replace_from: 1,
            pending: HashMap::new(),
            since_apply: 0.0,
            prefab_name: "my_prefab".into(),
            prefabs,
            show_tools: true,
            show_history: false,
            status: String::new(),
        }
    }

    pub fn selection(&self) -> Option<(IVec3, IVec3)> {
        match (self.sel_a, self.sel_b) {
            (Some(a), Some(b)) => Some((a.min(b), a.max(b))),
            (Some(a), None) => Some((a, a)),
            _ => None,
        }
    }

    fn selection_ok(&mut self) -> Option<(IVec3, IVec3)> {
        let (a, b) = self.selection()?;
        let s = (b - a + IVec3::ONE).as_i64vec3();
        if s.x * s.y * s.z > MAX_SELECTION {
            self.status = "selection too large (max 256^3)".into();
            return None;
        }
        Some((a, b))
    }

    fn write(world: &mut World, label: &str, writes: &[(IVec3, u16)]) {
        let touched = world.overlay.apply(label, writes);
        world.invalidate_blocks(&touched);
    }

    /// Handle mouse input for the active tool. Returns true if the input was consumed.
    #[allow(clippy::too_many_arguments)]
    pub fn handle_input(
        &mut self,
        input: &Input,
        dt: f32,
        fly_mode: bool,
        pick: Option<PickHit>,
        cam_forward: glam::Vec3,
        world: &mut World,
        ui: &mut UiState,
        blocks: &BlockRegistry,
    ) {
        self.since_apply += dt;
        // Alt + wheel: brush radius
        if input.alt() && input.scroll != 0.0 {
            self.radius = (self.radius + input.scroll.signum() as i32).clamp(1, 32);
        }
        let Some(hit) = pick else { return };
        let primary = input.mouse_pressed[0] && (fly_mode || !input.shift());
        let secondary = if fly_mode { input.mouse_pressed[1] } else { input.mouse_pressed[0] && input.shift() };
        let primary_held = input.mouse_held[0] && (fly_mode || !input.shift());
        let secondary_held = if fly_mode { input.mouse_held[1] } else { input.mouse_held[0] && input.shift() };
        if input.mouse_pressed[2] {
            if hit.block != 0 {
                ui.selected_block = crate::blocks::block_id(hit.block); // placing orients it again
                ui.toast(format!("Picked {}", blocks.get(hit.block).display));
            }
            return;
        }
        match self.kind {
            ToolKind::Block => {
                if primary {
                    Self::write(world, "Break", &[(hit.block_pos, 0)]);
                } else if secondary {
                    let b = blocks.oriented(ui.selected_block, hit.normal, cam_forward);
                    Self::write(world, "Place", &[(hit.block_pos + hit.normal, b)]);
                }
            }
            ToolKind::Brush => {
                let repeat = self.since_apply > 0.12;
                let fire_p = primary || (primary_held && repeat);
                let fire_s = secondary || (secondary_held && repeat);
                if !(fire_p || fire_s) {
                    return;
                }
                self.since_apply = 0.0;
                let mode = match (self.mode, fire_s) {
                    (BrushMode::Add, true) => BrushMode::Remove,
                    (BrushMode::Remove, true) => BrushMode::Add,
                    (m, _) => m,
                };
                match mode {
                    BrushMode::Add => {
                        let c = hit.block_pos + hit.normal * self.radius;
                        let w: Vec<(IVec3, u16)> = shape_positions(self.shape, c, self.radius).into_iter().map(|p| (p, ui.selected_block)).collect();
                        Self::write(world, "Brush add", &w);
                    }
                    BrushMode::Remove => {
                        let w: Vec<(IVec3, u16)> = shape_positions(self.shape, hit.block_pos, self.radius).into_iter().map(|p| (p, 0)).collect();
                        Self::write(world, "Brush remove", &w);
                    }
                    BrushMode::Paint | BrushMode::Smooth => {
                        let positions = shape_positions(self.shape, hit.block_pos, self.radius);
                        let r = IVec3::splat(self.radius + 1);
                        let id = world.request_region(hit.block_pos - r, hit.block_pos + r);
                        let op = if mode == BrushMode::Paint {
                            Pending::Paint { positions, block: ui.selected_block }
                        } else {
                            Pending::Smooth { positions }
                        };
                        self.pending.insert(id, op);
                    }
                }
            }
            ToolKind::Select => {
                if primary {
                    self.sel_a = Some(hit.block_pos);
                    if self.sel_b.is_none() {
                        self.sel_b = Some(hit.block_pos);
                    }
                } else if secondary {
                    self.sel_b = Some(hit.block_pos);
                    if self.sel_a.is_none() {
                        self.sel_a = Some(hit.block_pos);
                    }
                }
            }
        }
    }

    /// A region read finished: run the operation that requested it.
    pub fn region_ready(&mut self, r: RegionRead, world: &mut World, blocks: &BlockRegistry, ui: &mut UiState) {
        let Some(op) = self.pending.remove(&r.id) else { return };
        let solid = |b: u16| b != 0 && matches!(blocks.get(b).kind, BlockKind::Opaque | BlockKind::Cutout);
        match op {
            Pending::Paint { positions, block } => {
                let w: Vec<(IVec3, u16)> = positions.into_iter().filter(|p| solid(r.get(*p))).map(|p| (p, block)).collect();
                Self::write(world, "Brush paint", &w);
            }
            Pending::Smooth { positions } => {
                let mut w = Vec::new();
                for p in positions {
                    let mut count = 0;
                    let mut hist: HashMap<u16, u32> = HashMap::new();
                    for dz in -1..=1 {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let b = r.get(p + IVec3::new(dx, dy, dz));
                                if solid(b) {
                                    count += 1;
                                    *hist.entry(b).or_default() += 1;
                                }
                            }
                        }
                    }
                    let cur = r.get(p);
                    let new = if count >= 14 { if solid(cur) { cur } else { hist.into_iter().max_by_key(|e| e.1).map(|e| e.0).unwrap_or(cur) } } else { 0 };
                    if new != cur && (solid(cur) || cur == 0) {
                        w.push((p, new));
                    }
                }
                Self::write(world, "Brush smooth", &w);
            }
            Pending::Replace { from, to } => {
                let mut w = Vec::new();
                for y in 0..r.size.y {
                    for z in 0..r.size.z {
                        for x in 0..r.size.x {
                            let p = r.min + IVec3::new(x, y, z);
                            if crate::blocks::block_id(r.get(p)) == from {
                                w.push((p, to));
                            }
                        }
                    }
                }
                ui.toast(format!("Replaced {} blocks", w.len()));
                Self::write(world, "Replace", &w);
            }
            Pending::Copy => {
                self.clipboard = Some(Prefab { name: "clipboard".into(), size: r.size, blocks: r.blocks });
                ui.toast(format!("Copied {}x{}x{}", r.size.x, r.size.y, r.size.z));
            }
            Pending::SavePrefab(name) => {
                let p = Prefab { name: name.clone(), size: r.size, blocks: r.blocks };
                match self.prefabs.save(p, blocks) {
                    Ok(()) => ui.toast(format!("Saved prefab '{name}' (usable in the Prefab node)")),
                    Err(e) => ui.toast(format!("Prefab save failed: {e}")),
                }
            }
        }
    }

    /// Fill the selection with `block` (0 clears it).
    pub fn fill_selection(&mut self, world: &mut World, block: u16) {
        let Some((a, b)) = self.selection_ok() else { return };
        let mut w = Vec::new();
        for y in a.y..=b.y {
            for z in a.z..=b.z {
                for x in a.x..=b.x {
                    w.push((IVec3::new(x, y, z), block));
                }
            }
        }
        Self::write(world, if block == 0 { "Clear" } else { "Fill" }, &w);
    }

    /// Replace every `from` block in the selection with `to` (after an async region read).
    pub fn replace_in_selection(&mut self, world: &mut World, from: u16, to: u16) {
        if let Some((a, b)) = self.selection_ok() {
            let id = world.request_region(a, b);
            self.pending.insert(id, Pending::Replace { from, to });
        }
    }

    /// Copy the selection into the clipboard (after an async region read).
    pub fn copy_selection(&mut self, world: &mut World) {
        if let Some((a, b)) = self.selection_ok() {
            let id = world.request_region(a, b);
            self.pending.insert(id, Pending::Copy);
        }
    }

    /// Paste the clipboard against the targeted face.
    pub fn paste_at_pick(&mut self, world: &mut World, pick: Option<PickHit>) {
        if let Some(h) = pick {
            self.paste_at(world, h.block_pos + h.normal);
        }
    }

    pub fn rotate_clipboard(&mut self, blocks: &BlockRegistry) {
        if let Some(c) = &self.clipboard {
            self.clipboard = Some(c.rotated(blocks));
        }
    }

    /// Save the selection as a prefab named `name` (after an async region read).
    pub fn save_selection_as_prefab(&mut self, world: &mut World, name: &str) {
        let name = worldgen::compile::sanitize(name.trim()).to_lowercase();
        if name.is_empty() {
            return;
        }
        if let Some((a, b)) = self.selection_ok() {
            let id = world.request_region(a, b);
            self.pending.insert(id, Pending::SavePrefab(name));
        }
    }

    /// Region reads still waiting for their result.
    pub fn pending_ops(&self) -> usize {
        self.pending.len()
    }

    fn paste_at(&mut self, world: &mut World, at: IVec3) {
        let Some(cb) = &self.clipboard else { return };
        let mut w = Vec::with_capacity(cb.blocks.len());
        for y in 0..cb.size.y {
            for z in 0..cb.size.z {
                for x in 0..cb.size.x {
                    let b = cb.get(IVec3::new(x, y, z));
                    if b != 0 || self.paste_air {
                        w.push((at + IVec3::new(x, y, z), b));
                    }
                }
            }
        }
        Self::write(world, "Paste", &w);
        self.sel_a = Some(at);
        self.sel_b = Some(at + cb.size - IVec3::ONE);
    }

    pub fn draw_ui(&mut self, ui: &Ui, world: &mut World, blocks: &BlockRegistry, state: &mut UiState, pick: Option<PickHit>) {
        if self.show_tools {
            let mut open = self.show_tools;
            let ds = ui.io().display_size();
            ui.window("Tools").size([300.0, 420.0], Condition::FirstUseEver).position([ds[0] - 620.0, 30.0], Condition::FirstUseEver).opened(&mut open).build(|| {
                let sel = blocks.get(state.selected_block);
                ui.text(format!("Block: {}", sel.display));
                if ui.radio_button("Block", self.kind == ToolKind::Block) {
                    self.kind = ToolKind::Block;
                }
                ui.same_line();
                if ui.radio_button("Brush", self.kind == ToolKind::Brush) {
                    self.kind = ToolKind::Brush;
                }
                ui.same_line();
                if ui.radio_button("Select", self.kind == ToolKind::Select) {
                    self.kind = ToolKind::Select;
                }
                ui.separator();
                match self.kind {
                    ToolKind::Block => {
                        ui.text_wrapped("Fly mode: LMB break, RMB place, MMB pick block. Editor mode: LMB break, Shift+LMB place.");
                    }
                    ToolKind::Brush => {
                        for (m, name) in [(BrushMode::Add, "Add"), (BrushMode::Remove, "Remove"), (BrushMode::Paint, "Paint"), (BrushMode::Smooth, "Smooth")] {
                            if ui.radio_button(name, self.mode == m) {
                                self.mode = m;
                            }
                            ui.same_line();
                        }
                        ui.new_line();
                        for (sh, name) in [(Shape::Sphere, "Sphere"), (Shape::Cube, "Cube"), (Shape::Cylinder, "Cylinder")] {
                            if ui.radio_button(name, self.shape == sh) {
                                self.shape = sh;
                            }
                            ui.same_line();
                        }
                        ui.new_line();
                        ui.slider("Radius", 1, 32, &mut self.radius);
                        ui.text_disabled("Alt+wheel: radius, hold to repeat");
                        ui.text_disabled("RMB / Shift+LMB: inverse (add<->remove)");
                    }
                    ToolKind::Select => {
                        ui.text_disabled("LMB: corner A, RMB (Shift+LMB): corner B");
                        if let Some((a, b)) = self.selection() {
                            let s = b - a + IVec3::ONE;
                            ui.text(format!("{} {} {} -> {} {} {}", a.x, a.y, a.z, b.x, b.y, b.z));
                            ui.text(format!("size {}x{}x{} ({} blocks)", s.x, s.y, s.z, s.x as i64 * s.y as i64 * s.z as i64));
                        } else {
                            ui.text("no selection");
                        }
                        if ui.button("Fill") {
                            self.fill_selection(world, state.selected_block);
                        }
                        ui.same_line();
                        if ui.button("Clear") {
                            self.fill_selection(world, 0);
                        }
                        ui.same_line();
                        if ui.button("Deselect") {
                            self.sel_a = None;
                            self.sel_b = None;
                        }
                        let names: Vec<&str> = blocks.blocks.iter().map(|b| if b.deleted { "(deleted)" } else { b.display.as_str() }).collect();
                        let mut from = self.replace_from as usize;
                        if ui.combo_simple_string("Replace from", &mut from, &names) {
                            self.replace_from = from as u16;
                        }
                        if ui.button("Replace with selected block") {
                            self.replace_in_selection(world, self.replace_from, state.selected_block);
                        }
                        ui.separator();
                        if ui.button("Copy") {
                            self.copy_selection(world);
                        }
                        ui.same_line();
                        if ui.button("Paste at target") {
                            self.paste_at_pick(world, pick);
                        }
                        ui.same_line();
                        if ui.button("Rotate 90") {
                            self.rotate_clipboard(blocks);
                        }
                        ui.checkbox("Paste air", &mut self.paste_air);
                        if let Some(c) = &self.clipboard {
                            ui.text_disabled(format!("clipboard {}x{}x{}", c.size.x, c.size.y, c.size.z));
                        }
                        ui.separator();
                        ui.input_text("Prefab name", &mut self.prefab_name).build();
                        if ui.button("Save selection as prefab") {
                            let name = self.prefab_name.clone();
                            self.save_selection_as_prefab(world, &name);
                        }
                    }
                }
                ui.separator();
                ui.text("Prefabs");
                let mut load = None;
                for (i, p) in self.prefabs.prefabs.iter().enumerate() {
                    if ui.small_button(format!("{} ({}x{}x{})##pf{i}", p.name, p.size.x, p.size.y, p.size.z)) {
                        load = Some(i);
                    }
                }
                if let Some(i) = load {
                    self.clipboard = Some(self.prefabs.prefabs[i].clone());
                    self.kind = ToolKind::Select;
                    state.toast(format!("Prefab '{}' in clipboard: use Paste", self.prefabs.prefabs[i].name));
                }
                if !self.pending.is_empty() {
                    ui.text_colored([1.0, 0.8, 0.3, 1.0], format!("{} operation(s) reading blocks...", self.pending.len()));
                }
                if !self.status.is_empty() {
                    ui.text_wrapped(&self.status);
                }
                ui.checkbox("History panel", &mut self.show_history);
            });
            self.show_tools = open;
        }
        if self.show_history {
            let mut open = self.show_history;
            ui.window("History").size([260.0, 300.0], Condition::FirstUseEver).opened(&mut open).build(|| {
                let n = world.overlay.undo.len();
                ui.text(format!("{n} step(s), {} redo", world.overlay.redo.len()));
                let mut undo_to = None;
                for (i, op) in world.overlay.undo.iter().enumerate().rev() {
                    if ui.selectable(format!("{}: {} ({} blocks)##h{i}", i + 1, op.label, op.changes.len())) {
                        undo_to = Some(i);
                    }
                }
                if let Some(i) = undo_to {
                    while world.overlay.undo.len() > i {
                        if let Some((_, touched)) = world.overlay.undo() {
                            world.invalidate_blocks(&touched);
                        } else {
                            break;
                        }
                    }
                }
            });
            self.show_history = open;
        }
    }

    /// Selection box / brush preview drawn with ImGui on top of the scene.
    pub fn draw_overlay(&self, ui: &Ui, cam: &Camera, pick: Option<PickHit>) {
        let ds = ui.io().display_size();
        let m = cam.matrices(ds[0] / ds[1].max(1.0));
        let project = |p: glam::DVec3| -> Option<[f32; 2]> {
            let rel = (p - cam.pos).as_vec3();
            let c = m.view_proj * Vec4::new(rel.x, rel.y, rel.z, 1.0);
            if c.w < 0.05 {
                return None;
            }
            Some([(c.x / c.w * 0.5 + 0.5) * ds[0], (c.y / c.w * 0.5 + 0.5) * ds[1]])
        };
        let dl = ui.get_foreground_draw_list();
        let draw_box = |a: IVec3, b: IVec3, col: [f32; 4]| {
            let lo = a.as_dvec3();
            let hi = (b + IVec3::ONE).as_dvec3();
            let c = |i: u32| glam::DVec3::new(if i & 1 != 0 { hi.x } else { lo.x }, if i & 2 != 0 { hi.y } else { lo.y }, if i & 4 != 0 { hi.z } else { lo.z });
            for (i, j) in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)] {
                if let (Some(p), Some(q)) = (project(c(i)), project(c(j))) {
                    dl.add_line(p, q, col).thickness(2.0).build();
                }
            }
        };
        if let Some((a, b)) = self.selection() {
            draw_box(a, b, [1.0, 0.85, 0.2, 1.0]);
        }
        if let Some(h) = pick {
            match self.kind {
                ToolKind::Block => draw_box(h.block_pos, h.block_pos, [1.0, 1.0, 1.0, 0.8]),
                ToolKind::Brush => {
                    let c = if self.mode == BrushMode::Add { h.block_pos + h.normal * self.radius } else { h.block_pos };
                    let r = IVec3::splat(self.radius);
                    draw_box(c - r, c + r, [0.4, 0.8, 1.0, 0.7]);
                }
                ToolKind::Select => {
                    draw_box(h.block_pos, h.block_pos, [1.0, 0.85, 0.2, 0.6]);
                    if let Some(cb) = &self.clipboard {
                        let at = h.block_pos + h.normal;
                        draw_box(at, at + cb.size - IVec3::ONE, [0.5, 1.0, 0.5, 0.5]);
                    }
                }
            }
        }
    }
}

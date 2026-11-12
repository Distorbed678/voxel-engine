//! Block editor: every block's properties, live. A list with search and categories; create,
//! duplicate and delete; kind, model, per-face textures (with a texture picker and a jump to the
//! texture editor), tint, glow, light, rotation and fluid behaviour; a rotating preview; and
//! place-testing in the world. Changes apply at once and are saved by the registry's autosave.

use dear_imgui_rs::{Condition, TreeNodeFlags, Ui};

use crate::blockgpu::AtlasRef;
use crate::blocks::{BlockDef, BlockKind, BlockRegistry, FACE_NAMES, Model, Rotation, Tint};
use crate::texture_editor::{CubeFace, TextureEditorUi, draw_cube};

/// How the face textures are edited.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FaceMode {
    All = 0,
    TopSideBottom = 1,
    Each = 2,
}

#[derive(Default)]
pub struct BlockEditorActions {
    /// select this block for placing (and switch to the block tool)
    pub select: Option<u16>,
    /// place this block in front of the camera
    pub place: Option<u16>,
}

pub struct BlockEditorUi {
    pub open: bool,
    pub selected: Option<u16>,
    filter: String,
    new_name: String,
    tex_filter: String,
    face_mode: FaceMode,
    /// face being picked in the texture popup
    picking: usize,
    pub error: Option<String>,
    spin: f32,
    shown_warnings: bool,
}

impl Default for BlockEditorUi {
    fn default() -> Self {
        Self::new()
    }
}

fn face_mode_of(d: &BlockDef) -> FaceMode {
    let f = &d.faces;
    if f.iter().all(|t| *t == f[0]) {
        FaceMode::All
    } else if f[0] == f[1] && f[0] == f[4] && f[0] == f[5] {
        FaceMode::TopSideBottom
    } else {
        FaceMode::Each
    }
}

/// A texture thumbnail from the UI atlas (first frame), or its name if it isn't uploaded.
fn thumb(ui: &Ui, reg: &BlockRegistry, atlas: Option<AtlasRef>, name: &str, p: [f32; 2], size: f32, time: f32) {
    let dl = ui.get_window_draw_list();
    match (atlas, reg.texture_index.get(name)) {
        (Some(a), Some(&i)) => {
            let s = &reg.textures[i];
            let frame = if s.frames > 1 { (time * s.fps) as u16 % s.frames } else { 0 };
            let (uv0, uv1) = a.uv(s.base + frame);
            dl.set_sampler_nearest();
            dl.add_image(a.id, p, [p[0] + size, p[1] + size], uv0, uv1, [1.0, 1.0, 1.0, 1.0]);
            dl.set_sampler_linear();
        }
        _ => {
            dl.add_rect(p, [p[0] + size, p[1] + size], [0.25, 0.25, 0.3, 1.0]).filled(true).build();
            dl.add_text([p[0] + 2.0, p[1] + size * 0.5 - 7.0], [0.8, 0.8, 0.8, 1.0], "?");
        }
    }
}

impl BlockEditorUi {
    pub fn new() -> Self {
        Self {
            open: false,
            selected: None,
            filter: String::new(),
            new_name: String::new(),
            tex_filter: String::new(),
            face_mode: FaceMode::All,
            picking: 0,
            error: None,
            spin: 0.6,
            shown_warnings: false,
        }
    }

    pub fn select(&mut self, reg: &BlockRegistry, id: u16) {
        if let Some(d) = reg.blocks.get(id as usize).filter(|d| !d.deleted && id != 0) {
            self.selected = Some(id);
            self.face_mode = face_mode_of(d);
        }
    }

    pub fn draw(&mut self, ui: &Ui, reg: &mut BlockRegistry, atlas: Option<AtlasRef>, tex: &mut TextureEditorUi, time: f32) -> BlockEditorActions {
        let mut act = BlockEditorActions::default();
        if !self.open {
            return act;
        }
        if self.selected.is_none_or(|id| reg.blocks.get(id as usize).is_none_or(|d| d.deleted)) {
            self.selected = None;
            if let Some(id) = reg.try_id("stone") {
                self.select(reg, id);
            }
        }
        let ds = ui.io().display_size();
        let mut open = true;
        ui.window("Block Editor")
            .size([700.0, 620.0], Condition::FirstUseEver)
            .position([10.0, (ds[1] - 610.0).clamp(30.0, 110.0)], Condition::FirstUseEver)
            .opened(&mut open)
            .build(|| {
                if !self.shown_warnings && !reg.load_warnings.is_empty() {
                    self.error = Some(reg.load_warnings.join("\n"));
                    self.shown_warnings = true;
                }
                // ---- status line ----
                ui.checkbox("Autosave", &mut reg.autosave);
                ui.same_line();
                if reg.has_unsaved() {
                    if ui.small_button("Save now")
                        && let Err(e) = reg.save()
                    {
                        self.error = Some(format!("saving failed: {e:#}"));
                    }
                    ui.same_line();
                    ui.text_colored([1.0, 0.8, 0.3, 1.0], "unsaved changes");
                } else {
                    ui.text_disabled(match &reg.dir {
                        Some(d) => format!("saved in {}", d.display()),
                        None => "built-in blocks (not saved)".into(),
                    });
                }
                if let Some(e) = &self.error {
                    ui.text_colored([1.0, 0.45, 0.4, 1.0], e);
                    ui.same_line();
                    if ui.small_button("ok") {
                        self.error = None;
                    }
                }
                ui.separator();
                // ---- list ----
                ui.child_window("block_list").size([210.0, 0.0]).border(true).build(ui, || self.list(ui, reg));
                ui.same_line();
                ui.child_window("block_props").size([0.0, 0.0]).build(ui, || self.properties(ui, reg, atlas, tex, time, &mut act));
            });
        self.open = open;
        act
    }

    fn list(&mut self, ui: &Ui, reg: &mut BlockRegistry) {
        ui.set_next_item_width(-1.0);
        ui.input_text("##block_filter", &mut self.filter).hint("search").build();
        if ui.small_button("New") {
            self.new_name.clear();
            ui.open_popup("new_block");
        }
        ui.same_line();
        if ui.small_button("Duplicate")
            && let Some(id) = self.selected
        {
            self.new_name = format!("{}_copy", reg.get(id).name);
            ui.open_popup("dup_block");
        }
        ui.same_line();
        let deletable = self.selected.is_some_and(|id| !reg.get(id).is_core());
        if ui.small_button("Delete") && deletable {
            ui.open_popup("del_block");
        }
        let create = |this: &mut Self, ui: &Ui, reg: &mut BlockRegistry, from: Option<u16>| {
            ui.input_text("Name", &mut this.new_name).hint("my_block").build();
            ui.text_disabled("a-z, 0-9 and _; it can't be changed later");
            if ui.button("Create") {
                match reg.create(&this.new_name.clone(), from) {
                    Ok(id) => {
                        this.select(reg, id);
                        this.error = None;
                        ui.close_current_popup();
                    }
                    Err(e) => this.error = Some(format!("{e:#}")),
                }
            }
        };
        ui.popup("new_block", || create(self, ui, reg, None));
        let sel = self.selected;
        ui.popup("dup_block", || create(self, ui, reg, sel));
        ui.popup("del_block", || {
            if let Some(id) = self.selected {
                ui.text(format!("Delete {}? Blocks of this kind in the world turn into air.", reg.get(id).display));
                if ui.button("Delete") {
                    if let Err(e) = reg.delete(id) {
                        self.error = Some(format!("{e:#}"));
                    }
                    self.selected = None;
                    ui.close_current_popup();
                }
                ui.same_line();
                if ui.button("Cancel") {
                    ui.close_current_popup();
                }
            }
        });
        ui.separator();
        let f = self.filter.to_lowercase();
        let mut cats: Vec<String> = reg.live().map(|(_, d)| d.category.clone()).collect();
        cats.sort();
        cats.dedup();
        for cat in cats {
            let items: Vec<(u16, String)> = reg
                .live()
                .filter(|(_, d)| d.category == cat && (f.is_empty() || d.display.to_lowercase().contains(&f) || d.name.contains(&f)))
                .map(|(i, d)| (i, d.display.clone()))
                .collect();
            if items.is_empty() {
                continue;
            }
            let label = if cat.is_empty() { "(no category)".to_string() } else { cat };
            if ui.collapsing_header(format!("{label}##cat"), if f.is_empty() { TreeNodeFlags::empty() } else { TreeNodeFlags::DEFAULT_OPEN }) {
                for (id, name) in items {
                    if ui.selectable_config(format!("{name}##b{id}")).selected(self.selected == Some(id)).build() {
                        self.select(reg, id);
                    }
                }
            }
        }
    }

    fn properties(&mut self, ui: &Ui, reg: &mut BlockRegistry, atlas: Option<AtlasRef>, tex: &mut TextureEditorUi, time: f32, act: &mut BlockEditorActions) {
        let Some(id) = self.selected else {
            ui.text_disabled("Select a block");
            return;
        };
        let mut d = reg.get(id).clone();
        let before = d.clone();

        // preview + identity
        let p = ui.cursor_screen_pos();
        let side = 120.0;
        ui.invisible_button("##block_preview", [side, side]);
        self.spin += ui.io().delta_time() * 0.6;
        if let Some(a) = atlas {
            let layer = |i: usize| {
                reg.texture_index.get(&d.faces[i]).map(|&k| {
                    let s = &reg.textures[k];
                    s.base + if s.frames > 1 { (time * s.fps) as u16 % s.frames } else { 0 }
                })
            };
            let layers: Vec<Option<u16>> = (0..6).map(layer).collect();
            if layers.iter().all(Option::is_some) {
                draw_cube(ui, [p[0] + side * 0.5, p[1] + side * 0.5], side * 0.34, self.spin, &|f| CubeFace::Atlas(a, layers[f].unwrap()));
            }
        }
        ui.same_line();
        ui.group(|| {
            ui.text(format!("{}  (id {id})", d.name));
            ui.set_next_item_width(170.0);
            ui.input_text("Display name", &mut d.display).build();
            ui.set_next_item_width(170.0);
            ui.input_text("Category", &mut d.category).build();
            if ui.button("Select for placing") {
                act.select = Some(id);
            }
            ui.same_line();
            if ui.button("Place in front of camera") {
                act.place = Some(id);
            }
            if d.is_core() {
                ui.text_disabled("core block: the generator falls back to it (can't be deleted)");
            }
        });
        ui.separator();

        // kind and model
        let mut k = BlockKind::ALL.iter().position(|x| *x == d.kind).unwrap_or(1);
        let kinds: Vec<&str> = BlockKind::ALL.iter().map(|k| k.label()).collect();
        ui.set_next_item_width(260.0);
        if ui.combo_simple_string("Kind", &mut k, &kinds) {
            d.kind = BlockKind::ALL[k];
            if d.kind == BlockKind::Model && d.model == Model::None {
                d.model = Model::Cross;
            }
        }
        if d.kind == BlockKind::Model {
            let mut m = Model::ALL.iter().position(|x| *x == d.model).unwrap_or(0);
            let models: Vec<&str> = Model::ALL.iter().map(|m| m.label()).collect();
            ui.set_next_item_width(260.0);
            if ui.combo_simple_string("Model", &mut m, &models) {
                d.model = Model::ALL[m];
            }
        }

        // textures
        ui.text("Textures");
        ui.same_line();
        // a mode that can't show the block's faces (changed elsewhere) widens to one that can
        let natural = face_mode_of(&d);
        if natural as u8 > self.face_mode as u8 {
            self.face_mode = natural;
        }
        for (m, label) in [(FaceMode::All, "all faces"), (FaceMode::TopSideBottom, "top / sides / bottom"), (FaceMode::Each, "each face")] {
            if ui.radio_button(label, self.face_mode == m) && self.face_mode != m {
                self.face_mode = m;
                // a narrower mode makes the faces it groups the same
                match m {
                    FaceMode::All => {
                        let t = d.faces[2].clone();
                        d.faces = std::array::from_fn(|_| t.clone());
                    }
                    FaceMode::TopSideBottom => {
                        let s = d.faces[0].clone();
                        for i in [1, 4, 5] {
                            d.faces[i] = s.clone();
                        }
                    }
                    FaceMode::Each => {}
                }
            }
            ui.same_line();
        }
        ui.new_line();
        let slots: Vec<(&str, Vec<usize>)> = match self.face_mode {
            FaceMode::All => vec![("all", vec![0, 1, 2, 3, 4, 5])],
            FaceMode::TopSideBottom => vec![("top", vec![2]), ("sides", vec![0, 1, 4, 5]), ("bottom", vec![3])],
            FaceMode::Each => (0..6).map(|i| (FACE_NAMES[i], vec![i])).collect(),
        };
        for (label, faces) in &slots {
            let first = faces[0];
            let p = ui.cursor_screen_pos();
            if ui.invisible_button(format!("##face{first}"), [36.0, 36.0]) {
                self.picking = first;
                self.tex_filter.clear();
                ui.open_popup("pick_tex");
            }
            thumb(ui, reg, atlas, &d.faces[first], p, 36.0, time);
            if ui.is_item_hovered() {
                ui.tooltip_text("click to choose a texture");
            }
            ui.same_line();
            ui.group(|| {
                ui.text(format!("{label}: {}", d.faces[first]));
                if ui.small_button(format!("Edit texture##e{first}")) {
                    tex.edit(reg, &d.faces[first].clone());
                }
            });
            // one picker for all slots: remember which faces it sets
            if self.picking == first {
                let mut chosen = None;
                ui.popup("pick_tex", || {
                    chosen = self.texture_picker(ui, reg, atlas, time);
                });
                if let Some(t) = chosen {
                    for &f in faces {
                        d.faces[f] = t.clone();
                    }
                }
            }
        }
        ui.separator();

        // look and light
        let mut t = Tint::ALL.iter().position(|x| *x == d.tint).unwrap_or(0);
        let tints: Vec<&str> = Tint::ALL.iter().map(|t| t.label()).collect();
        ui.set_next_item_width(260.0);
        if ui.combo_simple_string("Biome tint", &mut t, &tints) {
            d.tint = Tint::ALL[t];
        }
        let mut e = d.emission as i32;
        ui.set_next_item_width(260.0);
        if ui.slider("Glow (self-lit)", 0, 255, &mut e) {
            d.emission = e as u8;
        }
        let mut lc = d.light_color.map(|c| c as f32 / 255.0);
        ui.set_next_item_width(260.0);
        if ui.color_edit3("Light colour", &mut lc) {
            d.light_color = lc.map(|c| (c * 255.0 + 0.5) as u8);
        }
        let mut ll = d.light_level as i32;
        ui.set_next_item_width(260.0);
        if ui.slider("Light reach (blocks)", 0, 15, &mut ll) {
            d.light_level = ll as u8;
        }
        let mut la = d.light_att as i32;
        ui.set_next_item_width(260.0);
        if ui.slider("Light absorption", 0, 15, &mut la) {
            d.light_att = la as u8;
        }
        if ui.is_item_hovered() {
            ui.tooltip_text("extra falloff for light passing through (water 2, leaves 1)");
        }
        ui.separator();

        // behaviour
        let mut r = Rotation::ALL.iter().position(|x| *x == d.rotation).unwrap_or(0);
        let rots: Vec<&str> = Rotation::ALL.iter().map(|r| r.label()).collect();
        ui.set_next_item_width(260.0);
        if ui.combo_simple_string("Rotation", &mut r, &rots) {
            d.rotation = Rotation::ALL[r];
        }
        ui.checkbox("Washed away by flowing fluids", &mut d.washes_away);
        ui.checkbox("Fluid (flow levels, lowered surface)", &mut d.fluid);
        if ui.is_item_hovered() {
            ui.tooltip_text("the flowing-water simulation handles water only for now");
        }

        if d != before
            && let Err(e) = reg.update(id, d)
        {
            self.error = Some(format!("{e:#}"));
        }
    }

    /// A grid of every texture with a search box. Returns the chosen name.
    fn texture_picker(&mut self, ui: &Ui, reg: &mut BlockRegistry, atlas: Option<AtlasRef>, time: f32) -> Option<String> {
        let mut out = None;
        ui.set_next_item_width(240.0);
        ui.input_text("##tex_filter", &mut self.tex_filter).hint("search textures").build();
        let f = self.tex_filter.to_lowercase();
        let names: Vec<String> = reg.all_texture_names().into_iter().filter(|n| f.is_empty() || n.contains(&f)).collect();
        ui.child_window("tex_grid").size([8.0 * 40.0 + 16.0, 300.0]).build(ui, || {
            for (i, n) in names.iter().enumerate() {
                if i % 8 != 0 {
                    ui.same_line();
                }
                let p = ui.cursor_screen_pos();
                if ui.invisible_button(format!("##t_{n}"), [36.0, 36.0]) {
                    out = Some(n.clone());
                    ui.close_current_popup();
                }
                thumb(ui, reg, atlas, n, p, 36.0, time);
                if ui.is_item_hovered() {
                    ui.tooltip_text(n);
                }
            }
        });
        out
    }
}

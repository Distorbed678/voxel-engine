//! In-game texture editor: 32x32 pixel painting with animation frames.
//!
//! Tools: pencil, eraser, fill, line, rectangle (outline or filled), colour picker; mirror X/Y;
//! a palette, the texture's own colours and recent colours; undo/redo; zoom and grid. Frames can
//! be added, duplicated, deleted and reordered, with a speed, playback and onion skin. Every
//! change is applied to the world at once (the texture's layers are uploaded again), and saved
//! as `assets/textures/NAME.png` (frames stacked vertically) by the block registry's autosave.
//! The height map of a painted texture comes from its brightness.

use dear_imgui_rs::{Condition, Key, MouseButton, Ui};

use crate::blockgpu::AtlasRef;
use crate::blocks::{BlockRegistry, valid_name};
use crate::textures::{MAX_FRAMES, TEX, TexAsset, TexImage};

const N: i32 = TEX as i32;
type Px = [u8; 4];
type Frame = Vec<Px>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    Pencil,
    Eraser,
    Fill,
    Line,
    Rect,
    FilledRect,
    Picker,
}

impl Tool {
    const ALL: [(Tool, &'static str, &'static str); 7] = [
        (Tool::Pencil, "Pencil", "B"),
        (Tool::Eraser, "Eraser", "E"),
        (Tool::Fill, "Fill", "G"),
        (Tool::Line, "Line", "L"),
        (Tool::Rect, "Rect", "R"),
        (Tool::FilledRect, "Filled", "Shift+R"),
        (Tool::Picker, "Picker", "I"),
    ];
}

/// A default palette: greys, earth, greens, water, warm and magic tones.
const PALETTE: [u32; 32] = [
    0x000000, 0x1d1d21, 0x3b3b40, 0x5c5c63, 0x828282, 0xa8a8a8, 0xd6d6d6, 0xffffff, //
    0x3b2416, 0x5a3e28, 0x86603f, 0xb08c5a, 0xd8c690, 0xc8b47e, 0xa45222, 0x6b4520, //
    0x1f4a2a, 0x2f6a1c, 0x3f7d22, 0x5a9e2e, 0x86b84a, 0x123f86, 0x2a62b0, 0x8fb8e8, //
    0xc0300a, 0xff5a10, 0xffa020, 0xffe070, 0x7a4cc0, 0xd2a8ff, 0x2a8a9a, 0x8af0ff,
];

fn hex_px(c: u32) -> Px {
    [(c >> 16) as u8, (c >> 8) as u8, c as u8, 255]
}

fn px_f(p: Px) -> [f32; 4] {
    p.map(|v| v as f32 / 255.0)
}

fn f_px(c: [f32; 4]) -> Px {
    c.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
}

fn idx(x: i32, y: i32) -> usize {
    (y * N + x) as usize
}

fn frame_from(t: &TexImage) -> Frame {
    t.albedo_rgba8().as_chunks::<4>().0.to_vec()
}

/// Points on the line from a to b (Bresenham).
fn line_points(a: (i32, i32), b: (i32, i32)) -> Vec<(i32, i32)> {
    let (mut x, mut y) = a;
    let (dx, dy) = ((b.0 - x).abs(), -(b.1 - y).abs());
    let (sx, sy) = (if x < b.0 { 1 } else { -1 }, if y < b.1 { 1 } else { -1 });
    let mut err = dx + dy;
    let mut out = Vec::new();
    loop {
        out.push((x, y));
        if (x, y) == b {
            return out;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn rect_points(a: (i32, i32), b: (i32, i32), filled: bool) -> Vec<(i32, i32)> {
    let (x0, x1, y0, y1) = (a.0.min(b.0), a.0.max(b.0), a.1.min(b.1), a.1.max(b.1));
    let mut out = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            if filled || x == x0 || x == x1 || y == y0 || y == y1 {
                out.push((x, y));
            }
        }
    }
    out
}

/// Flood fill (4-connected) of the region of `at`'s colour.
pub fn flood_fill(f: &mut Frame, at: (i32, i32), c: Px) {
    let target = f[idx(at.0, at.1)];
    if target == c {
        return;
    }
    let mut stack = vec![at];
    while let Some((x, y)) = stack.pop() {
        if !(0..N).contains(&x) || !(0..N).contains(&y) || f[idx(x, y)] != target {
            continue;
        }
        f[idx(x, y)] = c;
        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]);
    }
}

#[derive(Clone)]
struct Snapshot {
    frames: Vec<Frame>,
    fps: f32,
    frame: usize,
}

pub struct TextureEditorUi {
    pub open: bool,
    /// texture being edited (empty: none)
    pub name: String,
    frames: Vec<Frame>,
    fps: f32,
    frame: usize,
    pub tool: Tool,
    pub color: [f32; 4],
    recent: Vec<Px>,
    pub mirror_x: bool,
    pub mirror_y: bool,
    zoom: i32,
    grid: bool,
    onion: bool,
    playing: bool,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    /// stroke in progress: last texel (pencil/eraser) or start (line/rect)
    stroke: Option<(i32, i32)>,
    /// line/rect start
    anchor: Option<(i32, i32)>,
    /// pixels differ from what the registry has
    pending_apply: bool,
    /// the working copy was changed (painted textures get a brightness height map)
    modified: bool,
    new_name: String,
    new_from: String,
    filter: String,
    pub error: Option<String>,
    /// Canvas origin and cell size last frame (scripts aim the mouse with `texel`).
    pub canvas: Option<([f32; 2], f32)>,
    /// The canvas was hovered last frame: Ctrl+Z/Y undo painting, not world edits.
    pub hovered: bool,
    spin: f32,
}

impl Default for TextureEditorUi {
    fn default() -> Self {
        Self::new()
    }
}

impl TextureEditorUi {
    pub fn new() -> Self {
        Self {
            open: false,
            name: String::new(),
            frames: Vec::new(),
            fps: 8.0,
            frame: 0,
            tool: Tool::Pencil,
            color: [0.35, 0.62, 0.18, 1.0],
            recent: Vec::new(),
            mirror_x: false,
            mirror_y: false,
            zoom: 12,
            grid: true,
            onion: false,
            playing: false,
            undo: Vec::new(),
            redo: Vec::new(),
            stroke: None,
            anchor: None,
            pending_apply: false,
            modified: false,
            new_name: String::new(),
            new_from: String::new(),
            filter: String::new(),
            error: None,
            canvas: None,
            hovered: false,
            spin: 0.6,
        }
    }

    /// Open a texture (loading its current pixels from the registry).
    pub fn edit(&mut self, reg: &mut BlockRegistry, name: &str) {
        let a = reg.store.get(name).clone();
        self.name = name.to_string();
        self.frames = a.frames.iter().map(frame_from).collect();
        self.fps = a.fps;
        self.frame = 0;
        self.undo.clear();
        self.redo.clear();
        self.stroke = None;
        self.anchor = None;
        self.pending_apply = false;
        self.modified = false;
        self.open = true;
    }

    /// Create a texture `name` from a copy of `from` (empty: a transparent one).
    pub fn create(&mut self, reg: &mut BlockRegistry, name: &str, from: &str) -> anyhow::Result<()> {
        if !valid_name(name) {
            anyhow::bail!("`{name}` is not a valid texture name (a-z, 0-9, _)");
        }
        if reg.all_texture_names().iter().any(|n| n == name) {
            anyhow::bail!("a texture named `{name}` exists already");
        }
        let asset = if from.is_empty() {
            let mut t = TexImage::new();
            t.color.fill([0.0; 4]);
            TexAsset::still(t)
        } else {
            reg.store.get(from).clone()
        };
        let frames: Vec<Frame> = asset.frames.iter().map(frame_from).collect();
        reg.set_texture(name, Self::asset_of(&frames, asset.fps));
        self.edit(reg, name);
        Ok(())
    }

    fn asset_of(frames: &[Frame], fps: f32) -> TexAsset {
        TexAsset { frames: frames.iter().map(|f| TexImage::from_rgba8(&f.iter().flatten().copied().collect::<Vec<u8>>())).collect(), fps }
    }

    fn snapshot(&mut self) {
        self.undo.push(Snapshot { frames: self.frames.clone(), fps: self.fps, frame: self.frame });
        if self.undo.len() > 64 {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    fn changed(&mut self) {
        self.modified = true;
        self.pending_apply = true;
    }

    pub fn undo(&mut self) {
        if let Some(s) = self.undo.pop() {
            self.redo.push(Snapshot { frames: std::mem::replace(&mut self.frames, s.frames), fps: self.fps, frame: self.frame });
            self.fps = s.fps;
            self.frame = s.frame.min(self.frames.len() - 1);
            self.changed();
        }
    }

    pub fn redo(&mut self) {
        if let Some(s) = self.redo.pop() {
            self.undo.push(Snapshot { frames: std::mem::replace(&mut self.frames, s.frames), fps: self.fps, frame: self.frame });
            self.fps = s.fps;
            self.frame = s.frame.min(self.frames.len() - 1);
            self.changed();
        }
    }

    /// Push the working copy into the registry (uploads the changed layers).
    pub fn apply(&mut self, reg: &mut BlockRegistry) {
        if self.pending_apply && !self.name.is_empty() && self.modified {
            reg.set_texture(&self.name, Self::asset_of(&self.frames, self.fps));
        }
        self.pending_apply = false;
    }

    fn paint_px(&mut self, x: i32, y: i32, c: Px) {
        let f = &mut self.frames[self.frame];
        let mut pts = vec![(x, y)];
        if self.mirror_x {
            pts.push((N - 1 - x, y));
        }
        if self.mirror_y {
            pts.extend(pts.clone().into_iter().map(|(x, y)| (x, N - 1 - y)));
        }
        for (x, y) in pts {
            if (0..N).contains(&x) && (0..N).contains(&y) {
                f[idx(x, y)] = c;
            }
        }
    }

    fn use_color(&mut self, c: Px) {
        self.recent.retain(|r| *r != c);
        self.recent.insert(0, c);
        self.recent.truncate(12);
    }

    /// Paint with the current tool as if the mouse went down at `a` and up at `b` (scripts and
    /// the canvas use the same code).
    pub fn stroke(&mut self, a: (i32, i32), b: (i32, i32)) {
        self.snapshot();
        let c = f_px(self.color);
        match self.tool {
            Tool::Pencil | Tool::Eraser => {
                let c = if self.tool == Tool::Eraser { [0; 4] } else { c };
                for (x, y) in line_points(a, b) {
                    self.paint_px(x, y, c);
                }
            }
            Tool::Line => {
                for (x, y) in line_points(a, b) {
                    self.paint_px(x, y, c);
                }
            }
            Tool::Rect | Tool::FilledRect => {
                for (x, y) in rect_points(a, b, self.tool == Tool::FilledRect) {
                    self.paint_px(x, y, c);
                }
            }
            Tool::Fill => {
                let f = &mut self.frames[self.frame];
                flood_fill(f, b, c);
            }
            Tool::Picker => {
                self.color = px_f(self.frames[self.frame][idx(b.0, b.1)]);
                self.undo.pop();
                return;
            }
        }
        if self.tool != Tool::Eraser {
            self.use_color(c);
        }
        self.changed();
    }

    /// The colour of a texel of the working copy.
    #[cfg(test)]
    pub fn texel(&self, frame: usize, x: i32, y: i32) -> Option<Px> {
        self.frames.get(frame).map(|f| f[idx(x.clamp(0, N - 1), y.clamp(0, N - 1))])
    }

    pub fn set_frame(&mut self, f: usize) {
        if !self.frames.is_empty() {
            self.frame = f.min(self.frames.len() - 1);
        }
    }

    /// Frame operations (scripts and buttons).
    pub fn add_frame(&mut self, duplicate: bool) {
        if self.frames.len() >= MAX_FRAMES {
            return;
        }
        self.snapshot();
        let f = if duplicate { self.frames[self.frame].clone() } else { vec![[0; 4]; TEX * TEX] };
        self.frames.insert(self.frame + 1, f);
        self.frame += 1;
        self.changed();
    }

    pub fn delete_frame(&mut self) {
        if self.frames.len() > 1 {
            self.snapshot();
            self.frames.remove(self.frame);
            self.frame = self.frame.min(self.frames.len() - 1);
            self.changed();
        }
    }

    pub fn set_fps(&mut self, fps: f32) {
        self.fps = fps.clamp(0.125, 120.0);
        self.changed();
    }

    pub fn draw(&mut self, ui: &Ui, reg: &mut BlockRegistry, time: f32) {
        if !self.open {
            return;
        }
        let ds = ui.io().display_size();
        let mut open = true;
        self.hovered = false;
        ui.window("Texture Editor")
            .size([740.0, 660.0], Condition::FirstUseEver)
            .position([(ds[0] - 750.0).max(0.0), 30.0], Condition::FirstUseEver)
            .opened(&mut open)
            .build(|| self.contents(ui, reg, time));
        self.open = open;
        if !open {
            self.canvas = None;
        }
        self.apply(reg);
    }

    fn contents(&mut self, ui: &Ui, reg: &mut BlockRegistry, time: f32) {
        // ---- texture choice ----
        ui.text(if self.name.is_empty() { "No texture open".to_string() } else { format!("Texture: {}", self.name) });
        ui.same_line();
        if ui.small_button("Open...") {
            ui.open_popup("open_tex");
        }
        ui.same_line();
        if ui.small_button("New...") {
            self.new_from = self.name.clone();
            ui.open_popup("new_tex");
        }
        if !self.name.is_empty() {
            ui.same_line();
            let src = if self.modified || reg.store.is_dirty(&self.name) {
                "edited, not saved yet"
            } else if reg.store.is_file(&self.name) {
                "PNG file"
            } else {
                "built-in (procedural)"
            };
            ui.text_disabled(format!("  {src}, {} frame(s)", self.frames.len()));
        }
        ui.popup("open_tex", || {
            ui.input_text("##filter", &mut self.filter).hint("search").build();
            let f = self.filter.to_lowercase();
            ui.child_window("tex_list").size([260.0, 300.0]).build(ui, || {
                for n in reg.all_texture_names() {
                    if (f.is_empty() || n.contains(&f)) && ui.selectable_config(&n).build() {
                        self.edit(reg, &n);
                        ui.close_current_popup();
                    }
                }
            });
        });
        ui.popup("new_tex", || {
            ui.input_text("Name", &mut self.new_name).hint("my_texture").build();
            ui.input_text("Start from", &mut self.new_from).hint("(empty: transparent)").build();
            if ui.button("Create") {
                let (n, f) = (self.new_name.clone(), self.new_from.clone());
                match self.create(reg, &n, &f) {
                    Ok(()) => {
                        self.error = None;
                        ui.close_current_popup();
                    }
                    Err(e) => self.error = Some(format!("{e:#}")),
                }
            }
            if let Some(e) = &self.error {
                ui.text_colored([1.0, 0.4, 0.4, 1.0], e);
            }
        });
        if self.name.is_empty() || self.frames.is_empty() {
            ui.text_wrapped("Open a texture, or pick one from a block in the Block Editor.");
            return;
        }

        // ---- toolbar ----
        for (i, (t, label, key)) in Tool::ALL.iter().enumerate() {
            if i > 0 {
                ui.same_line();
            }
            if ui.radio_button(format!("{label}##tool"), self.tool == *t) {
                self.tool = *t;
            }
            if ui.is_item_hovered() {
                ui.tooltip_text(format!("{label} ({key})"));
            }
        }
        ui.checkbox("Mirror X", &mut self.mirror_x);
        ui.same_line();
        ui.checkbox("Mirror Y", &mut self.mirror_y);
        ui.same_line();
        ui.checkbox("Grid", &mut self.grid);
        ui.same_line();
        ui.set_next_item_width(90.0);
        ui.slider("Zoom", 6, 20, &mut self.zoom);
        ui.same_line();
        if ui.small_button("Undo") {
            self.undo();
        }
        ui.same_line();
        if ui.small_button("Redo") {
            self.redo();
        }

        // ---- canvas ----
        let cell = self.zoom as f32;
        let size = cell * N as f32;
        let p0 = ui.cursor_screen_pos();
        ui.invisible_button("##canvas", [size, size]);
        let hovered = ui.is_item_hovered();
        let active = ui.is_item_active();
        self.hovered = hovered;
        self.canvas = Some((p0, cell));
        let m = ui.mouse_pos();
        let at = (((m[0] - p0[0]) / cell).floor() as i32, ((m[1] - p0[1]) / cell).floor() as i32);
        let inside = (0..N).contains(&at.0) && (0..N).contains(&at.1);
        let clamp = |p: (i32, i32)| (p.0.clamp(0, N - 1), p.1.clamp(0, N - 1));
        if hovered && inside && ui.is_mouse_clicked(MouseButton::Right) {
            // right click picks a colour
            self.color = px_f(self.frames[self.frame][idx(at.0, at.1)]);
        }
        if hovered {
            let ctrl = ui.io().key_ctrl();
            if ctrl && ui.is_key_pressed(Key::Z) {
                self.undo();
            } else if ctrl && ui.is_key_pressed(Key::Y) {
                self.redo();
            } else if !ctrl {
                let shift = ui.io().key_shift();
                for (t, k) in [(Tool::Pencil, Key::B), (Tool::Eraser, Key::E), (Tool::Fill, Key::G), (Tool::Line, Key::L), (Tool::Picker, Key::I)] {
                    if ui.is_key_pressed(k) {
                        self.tool = t;
                    }
                }
                if ui.is_key_pressed(Key::R) {
                    self.tool = if shift { Tool::FilledRect } else { Tool::Rect };
                }
            }
        }
        if active && inside && ui.is_mouse_clicked(MouseButton::Left) {
            match self.tool {
                Tool::Pencil | Tool::Eraser => {
                    self.stroke(at, at);
                    self.stroke = Some(at);
                }
                Tool::Line | Tool::Rect | Tool::FilledRect => self.anchor = Some(at),
                Tool::Fill | Tool::Picker => self.stroke(at, at),
            }
        } else if active
            && ui.is_mouse_down(MouseButton::Left)
            && let Some(last) = self.stroke
            && clamp(at) != last
        {
            // continue the stroke without a new undo step
            let c = if self.tool == Tool::Eraser { [0; 4] } else { f_px(self.color) };
            for (x, y) in line_points(last, clamp(at)) {
                self.paint_px(x, y, c);
            }
            self.stroke = Some(clamp(at));
            self.changed();
        }
        if ui.is_mouse_released(MouseButton::Left) {
            if let Some(a) = self.anchor.take() {
                self.stroke(a, clamp(at));
            }
            self.stroke = None;
        }

        let dl = ui.get_window_draw_list();
        let frame = &self.frames[self.frame];
        let prev = if self.onion && self.frames.len() > 1 { Some(&self.frames[(self.frame + self.frames.len() - 1) % self.frames.len()]) } else { None };
        for y in 0..N {
            for x in 0..N {
                let (a, b) = ([p0[0] + x as f32 * cell, p0[1] + y as f32 * cell], [p0[0] + (x + 1) as f32 * cell, p0[1] + (y + 1) as f32 * cell]);
                let p = frame[idx(x, y)];
                if p[3] < 255 {
                    // checkerboard under transparent texels
                    let k = if (x + y) % 2 == 0 { 0.32 } else { 0.42 };
                    dl.add_rect(a, b, [k, k, k, 1.0]).filled(true).build();
                }
                if p[3] > 0 {
                    dl.add_rect(a, b, px_f(p)).filled(true).build();
                }
                if let Some(pf) = prev {
                    let q = pf[idx(x, y)];
                    if q[3] > 0 && q != p {
                        dl.add_rect(a, b, [q[0] as f32 / 255.0, q[1] as f32 / 255.0, q[2] as f32 / 255.0, 0.35]).filled(true).build();
                    }
                }
            }
        }
        if self.grid && cell >= 8.0 {
            for i in 0..=N {
                let k = if i % 8 == 0 { [0.0, 0.0, 0.0, 0.55] } else { [0.0, 0.0, 0.0, 0.18] };
                let o = i as f32 * cell;
                dl.add_line([p0[0] + o, p0[1]], [p0[0] + o, p0[1] + size], k).build();
                dl.add_line([p0[0], p0[1] + o], [p0[0] + size, p0[1] + o], k).build();
            }
        }
        // line / rect preview and the hovered texel
        if let Some(a) = self.anchor {
            let pts = match self.tool {
                Tool::Line => line_points(a, clamp(at)),
                _ => rect_points(a, clamp(at), self.tool == Tool::FilledRect),
            };
            let c = self.color;
            for (x, y) in pts {
                dl.add_rect([p0[0] + x as f32 * cell, p0[1] + y as f32 * cell], [p0[0] + (x + 1) as f32 * cell, p0[1] + (y + 1) as f32 * cell], [c[0], c[1], c[2], 0.7]).filled(true).build();
            }
        }
        if hovered && inside {
            let (a, b) = ([p0[0] + at.0 as f32 * cell, p0[1] + at.1 as f32 * cell], [p0[0] + (at.0 + 1) as f32 * cell, p0[1] + (at.1 + 1) as f32 * cell]);
            dl.add_rect(a, b, [1.0, 1.0, 1.0, 0.9]).thickness(1.5).build();
        }
        drop(dl);

        // ---- side panel: colour, palette, frames, preview ----
        ui.same_line();
        ui.child_window("tex_side").size([0.0, 0.0]).build(ui, || {
            ui.set_next_item_width(220.0);
            ui.color_edit4("Colour", &mut self.color);
            if hovered && inside {
                let p = self.frames[self.frame][idx(at.0, at.1)];
                ui.text_disabled(format!("{},{}  #{:02x}{:02x}{:02x} a{}", at.0, at.1, p[0], p[1], p[2], p[3]));
            } else {
                ui.text_disabled("right click: pick a colour");
            }
            let swatch = |ui: &Ui, label: String, c: Px| -> bool { ui.color_button(label, px_f(c)) };
            ui.text("Palette");
            for (i, c) in PALETTE.iter().enumerate() {
                if i % 8 != 0 {
                    ui.same_line();
                }
                if swatch(ui, format!("##pal{i}"), hex_px(*c)) {
                    self.color = px_f(hex_px(*c));
                }
            }
            // the texture's own colours, most used first
            let mut counts: std::collections::HashMap<Px, u32> = std::collections::HashMap::new();
            for p in self.frames.iter().flatten().filter(|p| p[3] > 0) {
                *counts.entry(*p).or_default() += 1;
            }
            let mut own: Vec<(Px, u32)> = counts.into_iter().collect();
            own.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            ui.text("In this texture");
            for (i, (c, _)) in own.iter().take(16).enumerate() {
                if i % 8 != 0 {
                    ui.same_line();
                }
                if swatch(ui, format!("##own{i}"), *c) {
                    self.color = px_f(*c);
                }
            }
            if !self.recent.is_empty() {
                ui.text("Recent");
                for (i, c) in self.recent.clone().iter().enumerate() {
                    if i % 8 != 0 {
                        ui.same_line();
                    }
                    if swatch(ui, format!("##rec{i}"), *c) {
                        self.color = px_f(*c);
                    }
                }
            }
            ui.separator();
            // frames
            ui.text(format!("Frames ({})", self.frames.len()));
            let thumb = 34.0;
            for i in 0..self.frames.len() {
                if i % 6 != 0 {
                    ui.same_line();
                }
                let p = ui.cursor_screen_pos();
                if ui.invisible_button(format!("##frame{i}"), [thumb, thumb]) {
                    self.frame = i;
                    self.playing = false;
                }
                draw_pixels(ui, &self.frames[i], p, thumb / N as f32, 2);
                let col = if i == self.frame { [1.0, 0.85, 0.3, 1.0] } else { [0.5, 0.5, 0.5, 1.0] };
                ui.get_window_draw_list().add_rect(p, [p[0] + thumb, p[1] + thumb], col).thickness(if i == self.frame { 2.0 } else { 1.0 }).build();
            }
            if ui.small_button("Add") {
                self.add_frame(false);
            }
            ui.same_line();
            if ui.small_button("Duplicate") {
                self.add_frame(true);
            }
            ui.same_line();
            if ui.small_button("Delete") {
                self.delete_frame();
            }
            ui.same_line();
            if ui.small_button("<") && self.frame > 0 {
                self.snapshot();
                self.frames.swap(self.frame, self.frame - 1);
                self.frame -= 1;
                self.changed();
            }
            ui.same_line();
            if ui.small_button(">") && self.frame + 1 < self.frames.len() {
                self.snapshot();
                self.frames.swap(self.frame, self.frame + 1);
                self.frame += 1;
                self.changed();
            }
            if self.frames.len() > 1 {
                let mut fps = self.fps;
                ui.set_next_item_width(120.0);
                if ui.slider("fps", 0.5f32, 30.0f32, &mut fps) {
                    self.set_fps(fps);
                }
                ui.same_line();
                ui.checkbox("Play", &mut self.playing);
                ui.same_line();
                ui.checkbox("Onion skin", &mut self.onion);
                if self.playing {
                    self.frame = (time * self.fps) as usize % self.frames.len();
                }
            }
            ui.separator();
            // rotating preview with the texture on every face
            ui.text("Preview");
            self.spin += ui.io().delta_time() * 0.6;
            let p = ui.cursor_screen_pos();
            let side = 130.0;
            ui.invisible_button("##tex_preview", [side, side]);
            let shown = if self.playing || self.frames.len() == 1 { self.frame } else { (time * self.fps) as usize % self.frames.len() };
            let px = &self.frames[shown];
            draw_cube(ui, [p[0] + side * 0.5, p[1] + side * 0.5], side * 0.36, self.spin, &|_| CubeFace::Pixels(px));
        });
    }
}

/// Texels drawn as rectangles (thumbnails), every `step`-th texel.
pub fn draw_pixels(ui: &Ui, f: &[Px], p0: [f32; 2], cell: f32, step: i32) {
    let dl = ui.get_window_draw_list();
    let c = cell * step as f32;
    for y in (0..N).step_by(step as usize) {
        for x in (0..N).step_by(step as usize) {
            let p = f[idx(x, y)];
            if p[3] > 0 {
                let a = [p0[0] + x as f32 * cell, p0[1] + y as f32 * cell];
                dl.add_rect(a, [a[0] + c, a[1] + c], px_f(p)).filled(true).build();
            }
        }
    }
}

/// What a cube face shows in a preview.
pub enum CubeFace<'a> {
    /// RGBA8 texels (drawn texel by texel)
    Pixels(&'a [Px]),
    /// a tile of the UI atlas
    Atlas(AtlasRef, u16),
}

/// A rotating cube in an isometric-ish view: faces turned towards the viewer, shaded by
/// direction. `face(dir)` gives each face's texture (dir: +X, -X, +Y, -Y, +Z, -Z).
pub fn draw_cube<'a>(ui: &Ui, center: [f32; 2], half: f32, yaw: f32, face: &dyn Fn(usize) -> CubeFace<'a>) {
    let (sy, cy) = yaw.sin_cos();
    let pitch = 0.55f32;
    let (sp, cp) = pitch.sin_cos();
    let project = |v: [f32; 3]| -> ([f32; 2], f32) {
        // yaw around Y, then tilt the view down by `pitch`
        let x = v[0] * cy - v[2] * sy;
        let z = v[0] * sy + v[2] * cy;
        let y = v[1] * cp - z * sp;
        let depth = v[1] * sp + z * cp;
        ([center[0] + x * half, center[1] - y * half], depth)
    };
    // corners of each face: top-left, top-right, bottom-right, bottom-left of its texture
    let faces: [([f32; 3], [[f32; 3]; 4], f32); 6] = [
        ([1.0, 0.0, 0.0], [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0]], 0.78),
        ([-1.0, 0.0, 0.0], [[-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]], 0.78),
        ([0.0, 1.0, 0.0], [[-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]], 1.0),
        ([0.0, -1.0, 0.0], [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], 0.5),
        ([0.0, 0.0, 1.0], [[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]], 0.64),
        ([0.0, 0.0, -1.0], [[1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0]], 0.64),
    ];
    let dl = ui.get_window_draw_list();
    dl.set_sampler_nearest();
    for (dir, (n, corners, shade)) in faces.iter().enumerate() {
        // only faces turned towards the viewer
        if project(*n).1 <= 0.01 {
            continue;
        }
        let p: Vec<[f32; 2]> = corners.iter().map(|c| project(*c).0).collect();
        let lerp = |a: [f32; 2], b: [f32; 2], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        let at = |u: f32, v: f32| lerp(lerp(p[0], p[1], u), lerp(p[3], p[2], u), v);
        match face(dir) {
            CubeFace::Pixels(px) => {
                for y in 0..N {
                    for x in 0..N {
                        let c = px[idx(x, y)];
                        if c[3] < 128 {
                            continue;
                        }
                        let (u0, v0, u1, v1) = (x as f32 / N as f32, y as f32 / N as f32, (x + 1) as f32 / N as f32, (y + 1) as f32 / N as f32);
                        let k = shade / 255.0;
                        let col = [c[0] as f32 * k, c[1] as f32 * k, c[2] as f32 * k, 1.0];
                        dl.add_quad_filled(at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1), col);
                    }
                }
            }
            CubeFace::Atlas(a, layer) => {
                let (uv0, uv1) = a.uv(layer);
                dl.add_image_quad(a.id, p[0], p[1], p[2], p[3], uv0, [uv1[0], uv0[1]], uv1, [uv0[0], uv1[1]], [*shade, *shade, *shade, 1.0]);
            }
        }
    }
    dl.set_sampler_linear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn painting_tools() {
        let mut reg = BlockRegistry::new();
        let mut t = TextureEditorUi::new();
        t.create(&mut reg, "test_tex", "").unwrap();
        assert!(t.create(&mut reg, "test_tex", "").is_err() && t.create(&mut reg, "Bad", "").is_err());
        t.color = [1.0, 0.0, 0.0, 1.0];
        t.tool = Tool::Line;
        t.stroke((0, 0), (31, 31));
        assert_eq!(t.texel(0, 15, 15), Some([255, 0, 0, 255]));
        assert_eq!(t.texel(0, 15, 16), Some([0; 4]));
        // mirror X paints both sides
        t.mirror_x = true;
        t.tool = Tool::Pencil;
        t.color = [0.0, 0.0, 1.0, 1.0];
        t.stroke((2, 5), (2, 5));
        assert_eq!(t.texel(0, 29, 5), Some([0, 0, 255, 255]));
        // fill stops at the diagonal line
        t.mirror_x = false;
        t.tool = Tool::Fill;
        t.color = [0.0, 1.0, 0.0, 1.0];
        t.stroke((20, 2), (20, 2));
        assert_eq!(t.texel(0, 31, 0), Some([0, 255, 0, 255]));
        assert_eq!(t.texel(0, 0, 31), Some([0; 4]));
        // rect, undo, redo
        t.tool = Tool::FilledRect;
        t.stroke((10, 20), (12, 22));
        assert_eq!(t.texel(0, 11, 21), Some([0, 255, 0, 255]));
        t.undo();
        assert_eq!(t.texel(0, 11, 21), Some([0; 4]));
        t.redo();
        assert_eq!(t.texel(0, 11, 21), Some([0, 255, 0, 255]));
        // frames reach the registry: a new layout once a block uses the texture
        t.add_frame(true);
        t.set_fps(4.0);
        t.apply(&mut reg);
        let a = reg.store.get("test_tex");
        assert_eq!((a.frames.len(), a.fps), (2, 4.0));
        assert_eq!(&a.frames[1].albedo_rgba8()[(21 * 32 + 11) * 4..][..4], &[0, 255, 0, 255]);
    }
}

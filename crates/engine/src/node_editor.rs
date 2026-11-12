//! Visual world-generation editor (imgui-node-editor via dear-node-editor).
//!
//! Left: graph browser (World / Zones / Biomes / Functions). Centre: node canvas.
//! Right: inspector for the selected node. Every edit marks the generator dirty; parameter-only
//! edits regenerate the world immediately, structural edits rebuild the GPU shaders in the
//! background first.

use std::collections::{HashMap, HashSet};

use dear_imgui_rs::{Condition, Key, MouseButton, Ui};
use dear_node_editor::{EditorConfig, EditorContext, LinkId, NodeEditorUiExt, NodeId, PinId, PinKind};
use worldgen::nodes::{self, ParamKind, PinType};
use worldgen::{Graph, GraphKind, Link, Node, Value};

use crate::blocks::BlockRegistry;
use crate::wg::WorldgenSystem;

/// A deferred structural change to the graph being drawn.
type GraphEdit = Box<dyn FnOnce(&mut Graph)>;

type GraphKey = (GraphKind, String);

pub struct NodeEditorUi {
    pub open: bool,
    pub current: GraphKey,
    contexts: HashMap<GraphKey, EditorContext>,
    placed: HashMap<GraphKey, HashSet<u32>>,
    pub selected: Vec<u32>,
    palette_filter: String,
    palette_pos: [f32; 2],
    palette_open_request: bool,
    pending_link_from: Option<(u32, bool, u8)>,
    new_name: String,
    new_kind: usize,
    rename_buf: String,
    clipboard: Option<(Vec<Node>, Vec<Link>)>,
    undo: Vec<(GraphKey, Graph)>,
    redo: Vec<(GraphKey, Graph)>,
    navigate: bool,
    status: String,
    /// Last on-screen size of each comment box, to turn resizes into width/height changes.
    comment_sizes: HashMap<u32, [f32; 2]>,
}

fn pin_id(node: u32, out: bool, idx: u8) -> PinId {
    PinId::new(((node as usize) << 8) | ((out as usize) << 7) | idx as usize)
}

fn decode_pin(p: PinId) -> (u32, bool, u8) {
    let v: usize = p.into();
    ((v >> 8) as u32, (v >> 7) & 1 == 1, (v & 0x7F) as u8)
}

fn link_id(l: &Link) -> LinkId {
    LinkId::new(((l.from_node as usize) << 40) | ((l.from_pin as usize) << 32) | ((l.to_node as usize) << 8) | (l.to_pin as usize + 1))
}

fn category_color(cat: &str) -> [f32; 4] {
    match cat.split('/').next().unwrap_or("") {
        "Output" => [0.65, 0.2, 0.2, 1.0],
        "Density" => [0.2, 0.4, 0.7, 1.0],
        "Caves" => [0.45, 0.32, 0.2, 1.0],
        "Material" => [0.7, 0.45, 0.15, 1.0],
        "Decoration" => [0.3, 0.6, 0.25, 1.0],
        "Props" => [0.15, 0.5, 0.3, 1.0],
        "Layout" => [0.5, 0.3, 0.65, 1.0],
        "Function" => [0.15, 0.55, 0.55, 1.0],
        _ => [0.4, 0.4, 0.4, 1.0],
    }
}

impl NodeEditorUi {
    pub fn new() -> Self {
        Self {
            open: true,
            current: (GraphKind::Biome, "oak_forest".into()),
            contexts: HashMap::new(),
            placed: HashMap::new(),
            selected: Vec::new(),
            palette_filter: String::new(),
            palette_pos: [0.0, 0.0],
            palette_open_request: false,
            pending_link_from: None,
            new_name: String::new(),
            new_kind: 2,
            rename_buf: String::new(),
            clipboard: None,
            undo: Vec::new(),
            redo: Vec::new(),
            navigate: true,
            status: String::new(),
            comment_sizes: HashMap::new(),
        }
    }

    /// Editor contexts must be created outside of a Dear ImGui frame.
    pub fn prepare(&mut self, imgui: &dear_imgui_rs::Context, wg: &WorldgenSystem) {
        if wg.project.get(self.current.0, &self.current.1).is_none()
            && let Some(g) = wg.project.graphs.first() {
                self.current = (g.kind, g.name.clone());
                self.navigate = true;
            }
        if !self.contexts.contains_key(&self.current) {
            let cfg = EditorConfig::new().no_settings_file();
            self.contexts.insert(self.current.clone(), EditorContext::create_with_config(imgui, cfg));
        }
    }

    fn snapshot(&mut self, wg: &WorldgenSystem) {
        if let Some(g) = wg.project.get(self.current.0, &self.current.1) {
            self.undo.push((self.current.clone(), g.clone()));
            if self.undo.len() > 200 {
                self.undo.remove(0);
            }
            self.redo.clear();
        }
    }

    fn restore(&mut self, wg: &mut WorldgenSystem, from_undo: bool) {
        let entry = if from_undo { self.undo.pop() } else { self.redo.pop() };
        let Some((key, graph)) = entry else { return };
        if let Some(g) = wg.project.get_mut(key.0, &key.1) {
            let current = std::mem::replace(g, graph);
            if from_undo {
                self.redo.push((key.clone(), current));
            } else {
                self.undo.push((key.clone(), current));
            }
            self.placed.remove(&key);
            self.current = key;
            wg.mark_dirty();
        }
    }

    pub fn draw(&mut self, ui: &Ui, wg: &mut WorldgenSystem, blocks: &BlockRegistry) {
        if !self.open {
            return;
        }
        let mut open = self.open;
        let ds = ui.io().display_size();
        ui.window("World Generator")
            .size([(ds[0] - 360.0).max(600.0), (ds[1] * 0.5).max(360.0)], Condition::FirstUseEver)
            .position([350.0, ds[1] * 0.5 - 10.0], Condition::FirstUseEver)
            .opened(&mut open)
            .build(|| {
                self.toolbar(ui, wg);
                let avail = ui.content_region_avail();
                ui.child_window("graphs").size([210.0, avail[1]]).border(true).build(ui, || self.graph_list(ui, wg));
                ui.same_line();
                ui.child_window("canvas_host").size([avail[0] - 210.0 - 300.0 - 16.0, avail[1]]).build(ui, || self.canvas(ui, wg, blocks));
                ui.same_line();
                ui.child_window("inspector").size([300.0, avail[1]]).border(true).build(ui, || self.inspector(ui, wg, blocks));
            });
        self.open = open;
    }

    fn toolbar(&mut self, ui: &Ui, wg: &mut WorldgenSystem) {
        if ui.button("Save") {
            wg.save();
            self.status = "saved".into();
        }
        ui.same_line();
        if ui.button("Revert") {
            match wg.reload_from_disk() {
                Ok(()) => {
                    self.placed.clear();
                    self.status = "reloaded from disk".into();
                }
                Err(e) => self.status = format!("{e:#}"),
            }
        }
        ui.same_line();
        if ui.button("Reset to defaults") {
            wg.reset_to_defaults();
            self.placed.clear();
        }
        ui.same_line();
        ui.checkbox("Autosave", &mut wg.autosave);
        ui.same_line();
        if ui.button("Auto-layout") {
            self.snapshot(wg);
            if let Some(g) = wg.project.get_mut(self.current.0, &self.current.1) {
                worldgen::layout::auto_layout(g);
                wg.unsaved = true;
            }
            self.placed.remove(&self.current);
            self.navigate = true;
        }
        ui.same_line();
        if ui.button("Undo") {
            self.restore(wg, true);
        }
        ui.same_line();
        if ui.button("Redo") {
            self.restore(wg, false);
        }
        ui.same_line();
        if wg.is_building() {
            ui.text_colored([1.0, 0.8, 0.3, 1.0], "building shaders...");
        } else if wg.is_optimizing() {
            ui.text_colored([0.8, 0.8, 0.4, 1.0], "optimising in background...");
        } else {
            ui.text_colored(
                [0.6, 0.9, 0.6, 1.0],
                format!("graph {:.2} ms, shaders {:.0} ms, {} params", wg.last_compile_ms, wg.last_build_ms, wg.compiled.params.len()),
            );
        }
        if wg.unsaved {
            ui.same_line();
            ui.text_disabled("(unsaved)");
        }
        if !self.status.is_empty() {
            ui.same_line();
            ui.text_disabled(&self.status);
        }
        if let Some(e) = &wg.last_error {
            ui.text_colored([1.0, 0.4, 0.4, 1.0], "Error:");
            ui.same_line();
            let first: String = e.lines().take(3).collect::<Vec<_>>().join(" | ");
            ui.text_wrapped(first);
        }
        if !wg.compiled.warnings.is_empty() {
            ui.text_colored([1.0, 0.75, 0.3, 1.0], format!("{} warning(s):", wg.compiled.warnings.len()));
            ui.same_line();
            ui.text_wrapped(&wg.compiled.warnings[0]);
        }
    }

    fn graph_list(&mut self, ui: &Ui, wg: &mut WorldgenSystem) {
        for kind in [GraphKind::World, GraphKind::Zone, GraphKind::Biome, GraphKind::Function] {
            ui.text_colored([0.7, 0.75, 0.9, 1.0], format!("{}s", kind.label()));
            for name in wg.project.names(kind) {
                let sel = self.current.0 == kind && self.current.1 == name;
                if ui.selectable_config(format!("  {name}##{}", kind.label())).selected(sel).build() && !sel {
                    self.current = (kind, name.clone());
                    self.selected.clear();
                    self.navigate = true;
                    self.rename_buf = name.clone();
                }
            }
            ui.spacing();
        }
        ui.separator();
        ui.text("New graph");
        let kinds = ["Zone", "Biome", "Function"];
        ui.combo_simple_string("##kind", &mut self.new_kind, &kinds);
        ui.input_text("##name", &mut self.new_name).build();
        if ui.button("Create") {
            let kind = [GraphKind::Zone, GraphKind::Biome, GraphKind::Function][self.new_kind.min(2)];
            let name = worldgen::compile::sanitize(self.new_name.trim()).to_lowercase();
            if !name.is_empty() && wg.project.get(kind, &name).is_none() {
                wg.project.graphs.push(Graph::new(&name, kind));
                self.current = (kind, name);
                self.navigate = true;
                wg.mark_dirty();
                self.new_name.clear();
            }
        }
        ui.separator();
        ui.text(format!("Selected: {}", self.current.1));
        if self.current.0 != GraphKind::World {
            ui.input_text("##rename", &mut self.rename_buf).build();
            ui.same_line();
            if ui.button("Rename") {
                let new = worldgen::compile::sanitize(self.rename_buf.trim()).to_lowercase();
                if !new.is_empty() && wg.project.get(self.current.0, &new).is_none() {
                    let old = self.current.1.clone();
                    if let Some(g) = wg.project.get_mut(self.current.0, &old) {
                        g.name = new.clone();
                    }
                    for g in &mut wg.project.graphs {
                        for n in &mut g.nodes {
                            for v in n.params.values_mut() {
                                if *v == Value::Ref(old.clone()) {
                                    *v = Value::Ref(new.clone());
                                }
                            }
                        }
                    }
                    self.current.1 = new;
                    wg.mark_dirty();
                }
            }
            if ui.button("Duplicate")
                && let Some(g) = wg.project.get(self.current.0, &self.current.1).cloned() {
                    let mut n = 2;
                    while wg.project.get(g.kind, &format!("{}_{n}", g.name)).is_some() {
                        n += 1;
                    }
                    let mut g2 = g;
                    g2.name = format!("{}_{n}", g2.name);
                    self.current = (g2.kind, g2.name.clone());
                    wg.project.graphs.push(g2);
                    self.navigate = true;
                    wg.mark_dirty();
                }
            ui.same_line();
            let count = wg.project.graphs.iter().filter(|g| g.kind == self.current.0).count();
            if ui.button("Delete") && count > 1 {
                let key = self.current.clone();
                wg.project.graphs.retain(|g| !(g.kind == key.0 && g.name == key.1));
                self.contexts.remove(&key);
                wg.mark_dirty();
            }
        }
    }

    fn canvas(&mut self, ui: &Ui, wg: &mut WorldgenSystem, blocks: &BlockRegistry) {
        let key = self.current.clone();
        let Some(ctx) = self.contexts.get(&key) else { return };
        // keyboard shortcuts when the canvas is hovered
        let hovered = ui.is_window_hovered();
        let ctrl = ui.io().key_ctrl();
        let mut do_copy = false;
        let mut do_paste = false;
        let mut do_dup = false;
        if hovered && !ui.is_any_item_active() {
            if ctrl && ui.is_key_pressed(Key::C) {
                do_copy = true;
            }
            if ctrl && ui.is_key_pressed(Key::V) {
                do_paste = true;
            }
            if ctrl && ui.is_key_pressed(Key::D) {
                do_dup = true;
            }
            if ctrl && ui.is_key_pressed(Key::Z) {
                self.restore(wg, true);
                return;
            }
            if ctrl && ui.is_key_pressed(Key::Y) {
                self.restore(wg, false);
                return;
            }
            if ui.is_key_pressed(Key::Space) {
                self.palette_open_request = true;
                self.pending_link_from = None;
            }
        }
        let Some(ctx) = self.contexts.get(&key).or(Some(ctx)) else { return };
        let size = ui.content_region_avail();
        let ed = ui.node_editor(ctx, "wg_canvas", size);
        let placed = self.placed.entry(key.clone()).or_default();
        let mut changed_params = false;
        let mut structural: Vec<GraphEdit> = Vec::new();
        let mut snapshot_needed = false;

        let graph = wg.project.get_mut(key.0, &key.1);
        let Some(graph) = graph else {
            ed.end();
            return;
        };
        let names_biome: Vec<String> = Vec::new();
        let _ = names_biome;

        for ni in 0..graph.nodes.len() {
            let node_id = graph.nodes[ni].id;
            let nid = NodeId::new(node_id as usize);
            if !placed.contains(&node_id) {
                ed.set_node_position(nid, graph.nodes[ni].pos);
                placed.insert(node_id);
            }
            if graph.nodes[ni].kind == "comment" {
                // a resizable group box with a note; dragging it moves the nodes inside
                let n = &graph.nodes[ni];
                let size = [
                    n.params.get("width").map(|v| v.as_f32()).unwrap_or(420.0),
                    n.params.get("height").map(|v| v.as_f32()).unwrap_or(260.0),
                ];
                let text = if n.comment.is_empty() { "Comment".to_string() } else { n.comment.clone() };
                ed.node(nid, |_tok| {
                    ui.text_colored([0.95, 0.88, 0.55, 1.0], &text);
                    ed.group(size);
                });
                continue;
            }
            let Some(def) = nodes::def(&graph.nodes[ni].kind) else { continue };
            let color = category_color(def.category);
            let node = &mut graph.nodes[ni];
            ed.node(nid, |tok| {
                // header
                let p0 = ui.cursor_screen_pos();
                let title_w = ui.calc_text_size(def.title)[0].max(150.0);
                {
                    let dl = ui.get_window_draw_list();
                    dl.add_rect([p0[0] - 8.0, p0[1] - 8.0], [p0[0] + title_w + 8.0, p0[1] + 18.0], color).filled(true).rounding(4.0).build();
                }
                ui.text(def.title);
                ui.dummy([title_w, 4.0]);
                // input pins
                for (i, (name, ty)) in def.inputs.iter().enumerate() {
                    tok.pin(pin_id(node_id, false, i as u8), PinKind::Input, |_| {
                        let p = ui.cursor_screen_pos();
                        ui.get_window_draw_list().add_circle([p[0] + 5.0, p[1] + 8.0], 5.0, ty.color()).filled(true).build();
                        ui.dummy([12.0, 16.0]);
                        ui.same_line();
                        ui.text(name);
                    });
                }
                // inline params
                let _w = ui.push_item_width(130.0);
                for pd in def.params {
                    let label = format!("{}##{}_{}", pd.name, node_id, pd.name);
                    match pd.kind {
                        ParamKind::Float { min, max, .. } => {
                            let mut v = node.params.get(pd.name).map(|v| v.as_f32()).unwrap_or(0.0);
                            let speed = ((max - min) * 0.002).max(0.001);
                            if ui.drag_config::<_, f32>(&label).speed(speed).range(min, max).build(ui, &mut v) {
                                node.params.insert(pd.name.to_string(), Value::F(v));
                                changed_params = true;
                            }
                        }
                        ParamKind::Int { min, max, .. } => {
                            let mut v = node.params.get(pd.name).map(|v| v.as_f32() as i32).unwrap_or(0);
                            if ui.drag_config::<_, i32>(&label).speed(0.1).range(min, max).build(ui, &mut v) {
                                node.params.insert(pd.name.to_string(), Value::I(v));
                                changed_params = true;
                            }
                        }
                        ParamKind::Bool(_) => {
                            let mut v = matches!(node.params.get(pd.name), Some(Value::B(true)));
                            if ui.checkbox(&label, &mut v) {
                                node.params.insert(pd.name.to_string(), Value::B(v));
                                snapshot_needed = true;
                                changed_params = true;
                            }
                        }
                        ParamKind::Curve => {
                            if let Some(Value::Curve(pts)) = node.params.get_mut(pd.name)
                                && curve_widget(ui, &format!("curve{node_id}"), pts, [200.0, 120.0]) {
                                    changed_params = true;
                                }
                        }
                        _ => {
                            let s = match node.params.get(pd.name) {
                                Some(Value::E(s)) | Some(Value::Block(s)) | Some(Value::Ref(s)) => s.clone(),
                                Some(Value::Color(c)) => format!("({:.2}, {:.2}, {:.2})", c[0], c[1], c[2]),
                                _ => String::new(),
                            };
                            ui.text_disabled(format!("{}: {}", pd.name, if s.is_empty() { "<none>" } else { &s }));
                        }
                    }
                }
                // output pins
                for (i, (name, ty)) in def.outputs.iter().enumerate() {
                    tok.pin(pin_id(node_id, true, i as u8), PinKind::Output, |_| {
                        let tw = ui.calc_text_size(name)[0];
                        ui.dummy([(title_w - tw - 14.0).max(0.0), 1.0]);
                        ui.same_line();
                        ui.text(name);
                        ui.same_line();
                        let p = ui.cursor_screen_pos();
                        ui.get_window_draw_list().add_circle([p[0] + 5.0, p[1] + 8.0], 5.0, ty.color()).filled(true).build();
                        ui.dummy([12.0, 16.0]);
                    });
                }
            });
        }
        // links
        for l in &graph.links {
            let ty = nodes::def(&graph.node(l.from_node).map(|n| n.kind.clone()).unwrap_or_default())
                .and_then(|d| d.outputs.get(l.from_pin as usize).map(|o| o.1))
                .unwrap_or(PinType::Density);
            let c = ty.color();
            ed.link_colored(link_id(l), pin_id(l.from_node, true, l.from_pin), pin_id(l.to_node, false, l.to_pin), c, 2.0);
        }

        // ---- create links / nodes ----
        if let Some(cs) = ed.begin_create([1.0, 1.0, 1.0, 1.0], 2.0) {
            if let Some((a, b)) = cs.query_new_link() {
                let (na, oa, ia) = decode_pin(a);
                let (nb, ob, ib) = decode_pin(b);
                let (from, to) = if oa && !ob { ((na, ia), (nb, ib)) } else if ob && !oa { ((nb, ib), (na, ia)) } else { ((0, 0), (0, 0)) };
                let ok = from.0 != 0 && {
                    let mut probe = graph.clone();
                    probe.connect(from.0, from.1, to.0, to.1)
                };
                if ok {
                    if cs.accept_new_item_styled([0.4, 1.0, 0.4, 1.0], 3.0) {
                        snapshot_needed = true;
                        structural.push(Box::new(move |g: &mut Graph| {
                            g.connect(from.0, from.1, to.0, to.1);
                        }));
                    }
                } else {
                    cs.reject_new_item_styled([1.0, 0.3, 0.3, 1.0], 2.0);
                }
            }
            if let Some(p) = cs.query_new_node()
                && cs.accept_new_item() {
                    self.pending_link_from = Some(decode_pin(p));
                    self.palette_open_request = true;
                }
            cs.end();
        }
        // ---- delete ----
        if let Some(ds) = ed.begin_delete() {
            while let Some((lid, _, _)) = ds.query_deleted_link() {
                if let Some(l) = graph.links.iter().find(|l| link_id(l) == lid).copied()
                    && ds.accept_deleted_item(true) {
                        snapshot_needed = true;
                        structural.push(Box::new(move |g: &mut Graph| g.links.retain(|x| *x != l)));
                    }
            }
            while let Some(nid) = ds.query_deleted_node() {
                let id: usize = nid.into();
                let is_output = graph.node(id as u32).map(|n| n.kind.ends_with("_output")).unwrap_or(true);
                if is_output {
                    ds.reject_deleted_item();
                } else if ds.accept_deleted_item(true) {
                    snapshot_needed = true;
                    structural.push(Box::new(move |g: &mut Graph| g.remove_node(id as u32)));
                }
            }
            ds.end();
        }
        if ed.show_background_context_menu() {
            self.palette_open_request = true;
            self.pending_link_from = None;
        }
        if self.palette_open_request {
            self.palette_pos = ed.screen_to_canvas(ui.mouse_pos());
        }
        // comment boxes resized by dragging their edges
        for n in graph.nodes.iter_mut().filter(|n| n.kind == "comment") {
            let s = ed.node_size(NodeId::new(n.id as usize));
            if !(s[0].is_finite() && s[0] > 0.0) {
                continue;
            }
            if let Some(last) = self.comment_sizes.insert(n.id, s)
                && ((s[0] - last[0]).abs() > 0.5 || (s[1] - last[1]).abs() > 0.5)
            {
                for (k, d) in [("width", s[0] - last[0]), ("height", s[1] - last[1])] {
                    let v = n.params.get(k).map(|v| v.as_f32()).unwrap_or(0.0) + d;
                    n.params.insert(k.into(), Value::F(v.max(40.0)));
                }
                wg.unsaved = true;
            }
        }
        // selection + positions
        self.selected = ed.selected_nodes().into_iter().map(|n| { let v: usize = n.into(); v as u32 }).collect();
        for n in graph.nodes.iter_mut() {
            let p = ed.node_position(NodeId::new(n.id as usize));
            if p[0].is_finite() && (p[0] - n.pos[0]).abs() + (p[1] - n.pos[1]).abs() > 0.5 {
                n.pos = p;
                wg.unsaved = true;
            }
        }
        if self.navigate {
            ed.navigate_to_content(0.0);
            self.navigate = false;
        }

        // ---- create-node palette (editor suspended so popups work) ----
        let susp = ed.suspend();
        if self.palette_open_request {
            ui.open_popup("create_node");
            self.palette_open_request = false;
            self.palette_filter.clear();
        }
        let mut create: Option<&'static str> = None;
        let kind = key.0;
        let link_from = self.pending_link_from;
        let from_type = link_from.and_then(|(n, out, idx)| {
            let node = graph.node(n)?;
            let d = nodes::def(&node.kind)?;
            if out { d.outputs.get(idx as usize).map(|p| (p.1, true)) } else { d.inputs.get(idx as usize).map(|p| (p.1, false)) }
        });
        ui.popup("create_node", || {
            ui.text("Add node");
            ui.input_text("##filter", &mut self.palette_filter).hint("search...").build();
            let f = self.palette_filter.to_lowercase();
            for cat in nodes::categories() {
                let items: Vec<&nodes::NodeDef> = nodes::NODES
                    .iter()
                    .filter(|d| d.category == cat && nodes::allowed_in(d, kind))
                    .filter(|d| f.is_empty() || d.title.to_lowercase().contains(&f) || d.category.to_lowercase().contains(&f))
                    .filter(|d| match from_type {
                        Some((t, true)) => d.inputs.iter().any(|i| i.1 == t),
                        Some((t, false)) => d.outputs.iter().any(|o| o.1 == t),
                        None => true,
                    })
                    .collect();
                if items.is_empty() {
                    continue;
                }
                ui.text_colored(category_color(cat), cat);
                for d in items {
                    if ui.selectable(format!("   {}", d.title)) {
                        create = Some(d.kind);
                    }
                    if ui.is_item_hovered() {
                        ui.tooltip_text(d.help);
                    }
                }
            }
        });
        susp.resume();
        ed.end();

        // ---- apply deferred edits ----
        if snapshot_needed || create.is_some() || do_paste || do_dup {
            self.snapshot(wg);
        }
        let graph = wg.project.get_mut(key.0, &key.1).unwrap();
        let mut structural_change = !structural.is_empty();
        for f in structural {
            f(graph);
        }
        if let Some(kind) = create {
            let id = graph.add(kind, self.palette_pos);
            if let Some((n, out, idx)) = link_from {
                let d = nodes::def(kind).unwrap();
                if out {
                    if let Some(t) = from_type.map(|t| t.0)
                        && let Some(i) = d.inputs.iter().position(|p| p.1 == t) {
                            graph.connect(n, idx, id, i as u8);
                        }
                } else if let Some(t) = from_type.map(|t| t.0)
                    && let Some(o) = d.outputs.iter().position(|p| p.1 == t) {
                        graph.connect(id, o as u8, n, idx);
                    }
            }
            self.pending_link_from = None;
            structural_change = true;
        }
        if do_copy && !self.selected.is_empty() {
            let set: HashSet<u32> = self.selected.iter().copied().collect();
            let nodes: Vec<Node> = graph.nodes.iter().filter(|n| set.contains(&n.id) && !n.kind.ends_with("_output")).cloned().collect();
            let links: Vec<Link> = graph.links.iter().filter(|l| set.contains(&l.from_node) && set.contains(&l.to_node)).copied().collect();
            self.status = format!("copied {} node(s)", nodes.len());
            self.clipboard = Some((nodes, links));
        }
        if do_dup && !self.selected.is_empty() {
            let set: HashSet<u32> = self.selected.iter().copied().collect();
            let nodes: Vec<Node> = graph.nodes.iter().filter(|n| set.contains(&n.id) && !n.kind.ends_with("_output")).cloned().collect();
            let links: Vec<Link> = graph.links.iter().filter(|l| set.contains(&l.from_node) && set.contains(&l.to_node)).copied().collect();
            paste(graph, &nodes, &links, [40.0, 40.0]);
            structural_change = true;
        }
        if do_paste
            && let Some((nodes, links)) = &self.clipboard {
                let allowed: Vec<Node> = nodes.iter().filter(|n| nodes::def(&n.kind).map(|d| nodes::allowed_in(d, key.0)).unwrap_or(false)).cloned().collect();
                paste(graph, &allowed, links, [60.0, 60.0]);
                structural_change = true;
            }
        if structural_change || changed_params {
            wg.mark_dirty();
        }
        let _ = blocks;
    }

    fn inspector(&mut self, ui: &Ui, wg: &mut WorldgenSystem, blocks: &BlockRegistry) {
        let key = self.current.clone();
        ui.text_colored([0.7, 0.75, 0.9, 1.0], format!("{} '{}'", key.0.label(), key.1));
        ui.text_disabled("Right-click / Space: add node");
        ui.text_disabled("Drag pin to empty space: add + link");
        ui.text_disabled("Del: delete, Ctrl+C/V/D, Ctrl+Z/Y");
        ui.separator();
        let Some(&sel) = self.selected.first() else {
            ui.text_wrapped("Select a node to edit all of its parameters here.");
            if key.0 == GraphKind::Biome {
                self.make_function_button(ui, wg);
            }
            return;
        };
        let biome_names = wg.project.names(GraphKind::Biome);
        let zone_names = wg.project.names(GraphKind::Zone);
        let fn_names = wg.project.names(GraphKind::Function);
        let block_names: Vec<&str> = blocks.blocks.iter().map(|b| b.name).collect();
        let prefab_names = wg.prefab_names.clone();
        let mut changed = false;
        let mut snapshot = false;
        {
            let Some(graph) = wg.project.get_mut(key.0, &key.1) else { return };
            let Some(node) = graph.node_mut(sel) else { return };
            let Some(def) = nodes::def(&node.kind) else { return };
            ui.text_colored(category_color(def.category), def.title);
            ui.text_wrapped(def.help);
            ui.separator();
            if node.kind == "comment" {
                if ui.input_text_multiline("##comment_text", &mut node.comment, [-1.0, 120.0]).build() {
                    wg.unsaved = true;
                }
                return;
            }
            for pd in def.params {
                let label = format!("{}##insp{}", pd.name, pd.name);
                match pd.kind {
                    ParamKind::Float { min, max, .. } => {
                        let mut v = node.params.get(pd.name).map(|v| v.as_f32()).unwrap_or(0.0);
                        if ui.slider(&label, min, max, &mut v) {
                            node.params.insert(pd.name.into(), Value::F(v));
                            changed = true;
                        }
                    }
                    ParamKind::Int { min, max, .. } => {
                        let mut v = node.params.get(pd.name).map(|v| v.as_f32() as i32).unwrap_or(0);
                        if ui.slider(&label, min, max, &mut v) {
                            node.params.insert(pd.name.into(), Value::I(v));
                            changed = true;
                        }
                    }
                    ParamKind::Bool(_) => {
                        let mut v = matches!(node.params.get(pd.name), Some(Value::B(true)));
                        if ui.checkbox(&label, &mut v) {
                            node.params.insert(pd.name.into(), Value::B(v));
                            changed = true;
                            snapshot = true;
                        }
                    }
                    ParamKind::Enum(opts, _) => {
                        let cur = match node.params.get(pd.name) {
                            Some(Value::E(s)) => opts.iter().position(|o| o == s).unwrap_or(0),
                            _ => 0,
                        };
                        let mut i = cur;
                        if ui.combo_simple_string(&label, &mut i, opts) && i != cur {
                            node.params.insert(pd.name.into(), Value::E(opts[i].into()));
                            changed = true;
                            snapshot = true;
                        }
                    }
                    ParamKind::Block(_) => {
                        let cur = match node.params.get(pd.name) {
                            Some(Value::Block(s)) => block_names.iter().position(|o| o == s).unwrap_or(0),
                            _ => 0,
                        };
                        let mut i = cur;
                        if ui.combo_simple_string(&label, &mut i, &block_names) && i != cur {
                            node.params.insert(pd.name.into(), Value::Block(block_names[i].into()));
                            changed = true;
                            snapshot = true;
                        }
                    }
                    ParamKind::Ref(kind) => {
                        let list = match kind {
                            GraphKind::Biome => &biome_names,
                            GraphKind::Zone => &zone_names,
                            _ => &fn_names,
                        };
                        let cur = match node.params.get(pd.name) {
                            Some(Value::Ref(s)) => list.iter().position(|o| o == s),
                            _ => None,
                        };
                        let mut i = cur.unwrap_or(0);
                        if !list.is_empty() && ui.combo_simple_string(&label, &mut i, list) && Some(i) != cur {
                            node.params.insert(pd.name.into(), Value::Ref(list[i].clone()));
                            changed = true;
                            snapshot = true;
                        }
                        if cur.is_some() && ui.small_button(format!("open {}##open{}", kind.label(), pd.name)) {
                            self.current = (kind, list[i].clone());
                            self.selected.clear();
                            self.navigate = true;
                            return;
                        }
                    }
                    ParamKind::Color(_) => {
                        let mut c = match node.params.get(pd.name) {
                            Some(Value::Color(c)) => *c,
                            _ => [1.0; 3],
                        };
                        if ui.color_edit3(&label, &mut c) {
                            node.params.insert(pd.name.into(), Value::Color(c));
                            changed = true;
                        }
                    }
                    ParamKind::Prefab => {
                        let cur = match node.params.get(pd.name) {
                            Some(Value::Ref(s)) => prefab_names.iter().position(|o| o == s),
                            _ => None,
                        };
                        let mut i = cur.unwrap_or(0);
                        if !prefab_names.is_empty() && ui.combo_simple_string(&label, &mut i, &prefab_names) && Some(i) != cur {
                            node.params.insert(pd.name.into(), Value::Ref(prefab_names[i].clone()));
                            changed = true;
                            snapshot = true;
                        }
                    }
                    ParamKind::Curve => {
                        if let Some(Value::Curve(pts)) = node.params.get_mut(pd.name) {
                            if curve_widget(ui, "insp_curve", pts, [270.0, 180.0]) {
                                changed = true;
                            }
                            ui.text_disabled("drag points, double-click: add, right-click: remove");
                        }
                    }
                }
            }
            ui.separator();
            ui.input_text_multiline("Comment", &mut node.comment, [270.0, 60.0]).build();
        }
        if snapshot {
            self.snapshot(wg);
        }
        if changed {
            wg.mark_dirty();
        }
        ui.separator();
        self.make_function_button(ui, wg);
    }

    /// Collapse the selected density nodes into a new reusable function graph.
    fn make_function_button(&mut self, ui: &Ui, wg: &mut WorldgenSystem) {
        if self.selected.len() < 2 || self.current.0 == GraphKind::World {
            return;
        }
        if !ui.button("Group selection into function") {
            return;
        }
        let key = self.current.clone();
        self.snapshot(wg);
        let Some(g) = wg.project.get(key.0, &key.1).cloned() else { return };
        let sel: HashSet<u32> = self
            .selected
            .iter()
            .copied()
            .filter(|id| g.node(*id).and_then(|n| nodes::def(&n.kind)).map(|d| d.outputs.iter().all(|o| o.1 == PinType::Density) && !d.outputs.is_empty()).unwrap_or(false))
            .collect();
        // outgoing link: exactly one value leaves the selection
        let outgoing: Vec<Link> = g.links.iter().filter(|l| sel.contains(&l.from_node) && !sel.contains(&l.to_node)).copied().collect();
        let incoming: Vec<Link> = g.links.iter().filter(|l| !sel.contains(&l.from_node) && sel.contains(&l.to_node)).copied().collect();
        let sources: Vec<(u32, u8)> = {
            let mut v: Vec<(u32, u8)> = incoming.iter().map(|l| (l.from_node, l.from_pin)).collect();
            v.sort();
            v.dedup();
            v
        };
        let out_srcs: HashSet<(u32, u8)> = outgoing.iter().map(|l| (l.from_node, l.from_pin)).collect();
        if out_srcs.len() != 1 || sources.len() > 4 {
            self.status = "function needs exactly one output and at most 4 inputs".into();
            return;
        }
        let (out_node, out_pin) = *out_srcs.iter().next().unwrap();
        let mut n = 1;
        while wg.project.get(GraphKind::Function, &format!("{}_fn{n}", key.1)).is_some() {
            n += 1;
        }
        let fname = format!("{}_fn{n}", key.1);
        let mut f = Graph::new(&fname, GraphKind::Function);
        let mut remap = HashMap::new();
        for id in &sel {
            let node = g.node(*id).unwrap().clone();
            let nid = f.add(&node.kind, node.pos);
            if let Some(nn) = f.node_mut(nid) {
                nn.params = node.params.clone();
            }
            remap.insert(*id, nid);
        }
        for l in g.links.iter().filter(|l| sel.contains(&l.from_node) && sel.contains(&l.to_node)) {
            f.connect(remap[&l.from_node], l.from_pin, remap[&l.to_node], l.to_pin);
        }
        for (i, src) in sources.iter().enumerate() {
            let inp = f.add("function_input", [-300.0, i as f32 * 120.0]);
            f.set(inp, "index", Value::I(i as i32));
            for l in incoming.iter().filter(|l| (l.from_node, l.from_pin) == *src) {
                f.connect(inp, 0, remap[&l.to_node], l.to_pin);
            }
        }
        let fo = f.output_node().unwrap().id;
        f.connect(remap[&out_node], out_pin, fo, 0);
        wg.project.graphs.push(f);
        // replace in the original graph
        let g = wg.project.get_mut(key.0, &key.1).unwrap();
        let center = sel.iter().filter_map(|id| g.node(*id)).fold([0.0f32, 0.0], |a, n| [a[0] + n.pos[0], a[1] + n.pos[1]]);
        let cnt = sel.len().max(1) as f32;
        let call = g.add("function_call", [center[0] / cnt, center[1] / cnt]);
        g.set(call, "function", Value::Ref(fname.clone()));
        for (i, src) in sources.iter().enumerate() {
            g.connect(src.0, src.1, call, i as u8);
        }
        for l in &outgoing {
            g.connect(call, 0, l.to_node, l.to_pin);
        }
        for id in &sel {
            g.remove_node(*id);
        }
        self.placed.remove(&key);
        self.selected.clear();
        self.status = format!("created function '{fname}'");
        wg.mark_dirty();
    }
}

fn paste(graph: &mut Graph, nodes: &[Node], links: &[Link], offset: [f32; 2]) {
    let mut remap = HashMap::new();
    for n in nodes {
        let id = graph.add(&n.kind, [n.pos[0] + offset[0], n.pos[1] + offset[1]]);
        if let Some(nn) = graph.node_mut(id) {
            nn.params = n.params.clone();
            nn.comment = n.comment.clone();
        }
        remap.insert(n.id, id);
    }
    for l in links {
        if let (Some(&a), Some(&b)) = (remap.get(&l.from_node), remap.get(&l.to_node)) {
            graph.connect(a, l.from_pin, b, l.to_pin);
        }
    }
}

/// Same interpolation as `curve_eval` in shaders/worldgen/prelude.glsl.
fn curve_sample(pts: &[[f32; 2]], x: f32) -> f32 {
    if pts.is_empty() {
        return x;
    }
    if x <= pts[0][0] {
        return pts[0][1];
    }
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        if x <= b[0] {
            let mut t = (x - a[0]) / (b[0] - a[0]).max(1e-6);
            t = t * t * (3.0 - 2.0 * t);
            return a[1] + (b[1] - a[1]) * t;
        }
    }
    pts[pts.len() - 1][1]
}

thread_local! {
    static CURVE_DRAG: std::cell::RefCell<HashMap<String, i32>> = std::cell::RefCell::new(HashMap::new());
}

/// Inline spline editor. Returns true if any point changed.
pub fn curve_widget(ui: &Ui, id: &str, pts: &mut Vec<[f32; 2]>, size: [f32; 2]) -> bool {
    if pts.is_empty() {
        pts.push([0.0, 0.0]);
    }
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for p in pts.iter() {
        xmin = xmin.min(p[0]);
        xmax = xmax.max(p[0]);
        ymin = ymin.min(p[1]);
        ymax = ymax.max(p[1]);
    }
    let padx = ((xmax - xmin) * 0.1).max(0.1);
    let pady = ((ymax - ymin) * 0.15).max(0.1);
    let (xmin, xmax, ymin, ymax) = (xmin - padx, xmax + padx, ymin - pady, ymax + pady);
    let origin = ui.cursor_screen_pos();
    let to_screen = |p: [f32; 2]| [origin[0] + (p[0] - xmin) / (xmax - xmin) * size[0], origin[1] + size[1] - (p[1] - ymin) / (ymax - ymin) * size[1]];
    let to_curve = |s: [f32; 2]| [xmin + (s[0] - origin[0]) / size[0] * (xmax - xmin), ymin + (origin[1] + size[1] - s[1]) / size[1] * (ymax - ymin)];
    ui.invisible_button(format!("##{id}"), size);
    let hovered = ui.is_item_hovered();
    let active = ui.is_item_active();
    let mouse = ui.mouse_pos();
    let dl = ui.get_window_draw_list();
    dl.add_rect(origin, [origin[0] + size[0], origin[1] + size[1]], [0.1, 0.1, 0.12, 1.0]).filled(true).build();
    // zero lines
    if xmin < 0.0 && xmax > 0.0 {
        let a = to_screen([0.0, ymin]);
        let b = to_screen([0.0, ymax]);
        dl.add_line(a, b, [0.3, 0.3, 0.35, 1.0]).build();
    }
    if ymin < 0.0 && ymax > 0.0 {
        let a = to_screen([xmin, 0.0]);
        let b = to_screen([xmax, 0.0]);
        dl.add_line(a, b, [0.3, 0.3, 0.35, 1.0]).build();
    }
    let mut line = Vec::with_capacity(64);
    for i in 0..64 {
        let x = xmin + (xmax - xmin) * i as f32 / 63.0;
        line.push(to_screen([x, curve_sample(pts, x)]));
    }
    dl.add_polyline(line, [0.4, 0.8, 1.0, 1.0]).thickness(2.0).build();
    let nearest = pts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let s = to_screen(*p);
            (i, (s[0] - mouse[0]).powi(2) + (s[1] - mouse[1]).powi(2))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1));
    let mut changed = false;
    let drag_idx = CURVE_DRAG.with(|m| m.borrow().get(id).copied().unwrap_or(-1));
    let mut drag = drag_idx;
    if hovered && ui.is_mouse_clicked(MouseButton::Left)
        && let Some((i, d2)) = nearest
            && d2 < 100.0 {
                drag = i as i32;
            }
    if !active {
        drag = -1;
    }
    if drag >= 0 && (drag as usize) < pts.len() {
        let c = to_curve(mouse);
        let i = drag as usize;
        let lo = if i > 0 { pts[i - 1][0] + 1e-3 } else { f32::MIN };
        let hi = if i + 1 < pts.len() { pts[i + 1][0] - 1e-3 } else { f32::MAX };
        let nx = c[0].clamp(lo, hi);
        if (nx - pts[i][0]).abs() > 1e-6 || (c[1] - pts[i][1]).abs() > 1e-6 {
            pts[i] = [nx, c[1]];
            changed = true;
        }
    }
    CURVE_DRAG.with(|m| {
        m.borrow_mut().insert(id.to_string(), drag);
    });
    if hovered && ui.is_mouse_double_clicked(MouseButton::Left) {
        let c = to_curve(mouse);
        let pos = pts.iter().position(|p| p[0] > c[0]).unwrap_or(pts.len());
        pts.insert(pos, c);
        changed = true;
    }
    if hovered && ui.is_mouse_clicked(MouseButton::Right) && pts.len() > 2
        && let Some((i, d2)) = nearest
            && d2 < 100.0 {
                pts.remove(i);
                changed = true;
            }
    for (i, p) in pts.iter().enumerate() {
        let s = to_screen(*p);
        let col = if drag == i as i32 { [1.0, 0.8, 0.2, 1.0] } else { [0.9, 0.9, 0.9, 1.0] };
        dl.add_circle(s, 4.0, col).filled(true).build();
    }
    changed
}

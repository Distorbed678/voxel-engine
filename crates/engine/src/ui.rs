//! Dear ImGui debug UI.

use std::time::Instant;

use dear_imgui_rs::{Condition, TreeNodeFlags, Ui, WindowFlags};

use crate::atmosphere::Atmosphere;
use crate::blocks::{BlockKind, BlockRegistry};
use crate::camera::Camera;
use crate::engine::{FrameStats, PickHit};
use crate::world::World;

pub const DEBUG_VIEWS: [&str; 11] = [
    "Lit", "Normals", "LOD level", "Chunks", "Meshlets", "Ambient occlusion", "Albedo", "Overdraw", "Zones", "Biomes", "Lighting only",
];

pub struct UiState {
    pub show_ui: bool,
    pub show_stats: bool,
    pub show_world: bool,
    pub show_render: bool,
    pub show_camera: bool,
    pub show_blocks: bool,
    pub show_help: bool,
    pub show_demo: bool,
    pub show_map: bool,
    pub map_zones: bool,
    pub fog_scale: f32,
    pub debug_view: u32,
    pub wireframe: bool,
    pub fog: bool,
    pub shadows: bool,
    pub clouds: crate::renderer::CloudSettings,
    pub shadow_distance: f32,
    pub shadow_strength: f32,
    pub selected_block: u16,
    pub mouse_sensitivity: f32,
    pub jobs_per_frame: usize,
    pub gen_budget_ms: f32,
    pub force_reload: bool,
    toasts: Vec<(String, Instant)>,
    block_filter: String,
}

impl UiState {
    pub fn new(blocks: &BlockRegistry) -> Self {
        Self {
            show_ui: true,
            show_stats: true,
            show_world: true,
            show_render: true,
            show_camera: false,
            show_blocks: true,
            show_help: true,
            show_demo: false,
            show_map: true,
            map_zones: false,
            fog_scale: 1.0,
            debug_view: 0,
            wireframe: false,
            fog: true,
            shadows: true,
            clouds: Default::default(),
            shadow_distance: 320.0,
            shadow_strength: 0.85,
            selected_block: blocks.id("stone"),
            mouse_sensitivity: 1.0,
            jobs_per_frame: 256,
            gen_budget_ms: 6.0,
            force_reload: false,
            toasts: Vec::new(),
            block_filter: String::new(),
        }
    }

    pub fn toast(&mut self, msg: String) {
        log::info!("{msg}");
        self.toasts.push((msg, Instant::now()));
        if self.toasts.len() > 6 {
            self.toasts.remove(0);
        }
    }
}

#[derive(Default)]
pub struct UiActions {
    pub regenerate: bool,
    pub reset_world: bool,
    pub clear_edits: bool,
    pub screenshot: bool,
    pub set_vsync: Option<bool>,
    pub undo: bool,
    pub redo: bool,
    pub quit: bool,
    pub save_generator: bool,
    pub teleport: Option<(f64, f64)>,
    pub benchmark: bool,
    pub save_world: bool,
    pub load_world: bool,
}

pub struct UiCtx<'a> {
    pub state: &'a mut UiState,
    pub camera: &'a mut Camera,
    pub atmosphere: &'a mut Atmosphere,
    pub world: &'a mut World,
    pub blocks: &'a BlockRegistry,
    pub stats: &'a FrameStats,
    pub gpu_times: &'a [(&'static str, f32)],
    pub pick: Option<PickHit>,
    pub fly_mode: bool,
    pub device_name: &'a str,
    pub seed: &'a mut u32,
    pub shader_errors: &'a [String],
    pub pool_capacity: u32,
    pub batch: (u32, u32),
    pub busy_slots: usize,
    pub vsync: Option<bool>,
    pub node_editor_open: &'a mut bool,
    pub tools_open: &'a mut bool,
    pub bench_report: Option<String>,
    pub probes: &'a crate::probe::Probes,
    pub biome_names: &'a [worldgen::BiomeInfo],
    pub zone_names: &'a [worldgen::ZoneInfo],
}

pub fn apply_style(ctx: &mut dear_imgui_rs::Context) {
    let style = ctx.style_mut();
    style.set_window_rounding(6.0);
    style.set_frame_rounding(4.0);
    style.set_grab_rounding(4.0);
    style.set_window_border_size(1.0);
}

pub fn draw(ui: &Ui, c: &mut UiCtx) -> UiActions {
    let mut a = UiActions::default();
    if !c.state.show_ui {
        draw_toasts(ui, c.state);
        return a;
    }

    ui.main_menu_bar(|| {
        ui.menu("File", || {
            if ui.menu_item("Save world generator") {
                a.save_generator = true;
            }
            if ui.menu_item("Save world edits") {
                a.save_world = true;
            }
            if ui.menu_item("Load world edits") {
                a.load_world = true;
            }
            if ui.menu_item("Run benchmark") {
                a.benchmark = true;
            }
            if ui.menu_item("Screenshot (F2)") {
                a.screenshot = true;
            }
            if ui.menu_item("Quit") {
                a.quit = true;
            }
        });
        ui.menu("Edit", || {
            if ui.menu_item("Undo (Ctrl+Z)") {
                a.undo = true;
            }
            if ui.menu_item("Redo (Ctrl+Y)") {
                a.redo = true;
            }
            if ui.menu_item("Clear all edits") {
                a.clear_edits = true;
            }
        });
        ui.menu("View", || {
            ui.checkbox("World Generator (node editor)", c.node_editor_open);
            ui.checkbox("Tools", c.tools_open);
            ui.checkbox("Stats", &mut c.state.show_stats);
            ui.checkbox("World", &mut c.state.show_world);
            ui.checkbox("Render", &mut c.state.show_render);
            ui.checkbox("Camera", &mut c.state.show_camera);
            ui.checkbox("Blocks", &mut c.state.show_blocks);
            ui.checkbox("Help", &mut c.state.show_help);
            ui.checkbox("Minimap", &mut c.state.show_map);
            ui.checkbox("ImGui demo", &mut c.state.show_demo);
        });
        ui.text_colored([0.6, 0.8, 1.0, 1.0], format!("  {:.0} fps  {:.2} ms", c.stats.fps, 1000.0 / c.stats.fps.max(1.0)));
        if c.fly_mode {
            ui.text_colored([1.0, 0.8, 0.3, 1.0], "  FLY MODE (Tab/Esc to release)");
        }
    });

    let ds = ui.io().display_size();
    if c.state.show_stats {
        let mut open = true;
        ui.window("Stats").size([330.0, 400.0], Condition::FirstUseEver).position([10.0, 30.0], Condition::FirstUseEver).opened(&mut open).build(|| {
            ui.text(c.device_name);
            ui.text(format!("{:.0} fps | frame {:.2} ms | cpu {:.2} ms", c.stats.fps, 1000.0 / c.stats.fps.max(1.0), c.stats.cpu_ms));
            ui.plot_lines("##frametimes", &c.stats.frame_ms_history);
            if ui.collapsing_header("GPU timings", TreeNodeFlags::DEFAULT_OPEN) {
                for (name, ms) in c.gpu_times {
                    ui.text(format!("{name:<14} {ms:7.3} ms"));
                }
            }
            if ui.collapsing_header("World", TreeNodeFlags::DEFAULT_OPEN) {
                let s = &c.world.stats;
                ui.text(format!("chunks {} (meshed {}, empty {})", s.chunks, s.meshed, s.empty));
                ui.text(format!("pending {} | in flight {} | retiring {}", s.pending, s.in_flight, s.retiring));
                ui.text(format!("jobs/s {} | total {}", s.jobs_last_second, s.jobs_total));
                ui.text(format!("batch quads {} | overflow {} | busy slots {}", c.batch.0, c.batch.1, c.busy_slots));
                let used = s.quads_resident as f32 / c.pool_capacity as f32;
                ui.text(format!("quad pool {:.1}M / {:.1}M", s.quads_resident as f32 / 1.0e6, c.pool_capacity as f32 / 1.0e6));
                ui.progress_bar(used).build();
                for (lod, (meshed, pending)) in s.per_lod.iter().enumerate().take(c.world.settings.levels as usize) {
                    ui.text(format!("  LOD{lod}: {meshed} meshed, {pending} pending"));
                }
                ui.text(format!("edits {} (undo {})", c.world.overlay.edit_count(), c.world.overlay.undo.len()));
            }
            if ui.collapsing_header("Draw", TreeNodeFlags::DEFAULT_OPEN) {
                ui.text(format!("visible chunks {}", c.stats.visible_chunks));
                ui.text(format!("quads drawn {:.2}M ({:.2}M tris)", c.stats.drawn_quads as f32 / 1e6, c.stats.drawn_quads as f32 * 2.0 / 1e6));
                ui.text(format!("task groups {} opaque / {} transparent / {} model", c.stats.draw_groups, c.stats.trans_groups, c.stats.model_groups));
            }
            if let Some(p) = c.pick {
                ui.separator();
                let name = c.blocks.blocks.get(p.block as usize).map(|b| b.display).unwrap_or("?");
                ui.text(format!("looking at {} @ {} {} {}", name, p.block_pos.x, p.block_pos.y, p.block_pos.z));
                ui.text(format!("normal {:?} lod {} dist {:.1}", p.normal.to_array(), p.lod, p.distance));
            }
            if !c.shader_errors.is_empty() {
                ui.separator();
                ui.text_colored([1.0, 0.4, 0.4, 1.0], "Shader errors:");
                for e in c.shader_errors {
                    ui.text_wrapped(e);
                }
            }
        });
        c.state.show_stats = open;
    }

    if c.state.show_render {
        let mut open = true;
        ui.window("Render").size([330.0, 300.0], Condition::FirstUseEver).position([10.0, 440.0], Condition::FirstUseEver).opened(&mut open).build(|| {
            let mut dv = c.state.debug_view as usize;
            if ui.combo_simple_string("Debug view", &mut dv, &DEBUG_VIEWS) {
                c.state.debug_view = dv as u32;
            }
            ui.checkbox("Wireframe", &mut c.state.wireframe);
            ui.checkbox("Fog", &mut c.state.fog);
            ui.same_line();
            ui.checkbox("Shadows", &mut c.state.shadows);
            ui.slider("Shadow distance", 64.0f32, 1024.0f32, &mut c.state.shadow_distance);
            ui.slider("Shadow strength", 0.0f32, 1.0f32, &mut c.state.shadow_strength);
            ui.checkbox("Clouds", &mut c.state.clouds.enabled);
            if c.state.clouds.enabled {
                ui.slider("Cloud coverage", 0.0f32, 1.0f32, &mut c.state.clouds.coverage);
                ui.slider("Cloud height", 100.0f32, 600.0f32, &mut c.state.clouds.base);
                ui.slider("Cloud thickness", 20.0f32, 300.0f32, &mut c.state.clouds.thickness);
                ui.slider("Cloud density", 0.005f32, 0.3f32, &mut c.state.clouds.density);
            }
            if let Some(vs) = c.vsync {
                let mut v = vs;
                if ui.checkbox("VSync", &mut v) {
                    a.set_vsync = Some(v);
                }
            }
            ui.separator();
            ui.slider("Time of day", 0.0f32, 1.0f32, &mut c.atmosphere.time_of_day);
            let mut running = !c.atmosphere.paused;
            if ui.checkbox("Day/night cycle", &mut running) {
                c.atmosphere.paused = !running;
            }
            ui.slider("Day length (s)", 30.0f32, 3600.0f32, &mut c.atmosphere.day_length_s);
            ui.slider("Exposure", 0.1f32, 4.0f32, &mut c.atmosphere.exposure);
            ui.slider("Fog amount", 0.0f32, 4.0f32, &mut c.state.fog_scale);
            ui.slider("Fog height falloff", 0.0f32, 0.1f32, &mut c.atmosphere.fog_height_falloff);
            if ui.button("Reload shaders (F5)") {
                c.state.force_reload = true;
            }
        });
        c.state.show_render = open;
    }

    if c.state.show_world {
        let mut open = true;
        ui.window("World").size([300.0, 290.0], Condition::FirstUseEver).position([ds[0] - 310.0, 30.0], Condition::FirstUseEver).opened(&mut open).build(|| {
            let mut seed = *c.seed as i32;
            if ui.input_int("Seed", &mut seed) {
                *c.seed = seed as u32;
                a.regenerate = true;
            }
            ui.same_line();
            if ui.button("Random") {
                *c.seed = (Instant::now().elapsed().as_nanos() as u32).wrapping_mul(2654435761) ^ c.stats.frame_ms_history.len() as u32 ^ 0x9e3779b9;
                *c.seed = c.seed.wrapping_add((c.stats.cpu_ms * 1.0e6) as u32);
                a.regenerate = true;
            }
            if ui.button("Regenerate") {
                a.regenerate = true;
            }
            ui.same_line();
            if ui.button("Reset streaming") {
                a.reset_world = true;
            }
            ui.separator();
            ui.text("View distance");
            let mut s = c.world.settings;
            let mut levels = s.levels as i32;
            let mut r = s.radius / 2;
            let mut rv = s.radius_v / 2;
            let mut changed = ui.slider("LOD levels", 1, 7, &mut levels);
            changed |= ui.slider("Radius (x2 chunks)", 2, 12, &mut r);
            changed |= ui.slider("Vertical radius (x2)", 1, 8, &mut rv);
            if changed {
                s.levels = levels as u32;
                s.radius = r * 2;
                s.radius_v = rv * 2;
                c.world.settings = s;
            }
            let reach = (32 << (s.levels - 1)) as f32 * s.radius as f32;
            ui.text(format!("horizon ~{:.1} km", reach / 1000.0));
            let mut jobs = c.state.jobs_per_frame as i32;
            if ui.slider("Chunk jobs / frame", 16, 256, &mut jobs) {
                c.state.jobs_per_frame = jobs as usize;
            }
            ui.slider("Generation budget (ms/frame)", 1.0f32, 30.0f32, &mut c.state.gen_budget_ms);
            ui.checkbox("Freeze streaming", &mut c.world.frozen);
        });
        c.state.show_world = open;
    }

    if c.state.show_camera {
        let mut open = true;
        ui.window("Camera").size([300.0, 160.0], Condition::FirstUseEver).opened(&mut open).build(|| {
            let mut p = [c.camera.pos.x as f32, c.camera.pos.y as f32, c.camera.pos.z as f32];
            if ui.input_float3("Position", &mut p).build() {
                c.camera.pos = glam::DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
            }
            ui.slider("Speed", 1.0f32, 2000.0f32, &mut c.camera.speed);
            ui.slider("FOV", 30.0f32, 120.0f32, &mut c.camera.fov_y_deg);
            ui.slider("Mouse sensitivity", 0.1f32, 4.0f32, &mut c.state.mouse_sensitivity);
            ui.text(format!("yaw {:.2} pitch {:.2}", c.camera.yaw, c.camera.pitch));
        });
        c.state.show_camera = open;
    }

    if c.state.show_blocks {
        let mut open = true;
        ui.window("Blocks").size([300.0, 300.0], Condition::FirstUseEver).position([ds[0] - 310.0, 330.0], Condition::FirstUseEver).opened(&mut open).build(|| {
            let sel = &c.blocks.blocks[c.state.selected_block as usize];
            ui.text(format!("Selected: {}", sel.display));
            ui.input_text("Filter", &mut c.state.block_filter).build();
            let filter = c.state.block_filter.to_lowercase();
            let mut cat = "";
            for (i, b) in c.blocks.blocks.iter().enumerate().skip(1) {
                if !filter.is_empty() && !b.display.to_lowercase().contains(&filter) {
                    continue;
                }
                if b.category != cat {
                    cat = b.category;
                    ui.separator();
                    ui.text_colored([0.7, 0.7, 0.7, 1.0], cat);
                }
                let label = match b.kind {
                    BlockKind::Model => format!("{} (model)##{i}", b.display),
                    _ => format!("{}##{i}", b.display),
                };
                if ui.selectable_config(&label).selected(c.state.selected_block as usize == i).build() {
                    c.state.selected_block = i as u16;
                }
            }
        });
        c.state.show_blocks = open;
    }

    if c.state.show_help {
        let mut open = true;
        ui.window("Controls").size([330.0, 230.0], Condition::FirstUseEver).position([350.0, 30.0], Condition::FirstUseEver).collapsed(true, Condition::FirstUseEver).opened(&mut open).build(|| {
            ui.text("Tab      toggle fly mode (mouse look)");
            ui.text("Esc      release mouse");
            ui.text("WASD     move,  Space/E up,  Q/Ctrl down");
            ui.text("Shift    fast,  Alt slow,  wheel = speed");
            ui.text("RMB drag look around (editor mode)");
            ui.separator();
            ui.text("Fly mode:  LMB break, RMB place, MMB pick");
            ui.text("Editor:    LMB break, Shift+LMB place");
            ui.text("Ctrl+Z / Ctrl+Y   undo / redo");
            ui.text("F1 hide UI, F2 screenshot, F5 reload shaders");
        });
        c.state.show_help = open;
    }

    if c.state.show_map {
        minimap(ui, c, &mut a);
    }
    if let Some(r) = &c.bench_report {
        ui.window("Benchmark result").size([460.0, 260.0], Condition::FirstUseEver).build(|| {
            ui.text(r);
        });
    }

    if c.state.show_demo {
        ui.show_demo_window(&mut c.state.show_demo);
    }

    draw_toasts(ui, c.state);
    a
}

fn minimap(ui: &Ui, c: &mut UiCtx, a: &mut UiActions) {
    let ds = ui.io().display_size();
    let mut open = c.state.show_map;
    ui.window("Minimap")
        .size([300.0, 380.0], Condition::FirstUseEver)
        .position([ds[0] - 310.0, 640.0], Condition::FirstUseEver)
        .opened(&mut open)
        .build(|| {
            if let Some(cam) = c.probes.camera {
                let b = c.biome_names.get(cam.biome as usize).map(|b| b.name.as_str()).unwrap_or("?");
                let z = c.zone_names.get(cam.zone as usize).map(|z| z.name.as_str()).unwrap_or("?");
                ui.text(format!("Zone: {z}   Biome: {b}"));
            }
            ui.checkbox("Zones", &mut c.state.map_zones);
            ui.same_line();
            ui.text_disabled("click to teleport");
            let Some(((ox, oz, sp), cells)) = &c.probes.map else {
                ui.text("waiting for probes...");
                return;
            };
            let n = crate::probe::MAP_N;
            let avail = ui.content_region_avail();
            let size = avail[0].min(avail[1]).max(64.0);
            let cell = size / n as f32;
            let p0 = ui.cursor_screen_pos();
            ui.invisible_button("##map", [size, size]);
            let clicked = ui.is_item_clicked();
            let mouse = ui.mouse_pos();
            let dl = ui.get_window_draw_list();
            let hs: Vec<f32> = cells.iter().map(|r| r.height).filter(|h| *h > -1000.0).collect();
            let (hmin, hmax) = hs.iter().fold((f32::MAX, f32::MIN), |(a, b), h| (a.min(*h), b.max(*h)));
            for j in 0..n {
                for i in 0..n {
                    let r = cells[j * n + i];
                    let col = if c.state.map_zones {
                        c.zone_names.get(r.zone as usize).map(|z| z.map_color).unwrap_or([1.0, 0.0, 1.0])
                    } else {
                        c.biome_names.get(r.biome as usize).map(|b| b.map_color).unwrap_or([1.0, 0.0, 1.0])
                    };
                    let shade = if r.height > -1000.0 && hmax > hmin { 0.65 + 0.35 * (r.height - hmin) / (hmax - hmin) } else { 0.6 };
                    let water = r.height > -1000.0 && r.height < 62.0;
                    let col = if water { [0.15, 0.3, 0.6] } else { col };
                    let x = p0[0] + i as f32 * cell;
                    let y = p0[1] + j as f32 * cell;
                    dl.add_rect([x, y], [x + cell + 0.5, y + cell + 0.5], [col[0] * shade, col[1] * shade, col[2] * shade, 1.0]).filled(true).build();
                }
            }
            // camera marker
            let cx = p0[0] + ((c.camera.pos.x - ox) / sp) as f32 * cell;
            let cz = p0[1] + ((c.camera.pos.z - oz) / sp) as f32 * cell;
            let f = c.camera.forward();
            dl.add_circle([cx, cz], 4.0, [1.0, 1.0, 1.0, 1.0]).filled(true).build();
            dl.add_line([cx, cz], [cx + f.x * 14.0, cz + f.z * 14.0], [1.0, 0.2, 0.2, 1.0]).thickness(2.0).build();
            if clicked {
                let wx = ox + ((mouse[0] - p0[0]) / cell) as f64 * sp;
                let wz = oz + ((mouse[1] - p0[1]) / cell) as f64 * sp;
                a.teleport = Some((wx, wz));
            }
            ui.text(format!("{:.1} km across", (n as f64 * sp) / 1000.0));
        });
    c.state.show_map = open;
}

fn draw_toasts(ui: &Ui, state: &mut UiState) {
    state.toasts.retain(|(_, t)| t.elapsed().as_secs_f32() < 3.0);
    if state.toasts.is_empty() {
        return;
    }
    let size = ui.io().display_size();
    ui.window("##toasts")
        .position([size[0] - 330.0, size[1] - 30.0 - state.toasts.len() as f32 * 22.0], Condition::Always)
        .size([320.0, 0.0], Condition::Always)
        .flags(WindowFlags::NO_DECORATION | WindowFlags::NO_INPUTS | WindowFlags::NO_SAVED_SETTINGS | WindowFlags::ALWAYS_AUTO_RESIZE)
        .bg_alpha(0.6)
        .build(|| {
            for (m, _) in &state.toasts {
                ui.text(m);
            }
        });
}

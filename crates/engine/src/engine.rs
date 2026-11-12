//! Frame orchestration: owns the Vulkan context and every subsystem.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result};
use glam::{DVec3, IVec3, Mat4, Vec3, Vec4};
use vkcore::context::ContextDesc;
use vkcore::image::ImageDesc;
use vkcore::sync::{Timeline, image_barrier, memory_barrier};
use vkcore::{Bindless, Buffer, GpuProfiler, Image, Swapchain, VkContext, vk};
use winit::keyboard::KeyCode;
use winit::window::{CursorGrabMode, Window};

use crate::atmosphere::Atmosphere;
use crate::blocks::BlockRegistry;
use crate::camera::Camera;
use crate::input::Input;
use crate::renderer::{FLAG_CROSSHAIR, FLAG_FOG, FrameData, Renderer};
use crate::shaders::{PipelineHandle, PipelineManager, PipelineSource};
use crate::ui::{self, UiActions, UiState};
use crate::world::World;
use crate::world::gpu::{MAX_JOBS, WorldGpu, WorldPipelines};
use crate::node_editor::NodeEditorUi;
use crate::wg::WorldgenSystem;

pub const FIF: usize = 2;
pub const OUTPUT_FORMAT_HEADLESS: vk::Format = vk::Format::B8G8R8A8_UNORM;

#[derive(Clone, Debug, Default)]
pub struct Options {
    pub headless: bool,
    pub width: u32,
    pub height: u32,
    pub frames: Option<u64>,
    pub screenshot: Option<PathBuf>,
    pub validation: bool,
    pub cam_pos: Option<DVec3>,
    pub cam_angles: Option<(f32, f32)>,
    pub no_ui: bool,
    pub time_of_day: Option<f32>,
    pub debug_view: Option<u32>,
    pub seed: Option<u32>,
    pub worldgen_dir: Option<PathBuf>,
    pub benchmark: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PickHit {
    pub block_pos: IVec3,
    pub normal: IVec3,
    pub block: u16,
    pub lod: u32,
    pub world: DVec3,
    pub distance: f32,
}

struct FrameRes {
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    acquire: vk::Semaphore,
    cam_pos: DVec3,
    inv_view_proj: Mat4,
    pick_px: (u32, u32),
    extent: vk::Extent2D,
    screenshot: Option<(PathBuf, Buffer, vk::Extent2D)>,
}

pub struct ImguiLayer {
    pub ctx: dear_imgui_rs::Context,
    pub platform: Option<dear_imgui_winit::WinitPlatform>,
    pub renderer: Option<dear_imgui_ash::AshRenderer>,
    pool: vk::CommandPool,
}

#[derive(Default, Clone)]
pub struct FrameStats {
    pub cpu_ms: f32,
    pub fps: f32,
    pub frame_ms_history: Vec<f32>,
    pub visible_chunks: u32,
    pub drawn_quads: u32,
    pub draw_groups: u32,
    pub trans_groups: u32,
    pub model_groups: u32,
}

pub struct Engine {
    pub opts: Options,
    pub window: Option<Arc<Window>>,
    swapchain: Option<Swapchain>,
    offscreen: Option<Image>,
    frames: Vec<FrameRes>,
    timeline: Timeline,
    pub frame_counter: u64,
    pub bindless: Bindless,
    pub pm: PipelineManager,
    pub profiler: GpuProfiler,
    pub renderer: Renderer,
    pub wgpu: WorldGpu,
    pub world: World,
    pub blocks: BlockRegistry,
    gen_pipeline: PipelineHandle,
    props_pipeline: PipelineHandle,
    probe_pipeline: PipelineHandle,
    pub probes: crate::probe::Probes,
    pub wg: WorldgenSystem,
    pub node_ui: NodeEditorUi,
    pub tools: crate::tools::Tools,
    prefab_revision: u64,
    pub camera: Camera,
    pub atmosphere: Atmosphere,
    pub input: Input,
    pub ui: UiState,
    pub imgui: ImguiLayer,
    pub seed: u32,
    start: Instant,
    last_frame: Instant,
    pub dt: f32,
    pub pick: Option<PickHit>,
    pub stats: FrameStats,
    pub fly_mode: bool,
    look_drag: bool,
    resize_pending: bool,
    pub quit: bool,
    screenshot_request: Option<PathBuf>,
    prev_view_proj: Mat4,
    /// accumulated GPU scope timings for headless benchmark summaries
    pub gpu_sums: std::collections::BTreeMap<&'static str, (f64, u32)>,
    pub bench: Option<crate::bench::Benchmark>,
    /// jobs recorded in each frame slot (to relate GPU timings to batch size)
    batch_sizes: [usize; FIF],
    /// EMA of GPU milliseconds per generated chunk (gen + props + meshing)
    pub ms_per_job: f32,
    ctx: VkContext,
}

/// Full GLSL sources (terrain pass, props pass, probe pass) around the generated code.
pub fn compose_source(generated: &str) -> (String, String, String) {
    let wrap = |main: &str| {
        format!("#version 460\n#include \"worldgen/prelude.glsl\"\n// ---- generated ----\n{generated}\n// ---- main ----\n#include \"{main}\"\n")
    };
    (wrap("worldgen/terrain_main.glsl"), wrap("worldgen/props_main.glsl"), wrap("worldgen/probe_main.glsl"))
}

impl Engine {
    pub fn new(opts: Options, window: Option<Arc<Window>>) -> Result<Self> {
        let display = window.as_ref().map(|w| {
            use raw_window_handle::HasDisplayHandle;
            w.display_handle().unwrap().as_raw()
        });
        let ctx = VkContext::new(ContextDesc { display, validation: opts.validation, app_name: "voxel-engine" })?;

        let shader_root = find_shader_root()?;
        let blocks = BlockRegistry::new();
        let gen_dir = shader_root.join("generated");
        std::fs::create_dir_all(&gen_dir)?;
        let header = blocks.glsl_header();
        let header_path = gen_dir.join("blocks.glsl");
        if std::fs::read_to_string(&header_path).ok().as_deref() != Some(header.as_str()) {
            std::fs::write(&header_path, header)?;
        }

        let mut bindless = Bindless::new(&ctx)?;
        let layout = vkcore::pipeline::create_global_layout(&ctx, &[bindless.layout])?;
        let mut pm = PipelineManager::new(shader_root, layout)?;

        // ---- output target ----
        let (swapchain, extent, out_format) = if let Some(w) = &window {
            use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
            let surface = unsafe {
                ash_window::create_surface(
                    &ctx.entry,
                    &ctx.instance,
                    w.display_handle()?.as_raw(),
                    w.window_handle()?.as_raw(),
                    None,
                )?
            };
            let size = w.inner_size();
            let sc = Swapchain::new(&ctx, surface, size.width.max(1), size.height.max(1), false)?;
            let e = sc.extent;
            let f = sc.format;
            (Some(sc), e, f)
        } else {
            (None, vk::Extent2D { width: opts.width, height: opts.height }, OUTPUT_FORMAT_HEADLESS)
        };
        let offscreen = if window.is_none() {
            Some(Image::new(
                &ctx,
                &ImageDesc::d2(
                    "offscreen_output",
                    OUTPUT_FORMAT_HEADLESS,
                    extent.width,
                    extent.height,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                ),
            )?)
        } else {
            None
        };

        let renderer = Renderer::new(&ctx, &mut bindless, &mut pm, &blocks, extent, out_format, FIF)?;
        let block_ids = blocks.blocks.iter().enumerate().map(|(i, b)| (b.name.to_string(), i as u32)).collect();
        let mut prefabs = crate::prefab::PrefabLibrary::load(PathBuf::from("assets/prefabs"), &blocks);
        prefabs.upload(&ctx)?;
        let mut wg = WorldgenSystem::new(&ctx, &pm.compiler.root.clone(), pm.layout, block_ids, prefabs.compile_info(), opts.worldgen_dir.clone().unwrap_or_else(|| PathBuf::from("assets/worldgen")))?;
        // Build the generator synchronously at startup (fall back to the built-in one on error).
        let (terrain_src, props_src, probe_src) = wg.sources();
        let t0 = Instant::now();
        let src = |n: &str, s: String| PipelineSource::ComputeSource { name: format!("{n}.comp"), source: s, defines: vec![] };
        let (gen_pipeline, props_pipeline, probe_pipeline) = match (
            pm.create(&ctx, "worldgen_terrain", src("worldgen_terrain", terrain_src)),
            pm.create(&ctx, "worldgen_props", src("worldgen_props", props_src)),
            pm.create(&ctx, "worldgen_probe", src("worldgen_probe", probe_src)),
        ) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            (a, b, c) => {
                log::error!("world generator failed to build, using defaults: {:?} {:?} {:?}", a.err(), b.err(), c.err());
                wg.reset_to_defaults();
                wg.update();
                let (t, p, q) = wg.sources();
                (
                    pm.create(&ctx, "worldgen_terrain", src("worldgen_terrain", t))?,
                    pm.create(&ctx, "worldgen_props", src("worldgen_props", p))?,
                    pm.create(&ctx, "worldgen_probe", src("worldgen_probe", q))?,
                )
            }
        };
        wg.active_source = wg.compiled.source.clone();
        wg.dirty = false;
        wg.request_optimized();
        log::info!(
            "world generator: {} biomes, {} zones, {} prop layers, {} params, shaders built in {:.0} ms",
            wg.compiled.biomes.len(),
            wg.compiled.zones.len(),
            wg.compiled.prop_layers,
            wg.compiled.params.len(),
            t0.elapsed().as_secs_f32() * 1000.0
        );
        let quad_capacity = 64 << 20; // 512 MiB of packed quads
        let max_records = 1 << 16;
        let mut wgpu = WorldGpu::new(&ctx, FIF, quad_capacity, max_records)?;
        wgpu.set_params(&wg.compiled.params);
        wgpu.prefab_addrs = (prefabs.table.as_ref().unwrap().address, prefabs.data.as_ref().unwrap().address);
        let prefab_revision = prefabs.revision;
        let world = World::new(quad_capacity, max_records);
        let profiler = GpuProfiler::new(&ctx, FIF)?;
        let timeline = Timeline::new(&ctx)?;

        let mut frames = Vec::new();
        for i in 0..FIF {
            let pool = unsafe {
                ctx.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(ctx.queue_family)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )?
            };
            let cb = unsafe {
                ctx.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1),
                )?[0]
            };
            ctx.set_name(cb, &format!("frame_cb_{i}"));
            let acquire = unsafe { ctx.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
            frames.push(FrameRes {
                pool,
                cb,
                acquire,
                cam_pos: DVec3::ZERO,
                inv_view_proj: Mat4::IDENTITY,
                pick_px: (0, 0),
                extent,
                screenshot: None,
            });
        }

        // ---- Dear ImGui ----
        let mut ictx = dear_imgui_rs::Context::create();
        let _ = ictx.set_ini_filename(Some(PathBuf::from("imgui.ini")));
        {
            let io = ictx.io_mut();
            io.set_config_flags(io.config_flags() | dear_imgui_rs::ConfigFlags::DOCKING_ENABLE);
        }
        ui::apply_style(&mut ictx);
        let platform = if let Some(w) = &window {
            let mut p = dear_imgui_winit::WinitPlatform::new(&mut ictx)?;
            p.attach_window(w.clone(), dear_imgui_winit::HiDpiMode::Default, &mut ictx)?;
            Some(p)
        } else {
            let io = ictx.io_mut();
            io.set_display_size([extent.width as f32, extent.height as f32]);
            None
        };
        let imgui_pool = unsafe {
            ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let config = dear_imgui_ash::AshRendererConfig::with_dynamic_rendering(
            ctx.device.clone(),
            ctx.queue,
            imgui_pool,
            dear_imgui_ash::DynamicRendering { color_attachment_format: out_format, depth_attachment_format: None },
        )
        .with_options(dear_imgui_ash::Options { in_flight_frames: FIF, ..Default::default() });
        let imgui_renderer =
            unsafe { dear_imgui_ash::AshRenderer::with_default_allocator(&ctx.instance, ctx.physical_device, config, &mut ictx)? };

        let mut camera = Camera::default();
        if let Some(p) = opts.cam_pos {
            camera.pos = p;
        }
        if let Some((yaw, pitch)) = opts.cam_angles {
            camera.yaw = yaw;
            camera.pitch = pitch;
        }
        let mut atmosphere = Atmosphere::default();
        if let Some(t) = opts.time_of_day {
            atmosphere.time_of_day = t;
        }
        let mut ui_state = UiState::new(&blocks);
        if let Some(d) = opts.debug_view {
            ui_state.debug_view = d;
        }
        ui_state.show_ui = !opts.no_ui;

        let seed = opts.seed.unwrap_or(1337);
        Ok(Self {
            opts,
            window,
            swapchain,
            offscreen,
            frames,
            timeline,
            frame_counter: 0,
            bindless,
            pm,
            profiler,
            renderer,
            wgpu,
            world,
            blocks,
            gen_pipeline,
            props_pipeline,
            probe_pipeline,
            probes: crate::probe::Probes::new(&ctx, FIF)?,
            wg,
            node_ui: NodeEditorUi::new(),
            tools: crate::tools::Tools::new(prefabs),
            prefab_revision,
            camera,
            atmosphere,
            input: Input::default(),
            ui: ui_state,
            imgui: ImguiLayer { ctx: ictx, platform, renderer: Some(imgui_renderer), pool: imgui_pool },
            seed,
            start: Instant::now(),
            last_frame: Instant::now(),
            dt: 0.016,
            pick: None,
            stats: FrameStats::default(),
            fly_mode: false,
            look_drag: false,
            resize_pending: false,
            quit: false,
            screenshot_request: None,
            prev_view_proj: Mat4::IDENTITY,
            gpu_sums: Default::default(),
            bench: None,
            batch_sizes: [0; FIF],
            ms_per_job: 0.05,
            ctx,
        })
    }

    pub fn ctx(&self) -> &VkContext {
        &self.ctx
    }

    // ---------------------------------------------------------------------------------------
    // input
    // ---------------------------------------------------------------------------------------

    pub fn handle_window_event(&mut self, event: &winit::event::WindowEvent) {
        use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
        let mut imgui_wants_mouse = false;
        let mut imgui_wants_kb = false;
        if let (Some(p), Some(w)) = (&mut self.imgui.platform, &self.window) {
            let forward = match event {
                WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => {
                    !self.fly_mode && !self.look_drag
                }
                WindowEvent::KeyboardInput { .. } => !self.fly_mode,
                _ => true,
            };
            if forward {
                let _ = p.handle_window_event(&mut self.imgui.ctx, w.id(), event);
            }
            let io = self.imgui.ctx.io();
            imgui_wants_mouse = io.want_capture_mouse();
            imgui_wants_kb = io.want_capture_keyboard();
        }
        match event {
            WindowEvent::Resized(_) => self.resize_pending = true,
            WindowEvent::KeyboardInput { event, .. } => {
                if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        if imgui_wants_kb && !self.fly_mode && code != KeyCode::Escape {
                            return;
                        }
                        if !event.repeat {
                            self.input.pressed.insert(code);
                        }
                        self.input.held.insert(code);
                    } else {
                        self.input.held.remove(&code);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let idx = match button {
                    MouseButton::Left => 0,
                    MouseButton::Right => 1,
                    MouseButton::Middle => 2,
                    _ => return,
                };
                if *state == ElementState::Pressed {
                    if imgui_wants_mouse && !self.fly_mode && !self.look_drag {
                        return;
                    }
                    self.input.mouse_pressed[idx] = true;
                    self.input.mouse_held[idx] = true;
                } else {
                    self.input.mouse_held[idx] = false;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if imgui_wants_mouse && !self.fly_mode {
                    return;
                }
                self.input.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor = (position.x, position.y);
            }
            WindowEvent::Focused(false) => {
                self.set_fly_mode(false);
                self.input.held.clear();
            }
            _ => {}
        }
    }

    pub fn handle_mouse_motion(&mut self, delta: (f64, f64)) {
        if self.fly_mode || self.look_drag {
            self.input.mouse_delta.0 += delta.0;
            self.input.mouse_delta.1 += delta.1;
        }
    }

    fn set_cursor_grab(&self, grab: bool) {
        if let Some(w) = &self.window {
            if grab {
                if w.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                    let _ = w.set_cursor_grab(CursorGrabMode::Confined);
                }
                w.set_cursor_visible(false);
            } else {
                let _ = w.set_cursor_grab(CursorGrabMode::None);
                w.set_cursor_visible(true);
            }
        }
    }

    pub fn set_fly_mode(&mut self, on: bool) {
        if self.fly_mode == on {
            return;
        }
        self.fly_mode = on;
        self.set_cursor_grab(on);
    }

    fn update_camera(&mut self, dt: f32) {
        let inp = &self.input;
        // RMB-drag mouse look in editor mode
        if !self.fly_mode {
            let want = inp.mouse_held[1];
            if want != self.look_drag {
                self.look_drag = want;
                self.set_cursor_grab(want);
            }
        }
        let inp = &self.input;
        if self.fly_mode || self.look_drag {
            let sens = 0.0022 * self.ui.mouse_sensitivity;
            self.camera.yaw -= inp.mouse_delta.0 as f32 * sens;
            self.camera.pitch = (self.camera.pitch - inp.mouse_delta.1 as f32 * sens).clamp(-1.55, 1.55);
        }
        if inp.scroll != 0.0 && (self.fly_mode || !inp.ctrl()) {
            self.camera.speed = (self.camera.speed * 1.15f32.powf(inp.scroll)).clamp(1.0, 4000.0);
        }
        let mut v = Vec3::ZERO;
        let f = self.camera.forward();
        let r = self.camera.right();
        if inp.key(KeyCode::KeyW) {
            v += f;
        }
        if inp.key(KeyCode::KeyS) {
            v -= f;
        }
        if inp.key(KeyCode::KeyD) {
            v += r;
        }
        if inp.key(KeyCode::KeyA) {
            v -= r;
        }
        if inp.key(KeyCode::Space) || inp.key(KeyCode::KeyE) {
            v += Vec3::Y;
        }
        if inp.key(KeyCode::KeyQ) || (inp.key(KeyCode::ControlLeft) && !inp.key(KeyCode::KeyZ) && !inp.key(KeyCode::KeyY)) {
            v -= Vec3::Y;
        }
        if v != Vec3::ZERO {
            let mut speed = self.camera.speed;
            if inp.shift() {
                speed *= 4.0;
            }
            if inp.alt() {
                speed *= 0.2;
            }
            self.camera.pos += (v.normalize() * speed * dt).as_dvec3();
        }
    }

    fn handle_shortcuts(&mut self) {
        if self.input.pressed(KeyCode::Tab) {
            let on = !self.fly_mode;
            self.set_fly_mode(on);
        }
        if self.input.pressed(KeyCode::Escape) {
            self.set_fly_mode(false);
        }
        let inp = &self.input;
        if inp.pressed(KeyCode::F1) {
            self.ui.show_ui = !self.ui.show_ui;
        }
        if inp.pressed(KeyCode::F2) {
            let name = format!("screenshots/shot_{}.png", chrono_stamp());
            self.screenshot_request = Some(PathBuf::from(name));
        }
        if inp.pressed(KeyCode::F5) {
            self.ui.force_reload = true;
        }
        if inp.ctrl() && inp.pressed(KeyCode::KeyZ) {
            if let Some((label, touched)) = self.world.overlay.undo() {
                self.world.invalidate_blocks(&touched);
                self.ui.toast(format!("Undo: {label}"));
            }
        }
        let inp = &self.input;
        if inp.ctrl() && inp.pressed(KeyCode::KeyY) {
            if let Some((label, touched)) = self.world.overlay.redo() {
                self.world.invalidate_blocks(&touched);
                self.ui.toast(format!("Redo: {label}"));
            }
        }
    }

    fn handle_editing(&mut self) {
        let can_act = self.fly_mode || !self.look_drag;
        if !can_act {
            return;
        }
        self.tools.handle_input(&self.input, self.fly_mode, self.pick, &mut self.world, &mut self.ui, &self.blocks);
    }

    /// New / changed prefabs: re-upload and recompile the generator so Prefab nodes see them.
    fn sync_prefabs(&mut self) -> Result<()> {
        if self.tools.prefabs.revision == self.prefab_revision {
            return Ok(());
        }
        self.prefab_revision = self.tools.prefabs.revision;
        self.tools.prefabs.upload(&self.ctx)?;
        self.wgpu.prefab_addrs = (
            self.tools.prefabs.table.as_ref().map(|b| b.address).unwrap_or(0),
            self.tools.prefabs.data.as_ref().map(|b| b.address).unwrap_or(0),
        );
        self.wg.prefab_info = self.tools.prefabs.compile_info();
        self.wg.prefab_names = self.tools.prefabs.names();
        self.wg.mark_dirty();
        Ok(())
    }

    // ---------------------------------------------------------------------------------------
    // frame
    // ---------------------------------------------------------------------------------------

    pub fn start_benchmark(&mut self) {
        self.set_fly_mode(false);
        self.world.reset(self.frame_counter + 1);
        self.bench = Some(crate::bench::Benchmark::new(self.camera.pos, self.world.stats.jobs_total));
        self.ui.toast("Benchmark started (24 s)".into());
    }

    pub fn save_world(&mut self, path: &std::path::Path) -> Result<()> {
        #[derive(serde::Serialize)]
        struct Save<'a> {
            seed: u32,
            camera: [f64; 3],
            overlay: &'a crate::world::edits::EditOverlay,
        }
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let s = ron::ser::to_string(&Save { seed: self.seed, camera: self.camera.pos.to_array(), overlay: &self.world.overlay })?;
        std::fs::write(path, s)?;
        Ok(())
    }

    pub fn load_world(&mut self, path: &std::path::Path) -> Result<()> {
        #[derive(serde::Deserialize)]
        struct Save {
            seed: u32,
            camera: [f64; 3],
            overlay: crate::world::edits::EditOverlay,
        }
        let s: Save = ron::from_str(&std::fs::read_to_string(path)?)?;
        self.seed = s.seed;
        self.camera.pos = DVec3::from(s.camera);
        self.world.overlay = s.overlay;
        self.world.invalidate_all();
        Ok(())
    }

    /// Blend fog colour/density toward the biome the camera is in.
    fn apply_environment(&mut self) {
        let Some(cam) = self.probes.camera else { return };
        let Some(b) = self.wg.compiled.biomes.get(cam.biome as usize) else { return };
        let k = (self.dt * 1.5).min(1.0);
        let target = Vec3::from(b.fog_color);
        self.atmosphere.fog_tint = self.atmosphere.fog_tint.lerp(target, k);
        self.atmosphere.fog_tint_strength = 0.55;
        let dens = 0.0012 * b.fog_density.max(0.05) * self.ui.fog_scale;
        self.atmosphere.fog_density += (dens - self.atmosphere.fog_density) * k;
    }

    fn read_pick(&mut self, slot: usize) {
        let words: Vec<u32> = self.renderer.per_slot[slot].pick_readback.read(0, 2);
        let payload = words[0];
        let depth = f32::from_bits(words[1]);
        let stats: Vec<u32> = self.renderer.per_slot[slot].stats_readback.read(0, 11);
        self.stats.model_groups = stats[8];
        self.stats.draw_groups = stats[0];
        self.stats.trans_groups = stats[3];
        self.stats.visible_chunks = stats[6];
        self.stats.drawn_quads = stats[7];
        if payload & 0x8000_0000 == 0 || depth <= 0.0 {
            self.pick = None;
            return;
        }
        let fr = &self.frames[slot];
        let dir = payload & 7;
        let block = ((payload >> 3) & 0xFFFF) as u16;
        let lod = (payload >> 19) & 15;
        let (px, py) = fr.pick_px;
        let ndc = Vec4::new(
            (px as f32 + 0.5) / fr.extent.width as f32 * 2.0 - 1.0,
            (py as f32 + 0.5) / fr.extent.height as f32 * 2.0 - 1.0,
            depth,
            1.0,
        );
        let rel = fr.inv_view_proj * ndc;
        let rel = rel.truncate() / rel.w;
        let world = fr.cam_pos + rel.as_dvec3();
        let mut normal = IVec3::ZERO;
        normal[(dir >> 1) as usize] = if dir & 1 == 0 { 1 } else { -1 };
        let is_model = payload & (1 << 30) != 0;
        let inside = if is_model { world - DVec3::new(0.0, 0.01, 0.0) } else { world - normal.as_dvec3() * 0.02 };
        self.pick = Some(PickHit {
            block_pos: inside.floor().as_ivec3(),
            normal,
            block,
            lod,
            world,
            distance: rel.length(),
        });
    }

    pub fn frame(&mut self) -> Result<()> {
        let now = Instant::now();
        self.dt = (now - self.last_frame).as_secs_f32().clamp(1e-5, 0.25);
        self.last_frame = now;
        let cpu_start = now;

        if self.resize_pending {
            self.resize_pending = false;
            self.recreate_swapchain()?;
        }

        let slot = (self.frame_counter % FIF as u64) as usize;
        let value = self.frame_counter + 1;
        if self.frame_counter >= FIF as u64 {
            self.timeline.wait(&self.ctx, value - FIF as u64)?;
            self.read_pick(slot);
            self.finish_screenshot(slot)?;
            self.probes.read(slot);
            self.apply_environment();
        }
        let completed = self.timeline.value(&self.ctx);
        self.profiler.begin_frame(&self.ctx, slot);
        for (name, ms) in &self.profiler.last {
            let e = self.gpu_sums.entry(name).or_insert((0.0, 0));
            e.0 += *ms as f64;
            e.1 += 1;
        }
        // adaptive generation budget: learn the GPU cost per chunk from the last batch
        let gen_ms: f32 = self.profiler.last.iter().filter(|(n, _)| matches!(*n, "worldgen" | "props" | "meshing")).map(|(_, m)| *m).sum();
        if self.batch_sizes[slot] >= 16 && gen_ms > 0.0 {
            let per = gen_ms / self.batch_sizes[slot] as f32;
            self.ms_per_job = self.ms_per_job * 0.8 + per * 0.2;
        }
        self.batch_sizes[slot] = 0;
        self.pm.collect_garbage(&self.ctx, completed);
        self.world.free_deferred(completed);

        // shader hot reload
        let force = std::mem::take(&mut self.ui.force_reload);
        let gen_before = (self.pm.get(self.gen_pipeline), self.pm.get(self.props_pipeline));
        let n = self.pm.hot_reload(&self.ctx, self.frame_counter, force);
        if n > 0 {
            self.ui.toast(format!("Reloaded {n} pipeline(s)"));
            if (self.pm.get(self.gen_pipeline), self.pm.get(self.props_pipeline)) != gen_before {
                self.world.invalidate_all();
            }
        }

        self.sync_prefabs()?;
        // ---- world generator edits ----
        let (params_changed, built) = self.wg.update();
        if params_changed {
            self.wgpu.set_params(&self.wg.compiled.params);
            if self.wg.compiled.source == self.wg.active_source {
                self.world.invalidate_all();
            }
        }
        for (current, b) in built {
            if current {
                let (t, p) = (b.terrain_source.clone(), b.props_source.clone());
                self.pm.replace(
                    self.probe_pipeline,
                    PipelineSource::ComputeSource { name: "worldgen_probe.comp".into(), source: b.probe_source.clone(), defines: vec![] },
                    b.probe,
                    b.deps.clone(),
                    self.frame_counter,
                );
                self.pm.replace(
                    self.gen_pipeline,
                    PipelineSource::ComputeSource { name: "worldgen_terrain.comp".into(), source: t, defines: vec![] },
                    b.terrain,
                    b.deps.clone(),
                    self.frame_counter,
                );
                self.pm.replace(
                    self.props_pipeline,
                    PipelineSource::ComputeSource { name: "worldgen_props.comp".into(), source: p, defines: vec![] },
                    b.props,
                    b.deps,
                    self.frame_counter,
                );
                self.wgpu.set_params(&self.wg.compiled.params);
                self.world.invalidate_all();
                self.ui.toast(format!(
                    "{} generator ready in {:.0} ms",
                    if b.optimized { "Optimised" } else { "Quick" },
                    b.ms
                ));
            } else {
                unsafe {
                    self.ctx.device.destroy_pipeline(b.terrain, None);
                    self.ctx.device.destroy_pipeline(b.props, None);
                    self.ctx.device.destroy_pipeline(b.probe, None);
                }
            }
        }

        // ---- world results ----
        for (s, jobs, results) in self.wgpu.poll(completed) {
            for (i, j) in jobs.iter().enumerate() {
                if j.readback {
                    if let Some(v) = self.wgpu.take_voxels(s, i) {
                        self.world.fill_region(j.key, &v);
                    }
                }
            }
            self.world.apply_results(s, &jobs, &results, value);
        }
        for region in self.world.take_finished_regions() {
            self.tools.region_ready(region, &mut self.world, &self.blocks, &mut self.ui);
        }

        // ---- input, camera, editing ----
        self.handle_shortcuts();
        if self.opts.benchmark && self.bench.is_none() && self.frame_counter == 2 {
            self.opts.benchmark = false;
            self.start_benchmark();
        }
        let mut bench_done = false;
        if let Some(b) = &mut self.bench {
            if b.report.is_none() && !b.update(&mut self.camera, self.dt, &self.profiler.last) {
                bench_done = true;
            }
        } else {
            self.update_camera(self.dt);
        }
        if bench_done {
            let rep = self.bench.as_mut().unwrap().finish(self.world.stats.jobs_total, self.world.stats.quads_resident, &self.ctx.device_name);
            log::info!("\n{rep}");
            let _ = std::fs::create_dir_all("benchmarks");
            let _ = std::fs::write(format!("benchmarks/bench_{}.txt", chrono_stamp()), &rep);
            self.ui.toast("Benchmark finished (see benchmarks/)".into());
            if self.opts.frames.is_none() && self.opts.headless {
                self.quit = true;
            }
        }
        self.handle_editing();
        self.atmosphere.update(self.dt);
        self.world.update_desired(self.camera.pos);
        self.world.process_retirement(value);
        self.world.update_stats();

        // ---- UI ----
        let actions = self.build_ui()?;
        self.apply_ui_actions(actions, value)?;

        // ---- acquire ----
        let mut image_index = 0u32;
        let fr_acquire = self.frames[slot].acquire;
        if let Some(sc) = &self.swapchain {
            match unsafe { self.ctx.swapchain_loader.acquire_next_image(sc.raw, u64::MAX, fr_acquire, vk::Fence::null()) } {
                Ok((i, suboptimal)) => {
                    image_index = i;
                    if suboptimal {
                        self.resize_pending = true;
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.resize_pending = true;
                    self.input.end_frame();
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
            }
        }

        // ---- record ----
        let cb = self.frames[slot].cb;
        unsafe {
            self.ctx.device.reset_command_pool(self.frames[slot].pool, vk::CommandPoolResetFlags::empty())?;
            self.ctx.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        let profiler_slot = slot;
        self.profiler.begin(&self.ctx, cb, profiler_slot, "frame");

        // uploads (pool copies + chunk records)
        let copies = std::mem::take(&mut self.world.copies);
        let updates = std::mem::take(&mut self.world.record_updates);
        self.wgpu.record_uploads(&self.ctx, cb, slot, value, &copies, &updates);

        // generation batch
        if self.wgpu.has_free_slot() && !self.world.frozen {
            let budget_jobs = (self.ui.gen_budget_ms / self.ms_per_job.max(0.002)) as usize;
            let max_jobs = budget_jobs.clamp(16, self.ui.jobs_per_frame.min(MAX_JOBS));
            let jobs = self.world.pick_jobs(self.camera.pos, self.camera.forward(), max_jobs);
            self.batch_sizes[slot] = jobs.len();
            if !jobs.is_empty() {
                let pipes = WorldPipelines {
                    generate: self.pm.get(self.gen_pipeline),
                    props: self.pm.get(self.props_pipeline),
                    apply_edits: self.pm.get(self.renderer.pipes.apply_edits),
                    mesh_count: self.pm.get(self.renderer.pipes.mesh_count),
                    mesh_write: self.pm.get(self.renderer.pipes.mesh_write),
                    prefix: self.pm.get(self.renderer.pipes.prefix),
                    layout: self.pm.layout,
                };
                unsafe {
                    self.ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.pm.layout, 0, &[self.bindless.set], &[]);
                }
                if let Err(rest) = self.wgpu.record_batch(
                    &self.ctx,
                    cb,
                    value,
                    jobs,
                    &pipes,
                    self.renderer.block_table.address,
                    self.seed,
                    &mut self.profiler,
                    profiler_slot,
                ) {
                    for j in rest {
                        if j.readback {
                            for r in &mut self.world.region_reads {
                                r.requeue(j.key.pos);
                            }
                        } else if let Some(c) = self.world.chunks.get_mut(&j.key) {
                            c.in_flight = false;
                        }
                    }
                }
            }
        }

        // biome probes (camera environment + minimap)
        {
            let pos = self.camera.pos;
            let n = self.probes.prepare(slot, pos, self.frame_counter);
            unsafe {
                self.ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pm.get(self.probe_pipeline));
                self.ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.pm.layout, 0, &[self.bindless.set], &[]);
            }
            let push = crate::world::gpu::GenPush {
                jobs: self.probes.input[slot].address,
                voxels: self.wgpu.voxels.address,
                results: self.probes.output[slot].address,
                edits: self.probes.input[slot].address,
                params: self.wgpu.params.address,
                columns: self.wgpu.columns.address,
                prefab_table: self.wgpu.prefab_addrs.0,
                prefab_data: self.wgpu.prefab_addrs.1,
                num_jobs: n,
                seed: self.seed,
                edit_count: 0,
                pad: 0,
            };
            unsafe {
                self.ctx.device.cmd_push_constants(cb, self.pm.layout, vk::ShaderStageFlags::ALL, 0, bytemuck::bytes_of(&push));
                self.ctx.device.cmd_dispatch(cb, n.div_ceil(64), 1, 1);
            }
            memory_barrier(
                &self.ctx,
                cb,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::HOST,
                vk::AccessFlags2::HOST_READ,
            );
        }

        // frame constants
        let extent = self.renderer.extent;
        let aspect = extent.width as f32 / extent.height.max(1) as f32;
        let m = self.camera.matrices(aspect);
        let pick_px = if self.fly_mode || self.window.is_none() {
            (extent.width / 2, extent.height / 2)
        } else {
            let scale = self.window.as_ref().map(|w| w.scale_factor()).unwrap_or(1.0);
            let _ = scale;
            (self.input.cursor.0.max(0.0) as u32, self.input.cursor.1.max(0.0) as u32)
        };
        self.write_frame_data(slot, &m, pick_px, extent);
        {
            let fr = &mut self.frames[slot];
            fr.cam_pos = self.camera.pos;
            fr.inv_view_proj = m.inv_view_proj;
            fr.pick_px = pick_px;
            fr.extent = extent;
        }
        self.prev_view_proj = m.view_proj;

        self.renderer.record_scene(
            &self.ctx,
            cb,
            slot,
            &self.pm,
            &self.bindless,
            &mut self.profiler,
            &self.wgpu.records,
            &self.wgpu.quads,
            self.world.record_count,
            self.ui.wireframe,
            self.wg.biome_table.address,
            self.ui.shadows,
            self.ui.clouds,
        );

        // ---- output (tonemap + UI) ----
        let (target_image, target_view, target_extent) = if let Some(sc) = &self.swapchain {
            (sc.images[image_index as usize], sc.views[image_index as usize], sc.extent)
        } else {
            let o = self.offscreen.as_ref().unwrap();
            (o.raw, o.view, vk::Extent2D { width: o.extent.width, height: o.extent.height })
        };
        image_barrier(
            &self.ctx,
            cb,
            target_image,
            vkcore::sync::color_range(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        self.profiler.begin(&self.ctx, cb, profiler_slot, "tonemap+ui");
        let draw_ui = self.ui.show_ui;
        let retire_batch;
        {
            let ictx = &mut self.imgui.ctx;
            let ir = self.imgui.renderer.as_mut().unwrap();
            let consumer = ir.renderer_consumer().map_err(|e| anyhow::anyhow!("{e}"))?;
            let pending = ictx.render(consumer);
            let (imgui_frame, retire) = ir.prepare_frame(pending).map_err(|e| anyhow::anyhow!("{e}"))?;
            retire_batch = retire;
            let mut imgui_frame = Some(imgui_frame);
            self.renderer.record_output(&self.ctx, cb, slot, &self.pm, target_view, target_extent, |cb| {
                if draw_ui {
                    if let Some(f) = imgui_frame.take() {
                        if let Err(e) = unsafe { ir.cmd_draw(cb, f) } {
                            log::error!("imgui draw: {e}");
                        }
                    }
                }
            });
        }
        self.profiler.end(&self.ctx, cb, profiler_slot);

        // screenshot copy
        let want_shot = self.screenshot_request.take().or_else(|| {
            if Some(self.frame_counter + 1) == self.opts.frames {
                self.opts.screenshot.clone()
            } else {
                None
            }
        });
        let final_layout = if self.swapchain.is_some() { vk::ImageLayout::PRESENT_SRC_KHR } else { vk::ImageLayout::TRANSFER_SRC_OPTIMAL };
        if let Some(path) = want_shot {
            image_barrier(
                &self.ctx,
                cb,
                target_image,
                vkcore::sync::color_range(),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            );
            let buf = Buffer::readback(&self.ctx, "screenshot", target_extent.width as u64 * target_extent.height as u64 * 4)?;
            unsafe {
                self.ctx.device.cmd_copy_image_to_buffer(
                    cb,
                    target_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    buf.raw,
                    &[vk::BufferImageCopy::default()
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_extent(vk::Extent3D { width: target_extent.width, height: target_extent.height, depth: 1 })],
                );
            }
            memory_barrier(
                &self.ctx,
                cb,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::HOST,
                vk::AccessFlags2::HOST_READ,
            );
            image_barrier(
                &self.ctx,
                cb,
                target_image,
                vkcore::sync::color_range(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                final_layout,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                vk::AccessFlags2::NONE,
            );
            self.frames[slot].screenshot = Some((path, buf, target_extent));
        } else {
            image_barrier(
                &self.ctx,
                cb,
                target_image,
                vkcore::sync::color_range(),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                final_layout,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                vk::AccessFlags2::NONE,
            );
        }
        self.profiler.end(&self.ctx, cb, profiler_slot);

        unsafe { self.ctx.device.end_command_buffer(cb)? };

        // ---- submit ----
        let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
        let mut signals = vec![vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline.semaphore)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let mut waits = Vec::new();
        if let Some(sc) = &self.swapchain {
            waits.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(fr_acquire)
                    .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
            );
            signals.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(sc.present_semaphores[image_index as usize])
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&cbs)
            .wait_semaphore_infos(&waits)
            .signal_semaphore_infos(&signals);
        unsafe { self.ctx.device.queue_submit2(self.ctx.queue, &[submit], vk::Fence::null())? };

        if let Some(sc) = &self.swapchain {
            let waits = [sc.present_semaphores[image_index as usize]];
            let swapchains = [sc.raw];
            let indices = [image_index];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&waits)
                .swapchains(&swapchains)
                .image_indices(&indices);
            match unsafe { self.ctx.swapchain_loader.queue_present(self.ctx.queue, &present) } {
                Ok(suboptimal) => {
                    if suboptimal {
                        self.resize_pending = true;
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.resize_pending = true,
                Err(e) => return Err(e.into()),
            }
        }
        self.frame_counter += 1;
        self.input.end_frame();
        if let Some(batch) = retire_batch {
            if let Some(ir) = self.imgui.renderer.as_mut() {
                ir.wait_for_texture_retirements(batch).map_err(|e| anyhow::anyhow!("{e}"))?;
            }
        }

        // stats
        let ms = self.dt * 1000.0;
        self.stats.cpu_ms = cpu_start.elapsed().as_secs_f32() * 1000.0;
        self.stats.frame_ms_history.push(ms);
        if self.stats.frame_ms_history.len() > 240 {
            self.stats.frame_ms_history.remove(0);
        }
        let n = self.stats.frame_ms_history.len().min(60) as f32;
        let avg: f32 = self.stats.frame_ms_history.iter().rev().take(60).sum::<f32>() / n;
        self.stats.fps = 1000.0 / avg.max(0.001);

        {
            if let Some(f) = self.opts.frames {
                if self.frame_counter >= f {
                    // flush outstanding screenshot
                    self.timeline.wait(&self.ctx, self.frame_counter)?;
                    for s in 0..FIF {
                        self.finish_screenshot(s)?;
                    }
                    self.quit = true;
                }
            }
        }
        Ok(())
    }

    fn finish_screenshot(&mut self, slot: usize) -> Result<()> {
        if let Some((path, mut buf, extent)) = self.frames[slot].screenshot.take() {
            let data = buf.mapped().to_vec();
            buf.destroy(&self.ctx);
            let bgra = matches!(
                self.swapchain.as_ref().map(|s| s.format).unwrap_or(OUTPUT_FORMAT_HEADLESS),
                vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB
            );
            std::thread::spawn(move || {
                let mut rgba = data;
                if bgra {
                    for px in rgba.chunks_mut(4) {
                        px.swap(0, 2);
                        px[3] = 255;
                    }
                }
                if let Some(dir) = path.parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
                match image::save_buffer(&path, &rgba, extent.width, extent.height, image::ColorType::Rgba8) {
                    Ok(()) => log::info!("saved screenshot {}", path.display()),
                    Err(e) => log::error!("screenshot failed: {e}"),
                }
            })
            .join()
            .ok();
            self.ui.toast("Screenshot saved".to_string());
        }
        Ok(())
    }

    fn write_frame_data(&mut self, slot: usize, m: &crate::camera::CameraMatrices, pick_px: (u32, u32), extent: vk::Extent2D) {
        let sky = self.atmosphere.colors();
        let mut flags = 0;
        if self.ui.fog {
            flags |= FLAG_FOG;
        }
        if self.fly_mode {
            flags |= FLAG_CROSSHAIR;
        }
        let view_distance = (crate::world::CHUNK << (self.world.settings.levels - 1)) as f32 * self.world.settings.radius as f32;
        let fd = FrameData {
            view_proj: m.view_proj.to_cols_array_2d(),
            inv_view_proj: m.inv_view_proj.to_cols_array_2d(),
            view: m.view.to_cols_array_2d(),
            proj: m.proj.to_cols_array_2d(),
            prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
            frustum: m.frustum.map(|p| p.to_array()),
            cam_block: [m.cam_block.x, m.cam_block.y, m.cam_block.z, 0],
            cam_frac: [m.cam_frac.x, m.cam_frac.y, m.cam_frac.z, 0.0],
            sun_dir: [sky.sun_dir.x, sky.sun_dir.y, sky.sun_dir.z, sky.sun_intensity],
            sun_color: sky.sun_color.extend(1.0).to_array(),
            sky_zenith: sky.zenith.extend(1.0).to_array(),
            sky_horizon: sky.horizon.extend(1.0).to_array(),
            ground_color: sky.ground.extend(1.0).to_array(),
            fog_color: sky.fog.extend(self.atmosphere.fog_density).to_array(),
            fog_params: [self.atmosphere.fog_height_falloff, self.atmosphere.fog_start, self.atmosphere.sea_level, view_distance],
            resolution: [extent.width as f32, extent.height as f32],
            time: self.start.elapsed().as_secs_f32(),
            dt: self.dt,
            frame_index: self.frame_counter as u32,
            debug_mode: self.ui.debug_view,
            flags,
            pick_pixel: pick_px.0.min(0xFFFF) | (pick_px.1.min(0xFFFF) << 16),
            block_tex_albedo: self.renderer.albedo_tex,
            block_tex_normal: self.renderer.normal_tex,
            hdr_tex: self.renderer.hdr_tex,
            depth_tex: self.renderer.depth_tex,
            exposure: [self.atmosphere.exposure, self.atmosphere.time_of_day, 0.0, 0.0],
            shadow_vp: [[[0.0; 4]; 4]; 4],
            shadow_planes: [[0.0; 4]; 24],
            shadow_splits: [0.0; 4],
            cam_forward: [0.0; 4],
            shadow_tex: self.renderer.shadow_tex,
            shadow_res: crate::renderer::SHADOW_RES,
            pad: [0; 2],
        };
        let mut fd = fd;
        let light = if sky.sun_dir.y > -0.05 { sky.sun_dir } else { -sky.sun_dir };
        let casc = crate::camera::fit_cascades(&self.camera, extent.width as f32 / extent.height.max(1) as f32, light, self.ui.shadow_distance, crate::renderer::SHADOW_RES);
        for c in 0..4 {
            fd.shadow_vp[c] = casc.view_proj[c].to_cols_array_2d();
            for p in 0..6 {
                fd.shadow_planes[c * 6 + p] = casc.planes[c][p].to_array();
            }
        }
        fd.shadow_splits = casc.splits;
        let fw = self.camera.forward();
        let night = (1.0 - ((sky.sun_dir.y.abs() - 0.02) / 0.1).clamp(0.0, 1.0)) * 0.6;
        fd.cam_forward = [fw.x, fw.y, fw.z, self.ui.shadow_strength * (1.0 - night)];
        if self.ui.shadows {
            fd.flags |= crate::renderer::FLAG_SHADOWS;
        }
        self.renderer.per_slot[slot].frame_buf.write(0, std::slice::from_ref(&fd));
    }

    fn build_ui(&mut self) -> Result<UiActions> {
        if let (Some(p), Some(w)) = (&mut self.imgui.platform, &self.window) {
            p.prepare_frame(&mut self.imgui.ctx, w)?;
        } else {
            let io = self.imgui.ctx.io_mut();
            io.set_delta_time(self.dt);
            io.set_display_size([self.renderer.extent.width as f32, self.renderer.extent.height as f32]);
        }
        self.node_ui.prepare(&self.imgui.ctx, &self.wg);
        let ui = self.imgui.ctx.frame();
        if self.ui.show_ui {
            self.node_ui.draw(ui, &mut self.wg, &self.blocks);
            self.tools.draw_ui(ui, &mut self.world, &self.blocks, &mut self.ui, self.pick);
        }
        self.tools.draw_overlay(ui, &self.camera, self.pick);
        let actions = ui::draw(
            ui,
            &mut ui::UiCtx {
                state: &mut self.ui,
                camera: &mut self.camera,
                atmosphere: &mut self.atmosphere,
                world: &mut self.world,
                blocks: &self.blocks,
                stats: &self.stats,
                gpu_times: &self.profiler.last,
                pick: self.pick,
                fly_mode: self.fly_mode,
                device_name: &self.ctx.device_name,
                seed: &mut self.seed,
                shader_errors: &self.pm.last_errors,
                pool_capacity: self.wgpu.quad_capacity,
                batch: self.wgpu.last_batch,
                busy_slots: self.wgpu.busy_slots(),
                probes: &self.probes,
                biome_names: &self.wg.compiled.biomes,
                zone_names: &self.wg.compiled.zones,
                vsync: self.swapchain.as_ref().map(|s| s.vsync),
                node_editor_open: &mut self.node_ui.open,
                tools_open: &mut self.tools.show_tools,
                bench_report: self.bench.as_ref().and_then(|b| b.report.clone()),
            },
        );
        if let (Some(p), Some(w)) = (&mut self.imgui.platform, &self.window) {
            p.prepare_render(ui, w)?;
        }
        Ok(actions)
    }

    fn apply_ui_actions(&mut self, a: UiActions, value: u64) -> Result<()> {
        if a.regenerate {
            self.world.invalidate_all();
        }
        if a.reset_world {
            self.world.reset(value);
        }
        if a.clear_edits {
            let touched = self.world.overlay.clear();
            self.world.invalidate_blocks(&touched);
        }
        if a.screenshot {
            self.screenshot_request = Some(PathBuf::from(format!("screenshots/shot_{}.png", chrono_stamp())));
        }
        if let Some(v) = a.set_vsync {
            if let Some(sc) = &mut self.swapchain {
                self.ctx.wait_idle();
                let e = sc.extent;
                sc.recreate(&self.ctx, e.width, e.height, v)?;
            }
        }
        if a.undo {
            if let Some((_, touched)) = self.world.overlay.undo() {
                self.world.invalidate_blocks(&touched);
            }
        }
        if a.redo {
            if let Some((_, touched)) = self.world.overlay.redo() {
                self.world.invalidate_blocks(&touched);
            }
        }
        if a.quit {
            self.quit = true;
        }
        if let Some((x, z)) = a.teleport {
            self.camera.pos.x = x;
            self.camera.pos.z = z;
            if let Some(h) = self.probes.height_at(x, z) {
                self.camera.pos.y = h as f64 + 40.0;
            }
        }
        if a.save_generator {
            self.wg.save();
        }
        if a.benchmark {
            self.start_benchmark();
        }
        if a.save_world {
            match self.save_world(std::path::Path::new("saves/world.ron")) {
                Ok(()) => self.ui.toast("World edits saved to saves/world.ron".into()),
                Err(e) => self.ui.toast(format!("Save failed: {e}")),
            }
        }
        if a.load_world {
            match self.load_world(std::path::Path::new("saves/world.ron")) {
                Ok(()) => self.ui.toast("World edits loaded".into()),
                Err(e) => self.ui.toast(format!("Load failed: {e}")),
            }
        }
        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        let Some(w) = &self.window else { return Ok(()) };
        let size = w.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        self.ctx.wait_idle();
        let sc = self.swapchain.as_mut().unwrap();
        let vsync = sc.vsync;
        sc.recreate(&self.ctx, size.width, size.height, vsync)?;
        let extent = sc.extent;
        self.renderer.resize(&self.ctx, &mut self.bindless, extent)?;
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.ctx.wait_idle();
        if let Some(mut r) = self.imgui.renderer.take() {
            let _ = r.shutdown(&mut self.imgui.ctx);
        }
        if let Some(mut p) = self.imgui.platform.take() {
            let _ = p.shutdown(&mut self.imgui.ctx);
        }
        unsafe {
            self.ctx.device.destroy_command_pool(self.imgui.pool, None);
            for f in &mut self.frames {
                if let Some((_, mut b, _)) = f.screenshot.take() {
                    b.destroy(&self.ctx);
                }
                self.ctx.device.destroy_command_pool(f.pool, None);
                self.ctx.device.destroy_semaphore(f.acquire, None);
            }
        }
        self.timeline.destroy(&self.ctx);
        self.profiler.destroy(&self.ctx);
        self.wgpu.destroy(&self.ctx);
        self.tools.prefabs.destroy(&self.ctx);
        self.probes.destroy(&self.ctx);
        self.wg.destroy(&self.ctx);
        self.renderer.destroy(&self.ctx);
        if let Some(mut o) = self.offscreen.take() {
            o.destroy(&self.ctx);
        }
        if let Some(mut sc) = self.swapchain.take() {
            sc.destroy(&self.ctx);
        }
        self.pm.destroy(&self.ctx);
        self.bindless.destroy(&self.ctx);
    }
}

fn find_shader_root() -> Result<PathBuf> {
    let candidates = [
        PathBuf::from("shaders"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../shaders"),
    ];
    for c in candidates {
        if c.join("common.glsl").exists() {
            return c.canonicalize().context("canonicalize shader root");
        }
    }
    anyhow::bail!("could not find the shaders/ directory")
}

pub fn chrono_stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    format!("{secs}")
}

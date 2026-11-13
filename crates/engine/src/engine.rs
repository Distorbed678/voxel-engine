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
use crate::world::{RegionRead, World};
use crate::world::gpu::{MAX_JOBS, WorldGpu, WorldPipelines};
use crate::node_editor::NodeEditorUi;
use crate::wg::WorldgenSystem;

mod script;

pub const FIF: usize = 2;

/// A cached far shadow cascade. Its window is centred on the camera (wide enough for the frustum
/// slice in any direction, plus a margin), so turning around doesn't invalidate it. It is redrawn
/// when the camera leaves the margin, the sun turns by more than 0.1 degrees, or shadow-casting
/// chunks inside it change (at most every 8 frames while streaming).
#[derive(Default)]
struct ShadowCache {
    valid: bool,
    /// camera position when drawn: the window centre, and what the matrix is relative to
    at: DVec3,
    view_proj: Mat4,
    light: Vec3,
    radius: f32,
    axes: (Vec3, Vec3),
    drawn: u64,
    dirty: bool,
}

/// The cascades that are cached, and how far the camera may move before each is redrawn.
const CACHED_CASCADES: [usize; 2] = [2, 3];
const SHADOW_CACHE_MARGIN: [f32; 2] = [16.0, 32.0];
const SHADOW_CACHE_MIN_FRAMES: u64 = 8;
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
    /// Scripted input playback (headless): see `engine/script.rs`.
    pub script: Option<PathBuf>,
    /// World file to open (or to create on the first save if it doesn't exist).
    pub world: Option<PathBuf>,
    /// Block definitions folder (default `assets/blocks`).
    pub blocks_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
pub struct PickHit {
    pub block_pos: IVec3,
    pub normal: IVec3,
    pub block: u16,
    pub lod: u32,
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
    /// Chunks hidden by occlusion culling this frame.
    pub occluded_chunks: u32,
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
    /// Block table, texture arrays and the UI atlas.
    pub blockgpu: crate::blockgpu::BlockGpu,
    /// The UI atlas as an ImGui texture.
    atlas_tex: Option<dear_imgui_rs::TextureId>,
    pub block_ui: crate::block_editor::BlockEditorUi,
    /// The prop editor's 3D preview (shown with a Prop graph open in the node editor).
    pub prop_preview: crate::prop_preview::PropPreview,
    /// the prop graph whose preview the user closed (it stays closed until another one opens)
    prop_preview_closed: Option<String>,
    /// a layer jump waiting for the ground under the camera
    jump: Option<JumpState>,
    pub tex_ui: crate::texture_editor::TextureEditorUi,
    /// per generator layer: its terrain, props and probe pipelines (`wg::PASS_NAMES`)
    gen_layers: Vec<[PipelineHandle; 3]>,
    pub probes: crate::probe::Probes,
    pub light: crate::light::LightGpu,
    taa: crate::taa::Taa,
    particles: crate::particles::Particles,
    bloom: crate::bloom::Bloom,
    /// Camera position of the previous frame (TAA reprojection).
    prev_cam_pos: DVec3,
    /// Whether the previous frame used TAA (turning it on resets the history).
    prev_taa: bool,
    /// Jitter sequence position (scripts restart it after `settle` so screenshots don't depend
    /// on how many frames loading took).
    taa_frame: u64,
    /// Jump the environment (fog, grading, weather, particle shares) to its target on the next
    /// frame instead of blending (scripts set it after `settle`).
    env_snap: bool,
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
    /// The open world file: Ctrl+S saves here.
    pub world_path: Option<PathBuf>,
    /// Edit overlay revision when the world was last saved or opened.
    saved_revision: u64,
    shadow_cache: [ShadowCache; 2],
    /// shadow cascades to redraw this frame (bit c)
    shadow_mask: u32,
    /// solo preview (zone, biome) as last sent to the generator
    solo_applied: (Option<u32>, Option<u32>),
    pub fluid: crate::fluid::FluidSim,
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
    /// Scripted input playback, if running a script.
    pub script: Option<script::Script>,
    /// Animation clock (wind, water, clouds); advances by `dt`, so scripts are deterministic.
    anim_time: f32,
    /// Scripts can freeze the animation clock so screenshots don't depend on frame counts.
    anim_frozen: bool,
    ctx: VkContext,
}

/// Full GLSL sources (terrain pass, props pass, probe pass) around the generated code.
/// A layer jump in progress (`Engine::start_layer_jump`).
#[derive(Clone, Copy, Debug)]
enum JumpState {
    /// stand on what the probes find, after `wait` frames, above `sea`; with nothing under the
    /// camera, on the nearest ground of a minimap made after the jump (`map_serial` at the jump)
    Probe { wait: u32, sea: f32, map_serial: u64 },
    /// a column read (region read `id`) around `depth`: stand in the nearest open pocket
    Column { id: u64, depth: f32, tries: u32 },
}

pub fn compose_source(generated: &str) -> (String, String, String) {
    let wrap = |main: &str, defines: &str| {
        format!("#version 460\n{defines}#include \"worldgen/prelude.glsl\"\n// ---- generated ----\n{generated}\n// ---- main ----\n#include \"{main}\"\n")
    };
    // only the props pass builds prop graphs (their shared arrays and build functions)
    (wrap("worldgen/terrain_main.glsl", ""), wrap("worldgen/props_main.glsl", "#define PROPS_PASS\n"), wrap("worldgen/probe_main.glsl", ""))
}

impl Engine {
    pub fn new(opts: Options, window: Option<Arc<Window>>) -> Result<Self> {
        let display = window.as_ref().map(|w| {
            use raw_window_handle::HasDisplayHandle;
            w.display_handle().unwrap().as_raw()
        });
        let ctx = VkContext::new(ContextDesc { display, validation: opts.validation, app_name: "voxel-engine" })?;

        let shader_root = find_shader_root()?;
        let blocks_dir = opts.blocks_dir.clone().unwrap_or_else(|| PathBuf::from("assets/blocks"));
        let mut blocks = BlockRegistry::load(&blocks_dir, std::path::Path::new("assets/textures"))?;
        // scripted runs never write block or texture files
        blocks.autosave = opts.script.is_none();

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

        let renderer = Renderer::new(&ctx, &mut bindless, &mut pm, extent, out_format, FIF)?;
        let blockgpu = crate::blockgpu::BlockGpu::new(&ctx, &mut bindless, &mut blocks, FIF)?;
        let light = crate::light::LightGpu::new(&ctx, &mut pm, &mut bindless)?;
        let taa = crate::taa::Taa::new(&ctx, &mut pm, &mut bindless, extent)?;
        let particles = crate::particles::Particles::new(&ctx, &mut pm)?;
        let bloom = crate::bloom::Bloom::new(&ctx, &mut pm, &mut bindless, extent)?;
        let block_ids = blocks.id_map();
        let mut prefabs = crate::prefab::PrefabLibrary::load(PathBuf::from("assets/prefabs"), &blocks);
        prefabs.upload(&ctx)?;
        let mut wg = WorldgenSystem::new(&ctx, &pm.compiler.root.clone(), pm.layout, block_ids, blocks.glsl_header(), prefabs.compile_info(), opts.worldgen_dir.clone().unwrap_or_else(|| PathBuf::from("assets/worldgen")))?;
        // Build the generator synchronously at startup (fall back to the built-in one on error):
        // every layer's passes at once.
        let t0 = Instant::now();
        let build = |pm: &mut PipelineManager, layers: Vec<[String; 3]>| -> Result<Vec<[PipelineHandle; 3]>> {
            let items = layers
                .into_iter()
                .flat_map(|srcs| {
                    crate::wg::PASS_NAMES.into_iter().zip(srcs).map(|(n, s)| (n, PipelineSource::ComputeSource { name: format!("{n}.comp"), source: s, defines: vec![] }))
                })
                .collect();
            let h = pm.create_sources_parallel(&ctx, items)?;
            Ok(h.as_chunks::<3>().0.to_vec())
        };
        let gen_layers = match build(&mut pm, wg.sources()) {
            Ok(h) => h,
            Err(e) => {
                log::error!("world generator failed to build, using defaults: {e:#}");
                wg.reset_to_defaults();
                wg.update();
                build(&mut pm, wg.sources())?
            }
        };
        wg.active = wg.layer_sources();
        wg.dirty = false;
        wg.request_optimized();
        log::info!(
            "world generator: {} layers, {} biomes, {} zones, {} prop layers, {} params, shaders built in {:.0} ms",
            wg.compiled.layers.len(),
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
        let script = match &opts.script {
            Some(p) => Some(script::Script::load(p)?),
            None => None,
        };
        let mut ictx = dear_imgui_rs::Context::create();
        // scripted runs use the default window layout, never the user's imgui.ini
        let ini = if script.is_some() { None } else { Some(PathBuf::from("imgui.ini")) };
        let _ = ictx.set_ini_filename(ini);
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
        let mut imgui_renderer =
            unsafe { dear_imgui_ash::AshRenderer::with_default_allocator(&ctx.instance, ctx.physical_device, config, &mut ictx)? };
        // the block texture atlas for thumbnails and previews in the editors
        let atlas_tex = unsafe { imgui_renderer.register_external_texture(blockgpu.atlas.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) }
            .map_err(|e| log::warn!("could not register the block atlas with ImGui: {e}"))
            .ok();

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
        let world_arg = opts.world.clone();
        // what the command line sets explicitly wins over the world file
        let keep = (opts.seed.is_some(), opts.cam_pos.is_some(), opts.cam_angles.is_some(), opts.time_of_day.is_some());
        let mut engine = Self {
            world_path: None,
            saved_revision: 0,
            shadow_cache: Default::default(),
            shadow_mask: 0,
            solo_applied: (None, None),
            fluid: crate::fluid::FluidSim::new(&blocks),
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
            blockgpu,
            atlas_tex,
            block_ui: crate::block_editor::BlockEditorUi::new(),
            prop_preview: crate::prop_preview::PropPreview::new(),
            prop_preview_closed: None,
            jump: None,
            tex_ui: crate::texture_editor::TextureEditorUi::new(),
            gen_layers,
            probes: crate::probe::Probes::new(&ctx, FIF)?,
            light,
            taa,
            particles,
            bloom,
            prev_cam_pos: DVec3::ZERO,
            prev_taa: false,
            taa_frame: 0,
            env_snap: false,
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
            script,
            anim_time: 0.0,
            anim_frozen: false,
            ctx,
        };
        if let Some(path) = world_arg {
            if path.exists() {
                let (seed, pos, angles, time) = (engine.seed, engine.camera.pos, (engine.camera.yaw, engine.camera.pitch), engine.atmosphere.time_of_day);
                let unknown = engine.load_world(&path)?;
                if !unknown.is_empty() {
                    log::warn!("{}: unknown blocks left to the generator: {}", path.display(), unknown.join(", "));
                }
                if keep.0 {
                    engine.seed = seed;
                }
                if keep.1 {
                    engine.camera.pos = pos;
                }
                if keep.2 {
                    (engine.camera.yaw, engine.camera.pitch) = angles;
                }
                if keep.3 {
                    engine.atmosphere.time_of_day = time;
                }
            } else {
                log::info!("{} doesn't exist yet: it will be created on the first save", path.display());
                engine.set_world_path(Some(path));
            }
        }
        Ok(engine)
    }

    // ---------------------------------------------------------------------------------------
    // input
    // ---------------------------------------------------------------------------------------

    pub fn handle_window_event(&mut self, event: &winit::event::WindowEvent) {
        use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
        if crate::input::forward_to_imgui(event, self.fly_mode || self.look_drag, self.fly_mode) {
            if let (Some(p), Some(w)) = (&mut self.imgui.platform, &self.window) {
                let _ = p.handle_window_event(&mut self.imgui.ctx, w.id(), event);
            } else {
                // headless: scripted input reaches ImGui directly
                crate::input::feed_imgui(&mut self.imgui.ctx, event);
            }
        }
        let io = self.imgui.ctx.io();
        let imgui_wants_mouse = io.want_capture_mouse();
        let imgui_wants_kb = io.want_capture_keyboard();
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
                // a release that happens while unfocused never arrives: don't stay in look-drag
                self.input.mouse_held = [false; 3];
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
        // (editor mode: Ctrl+S saves rather than moving)
        let saving = !self.fly_mode && inp.ctrl() && inp.key(KeyCode::KeyS);
        if inp.key(KeyCode::KeyS) && !saving {
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
        if inp.key(KeyCode::KeyQ) || (inp.key(KeyCode::ControlLeft) && !inp.key(KeyCode::KeyZ) && !inp.key(KeyCode::KeyY) && !saving) {
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
        // Ctrl+S only with the cursor free: in fly mode Ctrl and S both move the camera
        if !self.fly_mode && inp.ctrl() && inp.pressed(KeyCode::KeyS) && !self.imgui.ctx.io().want_text_input() {
            match self.world_path.clone() {
                Some(p) => self.save_world_ui(&p),
                None => self.ui.show_worlds = true,
            }
        }
        // Ctrl+Z/Y over the texture canvas undo painting (the editor handles them)
        if self.tex_ui.hovered && self.tex_ui.open {
            return;
        }
        let inp = &self.input;
        if inp.ctrl() && inp.pressed(KeyCode::KeyZ)
            && let Some((label, touched)) = self.world.overlay.undo() {
                self.world.invalidate_blocks(&touched);
                self.ui.toast(format!("Undo: {label}"));
            }
        let inp = &self.input;
        if inp.ctrl() && inp.pressed(KeyCode::KeyY)
            && let Some((label, touched)) = self.world.overlay.redo() {
                self.world.invalidate_blocks(&touched);
                self.ui.toast(format!("Redo: {label}"));
            }
    }

    fn handle_editing(&mut self) {
        let can_act = self.fly_mode || !self.look_drag;
        if !can_act {
            return;
        }
        self.tools.handle_input(&self.input, self.dt, self.fly_mode, self.pick, self.camera.forward(), &mut self.world, &mut self.ui, &self.blocks);
    }

    /// Apply block and texture edits (block editor, texture editor, scripts): GPU table, texture
    /// layers, the generator's block header, and regeneration where the world changes.
    fn sync_blocks(&mut self, slot: usize) -> Result<()> {
        if let Some(Err(e)) = self.blocks.autosave_tick() {
            self.ui.toast(format!("Saving blocks failed: {e:#}"));
        }
        let ch = self.blocks.take_changes();
        if !ch.any() {
            return Ok(());
        }
        if ch.layout {
            self.blockgpu.relayout(&self.ctx, &mut self.bindless, &mut self.blocks)?;
            if std::mem::take(&mut self.blockgpu.atlas_replaced)
                && let (Some(id), Some(r)) = (self.atlas_tex, self.imgui.renderer.as_mut())
            {
                // SAFETY: the new atlas view lives until the next replacement or shutdown;
                // this waits for the GPU before swapping the descriptor
                let _ = unsafe { r.update_external_texture(id, self.blockgpu.atlas.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) };
            }
        } else if !ch.layers.is_empty() {
            let layers = ch.layers.iter().filter_map(|&l| self.blocks.layer_image(l).map(|t| (l, t))).collect();
            self.blockgpu.queue_layers(layers);
        }
        if ch.table || ch.layout {
            self.blockgpu.set_table(self.blocks.gpu_table());
            // this frame records with the new table too
            self.blockgpu.begin_frame(slot);
        }
        for &id in &ch.deleted {
            // blocks of a deleted kind turn into air
            let mut n = 0;
            for c in self.world.overlay.chunks.values_mut() {
                for v in c.values_mut() {
                    if crate::blocks::block_id(*v) == id {
                        *v = 0;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                self.world.overlay.revision += 1;
            }
            if crate::blocks::block_id(self.ui.selected_block) == id {
                self.ui.selected_block = self.blocks.id("stone");
            }
        }
        self.wg.set_blocks(self.blocks.id_map(), self.blocks.glsl_header());
        if ch.remesh || !ch.deleted.is_empty() {
            self.world.invalidate_all();
        }
        Ok(())
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

    /// Save the seed, camera, time of day and edits (see `world/save.rs`); `path` becomes the
    /// open world.
    pub fn save_world(&mut self, path: &std::path::Path) -> Result<()> {
        use crate::world::save::{CameraState, WorldState};
        let state = WorldState {
            seed: self.seed,
            camera: CameraState { pos: self.camera.pos.to_array(), yaw: self.camera.yaw, pitch: self.camera.pitch },
            time_of_day: self.atmosphere.time_of_day,
        };
        crate::world::save::save(path, &state, &self.world.overlay, &self.blocks)?;
        self.saved_revision = self.world.overlay.revision;
        self.set_world_path(Some(path.to_path_buf()));
        Ok(())
    }

    /// Open a world file. Returns the names of blocks this build doesn't know (left to the
    /// generator).
    pub fn load_world(&mut self, path: &std::path::Path) -> Result<Vec<String>> {
        let (st, overlay, unknown) = crate::world::save::load(path, &self.blocks)?;
        self.seed = st.seed;
        self.camera.pos = DVec3::from(st.camera.pos);
        (self.camera.yaw, self.camera.pitch) = (st.camera.yaw, st.camera.pitch);
        self.atmosphere.time_of_day = st.time_of_day;
        self.world.overlay = overlay;
        self.saved_revision = self.world.overlay.revision;
        self.world.invalidate_all();
        self.world.edited.clear();
        self.fluid.reset();
        self.set_world_path(Some(path.to_path_buf()));
        Ok(unknown)
    }

    /// The solo preview as generator indices: a zone alone, or a surface biome with a zone that
    /// uses it, or a cave biome in every zone's caves.
    /// Jump to another layer: move the camera into it, then find somewhere to stand
    /// (`update_jump`, `jump_take_region`).
    pub fn start_layer_jump(&mut self, j: crate::ui::LayerJump) {
        use worldgen::compile::LayerKind;
        let c = &self.wg.compiled;
        let own = c.owned_ranges();
        let find = |kind: LayerKind, y: Option<f32>| {
            let of_kind = |i: &usize| c.layers[*i].kind == kind && own[*i].0 < own[*i].1;
            y.and_then(|y| (0..c.layers.len()).filter(of_kind).find(|&i| y >= own[i].0 && y < own[i].1))
                .or_else(|| (0..c.layers.len()).find(of_kind))
        };
        let (kind, depth) = match j {
            crate::ui::LayerJump::Surface => (LayerKind::Surface, None),
            crate::ui::LayerJump::Underground(d) => (LayerKind::Underground, Some(d)),
            crate::ui::LayerJump::Sky => (LayerKind::Sky, None),
        };
        let Some(li) = find(kind, depth) else {
            self.ui.toast(format!("This world has no {kind:?} layer").to_lowercase());
            return;
        };
        let l = &c.layers[li];
        let (lo, hi, name, sea) = (c.value(l.y_min), c.value(l.y_max), l.name.clone(), c.value(l.sea));
        self.fly_mode = true;
        match j {
            crate::ui::LayerJump::Underground(d) => {
                let y = d.clamp(lo + 16.0, hi - 16.0);
                self.camera.pos.y = y as f64;
                let id = self.request_jump_column(y);
                self.jump = Some(JumpState::Column { id, depth: y, tries: 40 });
            }
            _ => {
                // the probes scan the layer from the top down: stand on what they find
                self.camera.pos.y = if kind == LayerKind::Surface { hi.min(420.0) } else { hi - 64.0 } as f64;
                self.jump = Some(JumpState::Probe { wait: 2 * FIF as u32 + 2, sea, map_serial: self.probes.map_serial });
            }
        }
        self.ui.toast(format!("Jumped to the {name} layer"));
    }

    /// Read a column of blocks around height y at the camera (Underground layer jumps).
    fn request_jump_column(&mut self, y: f32) -> u64 {
        let (x, z) = (self.camera.pos.x.floor() as i32, self.camera.pos.z.floor() as i32);
        let y = y as i32;
        self.world.request_region(glam::IVec3::new(x, y - 96, z), glam::IVec3::new(x, y + 96, z))
    }

    /// A Surface or Sky layer jump: once the probes have looked at the new layer, stand on the
    /// highest ground they found (or keep flying where there is none).
    fn update_jump(&mut self) {
        let Some(JumpState::Probe { wait, sea, map_serial }) = &mut self.jump else { return };
        if *wait > 0 {
            *wait -= 1;
            return;
        }
        let (sea, map_serial) = (*sea, *map_serial);
        // the scan steps down at least 4 blocks: the ground is within a step above h
        let step = |cam_y: f64, h: f32| 4.0f32.max((cam_y as f32 - h) / 160.0).ceil();
        if let Some(h) = self.probes.camera.map(|c| c.height).filter(|h| *h > -1000.0) {
            self.camera.pos.y = (h + step(self.camera.pos.y, h) + 2.0).max(sea + 2.0) as f64;
            self.jump = None;
            log::info!("layer jump: ground at {h:.0} under the camera, standing at {:.0}", self.camera.pos);
            return;
        }
        // nothing under the camera (open sky): the nearest ground on a fresh minimap
        if self.probes.map_serial < map_serial + 2 {
            return;
        }
        self.jump = None;
        let Some(((ox, oz, sp), cells)) = &self.probes.map else { return };
        let n = crate::probe::MAP_N;
        let (cx, cz) = (self.camera.pos.x, self.camera.pos.z);
        let best = (0..n * n)
            .filter(|&i| cells[i].height > -1000.0)
            .map(|i| (ox + ((i % n) as f64 + 0.5) * sp, oz + ((i / n) as f64 + 0.5) * sp, cells[i].height))
            .min_by(|a, b| ((a.0 - cx).powi(2) + (a.1 - cz).powi(2)).total_cmp(&((b.0 - cx).powi(2) + (b.1 - cz).powi(2))));
        match best {
            Some((x, z, h)) => {
                let y = (h + step(self.camera.pos.y, h) + 2.0).max(sea + 2.0) as f64;
                self.camera.pos = glam::DVec3::new(x, y, z);
                log::info!("layer jump: nearest ground at {h:.0}, standing at {:.0}", self.camera.pos);
            }
            None => self.ui.toast("No ground nearby: flying".into()),
        }
    }

    /// An Underground layer jump: stand in the open pocket nearest the chosen depth, else try a
    /// column further along.
    fn jump_take_region(&mut self, r: RegionRead) -> Option<RegionRead> {
        let Some(JumpState::Column { id, depth, tries }) = self.jump else { return Some(r) };
        if r.id != id {
            return Some(r);
        }
        let col: Vec<u16> = (0..r.size.y).map(|y| r.get(r.min + glam::IVec3::new(0, y, 0))).collect();
        // air runs at least 3 tall standing on a solid block: (floor y, run length)
        let mut best: Option<(i32, i32)> = None;
        let mut y = 1;
        while (y as usize) < col.len() {
            if col[y as usize] == 0 && col[y as usize - 1] != 0 {
                let start = y;
                while (y as usize) < col.len() && col[y as usize] == 0 {
                    y += 1;
                }
                let run = y - start;
                let floor = r.min.y + start;
                if run >= 3 && best.is_none_or(|(f, _)| (floor as f32 - depth).abs() < (f as f32 - depth).abs()) {
                    best = Some((floor, run));
                }
            }
            y += 1;
        }
        match best {
            Some((floor, run)) => {
                self.camera.pos.y = floor as f64 + (run as f64 / 3.0).clamp(1.6, 4.0);
                self.jump = None;
                log::info!("layer jump: cave at {:.1}", self.camera.pos);
            }
            None if tries == 0 => {
                self.ui.toast(format!("No cave found near y {depth:.0}"));
                self.jump = None;
            }
            None => {
                self.camera.pos.x += 53.0;
                self.camera.pos.z += 31.0;
                let id = self.request_jump_column(depth);
                self.jump = Some(JumpState::Column { id, depth, tries: tries - 1 });
            }
        }
        None
    }

    /// Whether a layer jump is still looking for somewhere to stand.
    pub fn jumping(&self) -> bool {
        self.jump.is_some()
    }

    /// The cloud sea's settings (World Output).
    pub fn cloud_sea(&self) -> crate::renderer::CloudSea {
        let c = &self.wg.compiled;
        crate::renderer::CloudSea { base: c.value(c.cloud_sea[0]), top: c.value(c.cloud_sea[1]), density: c.value(c.cloud_sea[2]) }
    }

    /// The generator layers that have pipelines, as the batch recorder dispatches them.
    fn layer_passes(&self) -> Vec<crate::world::gpu::LayerPass> {
        let own = self.wg.compiled.owned_ranges();
        self.wg
            .compiled
            .layers
            .iter()
            .zip(&self.gen_layers)
            .zip(own)
            .map(|((l, h), own)| crate::world::gpu::LayerPass { terrain: self.pm.get(h[0]), props: self.pm.get(h[1]), param_base: l.param_base, own })
            .collect()
    }

    /// Heights the probes scan for the minimap's surface in layer `li` (bottom, top): the
    /// surface's usual terrain heights, the whole sky, underground the floor below the camera.
    fn probe_scan(&self, li: usize, cam_y: f32) -> (f32, f32) {
        let c = &self.wg.compiled;
        let Some(l) = c.layers.get(li) else { return (-128.0, 400.0) };
        let (lo, hi) = (c.value(l.y_min), c.value(l.y_max));
        match l.kind {
            worldgen::compile::LayerKind::Surface => (lo.max(-128.0), hi.min(400.0)),
            worldgen::compile::LayerKind::Sky => (lo, hi),
            worldgen::compile::LayerKind::Underground => ((cam_y - 256.0).max(lo), (cam_y + 32.0).min(hi)),
        }
    }

    fn resolve_solo(&self) -> (Option<u32>, Option<u32>) {
        let c = &self.wg.compiled;
        match &self.node_ui.solo {
            Some((worldgen::GraphKind::Zone, name)) => (c.zones.iter().position(|z| &z.name == name).map(|i| i as u32), None),
            Some((worldgen::GraphKind::Biome, name)) => {
                let Some(b) = c.biomes.iter().position(|x| &x.name == name).map(|i| i as u32) else { return (None, None) };
                let zone = if c.biomes[b as usize].cave { None } else { c.zones.iter().position(|z| z.biomes.contains(&b)).map(|i| i as u32) };
                (zone, Some(b))
            }
            _ => (None, None),
        }
    }

    fn set_world_path(&mut self, path: Option<PathBuf>) {
        if let Some(w) = &self.window {
            match &path {
                Some(p) => w.set_title(&format!("Voxel Engine - {}", crate::world::save::name_of(p))),
                None => w.set_title("Voxel Engine"),
            }
        }
        self.world_path = path;
    }

    fn save_world_ui(&mut self, path: &std::path::Path) {
        match self.save_world(path) {
            Ok(()) => self.ui.toast(format!("Saved {}", path.display())),
            Err(e) => self.ui.toast(format!("Save failed: {e:#}")),
        }
        self.ui.refresh_worlds = true;
    }

    /// Blend fog colour/density toward the biome the camera is in.
    fn apply_environment(&mut self) {
        let Some(cam) = self.probes.camera else { return };
        let Some(b) = self.wg.compiled.biomes.get(cam.biome as usize) else { return };
        let k = if std::mem::take(&mut self.env_snap) { 1.0 } else { (self.dt * 1.5).min(1.0) };
        let target = Vec3::from(b.fog_color);
        self.atmosphere.fog_tint = self.atmosphere.fog_tint.lerp(target, k);
        self.atmosphere.fog_tint_strength = 0.55;
        self.atmosphere.grade_tint = self.atmosphere.grade_tint.lerp(Vec3::from(b.grade_tint), k);
        self.atmosphere.saturation += (b.saturation - self.atmosphere.saturation) * k;
        self.atmosphere.contrast += (b.contrast - self.atmosphere.contrast) * k;
        // weather: automatic (slow cycle on the animation clock) or the slider
        let weather = if self.ui.weather_auto {
            let t = self.anim_time as f64;
            let w = (t * 0.0041).sin() * 0.6 + (t * 0.0097 + 1.3).sin() * 0.4;
            ((w as f32 - 0.25) * 2.0).clamp(0.0, 1.0)
        } else {
            self.ui.weather
        };
        self.atmosphere.weather += (weather - self.atmosphere.weather) * k;
        let daylight = (self.atmosphere.colors().sun_dir.y * 5.0 + 0.5).clamp(0.0, 1.0);
        let target = crate::particles::Shares::for_biome(b.ambient, b.precipitation, b.particle_density, self.atmosphere.weather, daylight);
        self.particles.shares = self.particles.shares.lerp(target, if k >= 1.0 { 1.0 } else { (self.dt * 0.8).min(1.0) });
        let rain = self.particles.shares.rain;
        let dens = 0.0012 * b.fog_density.max(0.05) * self.ui.fog_scale * (1.0 + 1.5 * rain);
        self.atmosphere.fog_density += (dens - self.atmosphere.fog_density) * k;
    }

    fn read_pick(&mut self, slot: usize) {
        let words: Vec<u32> = self.renderer.per_slot[slot].pick_readback.read(0, 2);
        let payload = words[0];
        let depth = f32::from_bits(words[1]);
        let stats: Vec<u32> = self.renderer.per_slot[slot].stats_readback.read(0, 18);
        self.stats.model_groups = stats[8] + stats[14];
        self.stats.draw_groups = stats[0] + stats[11];
        self.stats.occluded_chunks = stats[17];
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
            distance: rel.length(),
        });
    }

    pub fn frame(&mut self) -> Result<()> {
        let now = Instant::now();
        self.dt = (now - self.last_frame).as_secs_f32().clamp(1e-5, 0.25);
        if self.script.is_some() {
            self.dt = 1.0 / 60.0;
        }
        if !self.anim_frozen {
            self.anim_time += self.dt;
        }
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
        // this slot's previous frame is done: its block table copy can be rewritten
        self.blockgpu.begin_frame(slot);
        self.renderer.block_table_addr = self.blockgpu.table_address(slot);
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
        let gen_pipes = |s: &Self| s.gen_layers.iter().map(|h| (s.pm.get(h[0]), s.pm.get(h[1]))).collect::<Vec<_>>();
        let gen_before = gen_pipes(self);
        let n = self.pm.hot_reload(&self.ctx, self.frame_counter, force);
        if n > 0 {
            self.ui.toast(format!("Reloaded {n} pipeline(s)"));
            if gen_pipes(self) != gen_before {
                self.world.invalidate_all();
            }
        }

        self.sync_prefabs()?;
        // ---- solo preview: one zone or biome everywhere ----
        let solo = self.resolve_solo();
        if solo != self.solo_applied {
            self.wgpu.set_solo(solo.0, solo.1);
            self.solo_applied = solo;
            self.world.invalidate_all();
        }
        // ---- world generator edits ----
        let (params_changed, built) = self.wg.update();
        if params_changed || built.iter().any(|(current, _)| *current) {
            self.prop_preview.stale = true;
        }
        if params_changed {
            self.wgpu.set_params(&self.wg.compiled.params);
            if self.wg.is_current() {
                self.world.invalidate_all();
            }
        }
        for (current, b) in built {
            if !current {
                b.destroy(&self.ctx.device);
                continue;
            }
            let n = b.layers.len();
            for l in b.layers {
                let existing = self.gen_layers.get(l.index).copied();
                let mut handles = Vec::with_capacity(3);
                for (k, ((pipeline, source), name)) in l.pipelines.into_iter().zip(l.sources).zip(crate::wg::PASS_NAMES).enumerate() {
                    let src = PipelineSource::ComputeSource { name: format!("{name}.comp"), source, defines: vec![] };
                    match existing {
                        Some(h) => self.pm.replace(h[k], src, pipeline, b.deps.clone(), self.frame_counter),
                        None => handles.push(self.pm.insert(name, src, pipeline, b.deps.clone())),
                    }
                }
                // a new layer (new layers come in order, after the existing ones)
                if existing.is_none() {
                    self.gen_layers.push([handles[0], handles[1], handles[2]]);
                }
            }
            self.wgpu.set_params(&self.wg.compiled.params);
            self.world.invalidate_all();
            self.ui.toast(format!(
                "{} generator ready in {:.0} ms ({n} layer{})",
                if b.optimized { "Optimised" } else { "Quick" },
                b.ms,
                if n == 1 { "" } else { "s" }
            ));
        }
        self.world.layers = self.wg.compiled.owned_ranges();
        // no sky light under the top of the underground layers
        let floor = (self.wg.compiled.layers.iter().zip(&self.world.layers))
            .filter(|(l, own)| l.kind == worldgen::compile::LayerKind::Underground && own.0 < own.1)
            .map(|(_, own)| own.1 as i32)
            .max()
            .unwrap_or(i32::MIN / 2);
        if floor != self.light.sky_floor {
            self.light.sky_floor = floor;
            self.light.mark_dirty();
        }
        self.world.camera_layer = self.wg.compiled.layer_near(self.camera.pos.y as f32);

        // ---- world results ----
        for (s, jobs, results) in self.wgpu.poll(completed) {
            for (i, j) in jobs.iter().enumerate() {
                if j.readback
                    && let Some(v) = self.wgpu.take_voxels(s, i) {
                        self.world.fill_region(j.key, j.preview, &v);
                    }
            }
            self.world.apply_results(s, &jobs, &results, value);
        }
        self.update_jump();
        for region in self.world.take_finished_regions() {
            let Some(region) = self.jump_take_region(region) else { continue };
            let Some(region) = self.prop_preview.take(region, &self.blocks) else { continue };
            let Some(region) = self.fluid.region_ready(region) else { continue };
            let Some(region) = self.script_take_region(region) else { continue };
            self.tools.region_ready(region, &mut self.world, &self.blocks, &mut self.ui);
        }

        // ---- input, camera, editing ----
        self.script_step()?;
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
        // ---- flowing water ----
        let edited = std::mem::take(&mut self.world.edited);
        if !edited.is_empty() {
            self.fluid.edited(&edited, &self.world);
        }
        self.fluid.enabled = self.ui.water_flow;
        self.fluid.update(self.dt, &mut self.world, &self.blocks);
        self.atmosphere.update(self.dt);
        self.world.update_desired(self.camera.pos);
        self.world.process_retirement(value);
        self.world.update_stats();

        // ---- UI ----
        let actions = self.build_ui()?;
        self.apply_ui_actions(actions, value)?;
        self.sync_blocks(slot)?;
        self.prop_preview.update(&mut self.world);

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
        // block texture layers changed by the editors
        self.blockgpu.record_uploads(&self.ctx, cb, slot);

        // light volume: images ready before the batch fills them
        self.light.record_init(&self.ctx, cb);
        if self.world.light.take_moved() {
            self.light.mark_dirty();
        }
        if self.world.coarse_light.take_moved() {
            self.light.mark_coarse_dirty();
        }

        // generation batch
        if self.wgpu.has_free_slot() && !self.world.frozen {
            let budget_jobs = (self.ui.gen_budget_ms / self.ms_per_job.max(0.002)) as usize;
            let mut max_jobs = budget_jobs.clamp(16, self.ui.jobs_per_frame.min(MAX_JOBS));
            if self.script.is_some() {
                // fixed batch sizes keep scripted runs deterministic
                max_jobs = self.ui.jobs_per_frame.min(MAX_JOBS);
            }
            let jobs = self.world.pick_jobs(self.camera.pos, self.camera.forward(), max_jobs, value);
            self.batch_sizes[slot] = jobs.len();
            if jobs.iter().any(|j| j.light & crate::world::gpu::GENJOB_LIGHT != 0) {
                self.light.mark_dirty();
            }
            if jobs.iter().any(|j| j.light & crate::world::gpu::GENJOB_COARSE != 0) {
                self.light.mark_coarse_dirty();
            }
            if !jobs.is_empty() {
                let pipes = WorldPipelines {
                    layers: self.layer_passes(),
                    apply_edits: self.pm.get(self.renderer.pipes.apply_edits),
                    mesh_count: self.pm.get(self.renderer.pipes.mesh_count),
                    mesh_write: self.pm.get(self.renderer.pipes.mesh_write),
                    prefix: self.pm.get(self.renderer.pipes.prefix),
                    light_fill: self.pm.get(self.light.fill),
                    coarse_fill: self.pm.get(self.light.coarse_fill),
                    layout: self.pm.layout,
                    light_blocks_img: self.light.blocks_img,
                    coarse_cells_img: self.light.coarse_cells_img,
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
                    self.blockgpu.table_address(slot),
                    self.seed,
                    &mut self.profiler,
                    profiler_slot,
                ) {
                    for j in rest {
                        if j.readback {
                            for r in self.world.region_reads.iter_mut().filter(|r| r.preview == j.preview) {
                                r.requeue(j.key.pos);
                            }
                        } else if let Some(c) = self.world.chunks.get_mut(&j.key) {
                            c.in_flight = false;
                        }
                    }
                }
            }
        }

        // light propagation (after this frame's fills)
        if let Some(origin) = self.world.light.origin_block()
            && self.ui.block_light
        {
            self.light.record(&self.ctx, cb, &self.pm, self.bindless.set, origin, self.world.coarse_light.origin_block(), self.blockgpu.table_address(slot), &mut self.profiler, profiler_slot);
        }

        // ambient particles and weather
        let particle_draw = if self.ui.particles {
            let t = self.anim_time;
            let wind = glam::Vec2::new((t * 0.05).sin(), (t * 0.037).cos()) * (0.6 + 1.6 * self.atmosphere.weather);
            let light_blocks = if self.ui.block_light && self.light.ready() { self.light.blocks_img } else { u32::MAX };
            let push = self.particles.push(
                self.renderer.per_slot[slot].frame_buf.address,
                self.blockgpu.table_address(slot),
                light_blocks,
                wind,
                self.taa_frame as u32,
                self.dt,
            );
            self.particles.record_sim(&self.ctx, cb, &self.pm, self.bindless.set, &push, &mut self.profiler, profiler_slot);
            Some((self.pm.get(self.particles.draw), push))
        } else {
            None
        };

        // biome probes (camera environment + minimap), in the camera's layer
        if !self.gen_layers.is_empty() {
            let pos = self.camera.pos;
            let n = self.probes.prepare(slot, pos, self.frame_counter);
            let li = self.wg.compiled.layer_near(pos.y as f32).min(self.gen_layers.len() - 1);
            let scan = self.probe_scan(li, pos.y as f32);
            unsafe {
                self.ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pm.get(self.gen_layers[li][2]));
                self.ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.pm.layout, 0, &[self.bindless.set], &[]);
            }
            let push = crate::world::gpu::GenPush {
                jobs: self.probes.input[slot].address,
                voxels: self.wgpu.voxels.address,
                results: self.probes.output[slot].address,
                edits: self.probes.input[slot].address,
                params: self.wgpu.params.address + self.wg.compiled.layers.get(li).map_or(0, |l| l.param_base as u64 * 4),
                columns: self.wgpu.columns.address,
                prefab_table: self.wgpu.prefab_addrs.0,
                prefab_data: self.wgpu.prefab_addrs.1,
                num_jobs: n,
                seed: self.seed,
                edit_count: 0,
                pad: 0,
                blocks: self.blockgpu.table_address(slot),
                prop_prims: 0,
                prop_segs: 0,
                job_list: 0,
                own_min: scan.0,
                own_max: scan.1,
                solo_zone: self.wgpu.solo[0],
                solo_biome: self.wgpu.solo[1],
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
        let unjittered = self.camera.matrices(aspect);
        let mut m = self.camera.matrices(aspect);
        if self.ui.taa {
            // sub-pixel jitter (culling keeps the unjittered frustum)
            let j = crate::taa::jitter_matrix(crate::taa::jitter(self.taa_frame), extent);
            m.view_proj = j * m.view_proj;
            m.proj = j * m.proj;
            m.inv_view_proj = m.view_proj.inverse();
            if !self.prev_taa || (self.camera.pos - self.prev_cam_pos).length() > 64.0 {
                self.taa.reset = true;
            }
        }
        self.prev_taa = self.ui.taa;
        let pick_px = if self.fly_mode || (self.window.is_none() && self.script.is_none()) {
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
        self.prev_view_proj = unjittered.view_proj;
        self.prev_cam_pos = self.camera.pos;

        let (sea, cam_y) = (self.cloud_sea(), self.camera.pos.y as f32);
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
            self.shadow_mask,
            {
                // weather thickens the clouds; up in (or above) the cloud sea they're hidden
                let mut c = self.ui.clouds;
                c.coverage = (c.coverage + 0.3 * self.atmosphere.weather).min(1.0);
                c.enabled &= !(sea.density > 0.0 && cam_y > sea.base + sea.amp());
                c
            },
            Some(sea).filter(|s| s.visible_from(cam_y)),
            self.ui.debug_view == crate::ui::DEBUG_OVERDRAW,
            particle_draw,
            self.ui.occlusion,
        );
        if self.ui.taa {
            let frame_addr = self.renderer.per_slot[slot].frame_buf.address;
            self.taa.record(&self.ctx, cb, &self.pm, self.bindless.set, frame_addr, self.renderer.hdr_tex, extent, self.frame_counter, self.taa_frame, &mut self.profiler, profiler_slot);
        }
        if self.ui.bloom {
            let src = self.scene_output();
            self.bloom.record(&self.ctx, cb, &self.pm, self.bindless.set, src, extent, self.ui.bloom_threshold, self.atmosphere.exposure, &mut self.profiler, profiler_slot);
        }

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
                if draw_ui
                    && let Some(f) = imgui_frame.take()
                        && let Err(e) = unsafe { ir.cmd_draw(cb, f) } {
                            log::error!("imgui draw: {e}");
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
        self.taa_frame += 1;
        self.input.end_frame();
        if let Some(batch) = retire_batch
            && let Some(ir) = self.imgui.renderer.as_mut() {
                ir.wait_for_texture_retirements(batch).map_err(|e| anyhow::anyhow!("{e}"))?;
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
            if let Some(f) = self.opts.frames
                && self.frame_counter >= f {
                    // flush outstanding screenshot
                    self.timeline.wait(&self.ctx, self.frame_counter)?;
                    for s in 0..FIF {
                        self.finish_screenshot(s)?;
                    }
                    self.quit = true;
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

    /// The image post-processing reads this frame: the TAA resolve when enabled.
    fn scene_output(&self) -> u32 {
        if self.ui.taa { self.taa.sampled[self.taa.output_index(self.frame_counter)] } else { self.renderer.hdr_tex }
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
            // previous frame's unjittered view-projection, re-based on this frame's camera
            prev_view_proj: (self.prev_view_proj * Mat4::from_translation((self.camera.pos - self.prev_cam_pos).as_vec3())).to_cols_array_2d(),
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
            time: self.anim_time,
            dt: self.dt,
            frame_index: self.frame_counter as u32,
            debug_mode: self.ui.debug_view,
            flags,
            pick_pixel: pick_px.0.min(0xFFFF) | (pick_px.1.min(0xFFFF) << 16),
            block_tex_albedo: self.blockgpu.albedo_tex,
            block_tex_normal: self.blockgpu.normal_tex,
            // what the tonemap pass reads: the TAA resolve when enabled
            hdr_tex: self.scene_output(),
            depth_tex: self.renderer.depth_tex,
            exposure: [
                self.atmosphere.exposure,
                self.atmosphere.time_of_day,
                if self.ui.bloom { self.ui.bloom_strength } else { 0.0 },
                if self.ui.grading { self.atmosphere.contrast } else { 1.0 },
            ],
            shadow_vp: [[[0.0; 4]; 4]; 4],
            shadow_planes: [[0.0; 4]; 24],
            shadow_splits: [0.0; 4],
            cam_forward: [0.0; 4],
            shadow_tex: self.renderer.shadow_tex,
            shadow_res: crate::renderer::SHADOW_RES,
            light_tex: self.light.sampled,
            bloom_tex: self.bloom.output(),
            light_origin: match self.world.light.origin_block() {
                Some(o) if self.ui.block_light && self.light.ready() => {
                    let r = o - m.cam_block;
                    [r.x, r.y, r.z, 1]
                }
                _ => [0; 4],
            },
            coarse_origin: match self.world.coarse_light.origin_block() {
                Some(o) if self.ui.block_light && self.light.ready() => {
                    let r = o - m.cam_block;
                    [r.x, r.y, r.z, 1]
                }
                _ => [0; 4],
            },
            coarse_tex: self.light.coarse_sampled,
            overdraw_img: self.renderer.overdraw_img,
            sky_floor: self.light.sky_floor as f32,
            pad: 0,
            shadow_texel: [0.0; 4],
            grade: if self.ui.grading {
                [self.atmosphere.grade_tint.x, self.atmosphere.grade_tint.y, self.atmosphere.grade_tint.z, self.atmosphere.saturation]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            },
        };
        let mut fd = fd;
        let light = if sky.sun_dir.y > -0.05 { sky.sun_dir } else { -sky.sun_dir };
        let aspect = extent.width as f32 / extent.height.max(1) as f32;
        let res = crate::renderer::SHADOW_RES;
        let mut casc = crate::camera::fit_cascades(&self.camera, aspect, light, self.ui.shadow_distance, res);
        let fitted = casc.radius;
        self.shadow_mask = if self.ui.shadows { 0b1111 } else { 0 };
        let changes = std::mem::take(&mut self.world.caster_changes);
        if self.ui.shadows && self.ui.shadow_cache {
            // what each cached cascade needs: (radius, forced redraw, optional redraw)
            let mut need = [(0.0f32, false, false); 2];
            for (k, &c) in CACHED_CASCADES.iter().enumerate() {
                let margin = SHADOW_CACHE_MARGIN[k];
                let reach = crate::camera::slice_reach(&self.camera, aspect, casc.splits[c]);
                let radius = ((reach + margin) * 16.0).ceil() / 16.0;
                let sc = &mut self.shadow_cache[k];
                if sc.valid && !sc.dirty {
                    // a changed chunk inside the window's footprint (light right/up axes)?
                    let (ax, ay) = sc.axes;
                    sc.dirty = changes.iter().any(|&(o, size)| {
                        let half = size as f32 * 0.5;
                        let d = (o.as_dvec3() + DVec3::splat(half as f64) - sc.at).as_vec3();
                        let reach = sc.radius + half * 1.74;
                        d.dot(ax).abs() <= reach && d.dot(ay).abs() <= reach
                    });
                }
                let moved = (self.camera.pos - sc.at).length() as f32 > margin;
                let forced = !sc.valid || moved || sc.radius != radius;
                let turned = sc.light.dot(light.normalize()) < 0.1f32.to_radians().cos();
                let changed = sc.dirty && self.frame_counter >= sc.drawn + SHADOW_CACHE_MIN_FRAMES;
                need[k] = (radius, forced, turned || changed);
            }
            // at most one optional redraw per frame (the oldest), so costs don't pile up
            let optional = (0..2).filter(|&k| need[k].2 && !need[k].1).min_by_key(|&k| self.shadow_cache[k].drawn);
            for (k, &c) in CACHED_CASCADES.iter().enumerate() {
                let (radius, forced, _) = need[k];
                let sc = &mut self.shadow_cache[k];
                if forced || optional == Some(k) {
                    let o = crate::camera::light_ortho(self.camera.pos, self.camera.pos, radius, light, res);
                    *sc = ShadowCache {
                        valid: true,
                        at: self.camera.pos,
                        view_proj: o.view_proj,
                        light: light.normalize(),
                        radius,
                        axes: crate::camera::light_axes(light),
                        drawn: self.frame_counter,
                        dirty: false,
                    };
                    casc.planes[c] = o.planes;
                } else {
                    self.shadow_mask &= !(1 << c);
                }
                // the matrix is relative to where the camera was when the cascade was drawn
                casc.view_proj[c] = sc.view_proj * Mat4::from_translation((self.camera.pos - sc.at).as_vec3());
                casc.radius[c] = radius;
            }
        } else {
            for sc in &mut self.shadow_cache {
                sc.valid = false;
            }
        }
        for (c, fitted) in fitted.iter().enumerate() {
            fd.shadow_vp[c] = casc.view_proj[c].to_cols_array_2d();
            for p in 0..6 {
                fd.shadow_planes[c * 6 + p] = casc.planes[c][p].to_array();
            }
            // receiver offset in texels, scaled like the slice-fitted cascades
            fd.shadow_texel[c] = casc.splits[c] * 2.2 / res as f32 * casc.radius[c] / fitted;
        }
        fd.shadow_splits = casc.splits;
        let fw = self.camera.forward();
        let night = (1.0 - ((sky.sun_dir.y.abs() - 0.02) / 0.1).clamp(0.0, 1.0)) * 0.6;
        fd.cam_forward = [fw.x, fw.y, fw.z, self.ui.shadow_strength * (1.0 - night)];
        if self.ui.shadows {
            fd.flags |= crate::renderer::FLAG_SHADOWS;
        }
        if self.ui.taa {
            fd.flags |= crate::renderer::FLAG_TAA;
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
        let mut block_act = crate::block_editor::BlockEditorActions::default();
        if self.ui.show_ui {
            self.node_ui.draw(ui, &mut self.wg, &self.blocks);
            self.tools.draw_ui(ui, &mut self.world, &self.blocks, &mut self.ui, self.pick);
            if let Some(id) = self.ui.edit_block.take() {
                self.block_ui.open = true;
                self.block_ui.select(&self.blocks, id);
            }
            let atlas = self.atlas_tex.map(|id| self.blockgpu.atlas_ref(id));
            block_act = self.block_ui.draw(ui, &mut self.blocks, atlas, &mut self.tex_ui, self.anim_time);
            self.tex_ui.draw(ui, &mut self.blocks, self.anim_time);
            // the prop preview follows the Prop graph open in the node editor
            let cur = &self.node_ui.current;
            if self.node_ui.open && cur.0 == worldgen::GraphKind::Prop && self.prop_preview_closed.as_deref() != Some(cur.1.as_str()) {
                let names = self.wg.project.names(worldgen::GraphKind::Prop);
                if let (Some(idx), Some(g)) = (names.iter().position(|n| *n == cur.1), self.wg.project.get(worldgen::GraphKind::Prop, &cur.1)) {
                    let out = g.output_node();
                    let b = |k: &str, d: f32| out.and_then(|o| o.params.get(k)).map(|v| v.as_f32()).unwrap_or(d);
                    self.prop_preview.show(&cur.1, idx as u32, [b("radius", 6.0), b("height", 12.0), b("below", 2.0)]);
                    self.prop_preview.open = true;
                }
            } else {
                self.prop_preview.open = false;
            }
            if self.prop_preview.open {
                self.prop_preview.draw(ui, atlas);
                if !self.prop_preview.open {
                    self.prop_preview_closed = Some(cur.1.clone());
                }
            }
            // opening another graph forgets that a preview was closed
            if cur.0 != worldgen::GraphKind::Prop || self.prop_preview_closed.as_deref() != Some(cur.1.as_str()) {
                self.prop_preview_closed = None;
            }
        }
        self.tools.draw_overlay(ui, &self.camera, self.pick);
        let unsaved = self.world.overlay.revision != self.saved_revision;
        let cam_layer = self.wg.compiled.layers.get(self.wg.compiled.layer_near(self.camera.pos.y as f32));
        let layer_name = cam_layer.map(|l| l.name.clone()).unwrap_or_default();
        let map_sea = cam_layer.map_or(62.0, |l| self.wg.compiled.value(l.sea));
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
                camera_layer: &layer_name,
                map_sea,
                has_layers: (
                    self.wg.compiled.layers.iter().any(|l| l.kind == worldgen::compile::LayerKind::Underground),
                    self.wg.compiled.layers.iter().any(|l| l.kind == worldgen::compile::LayerKind::Sky),
                ),
                biome_names: &self.wg.compiled.biomes,
                zone_names: &self.wg.compiled.zones,
                vsync: self.swapchain.as_ref().map(|s| s.vsync),
                node_editor_open: &mut self.node_ui.open,
                block_editor_open: &mut self.block_ui.open,
                texture_editor_open: &mut self.tex_ui.open,
                tools_open: &mut self.tools.show_tools,
                bench_report: self.bench.as_ref().and_then(|b| b.report.clone()),
                world_path: self.world_path.as_deref(),
                unsaved,
                solo: &mut self.node_ui.solo,
            },
        );
        if let (Some(p), Some(w)) = (&mut self.imgui.platform, &self.window) {
            p.prepare_render(ui, w)?;
        }
        if std::mem::take(&mut self.prop_preview.place) {
            match self.prop_preview.to_clipboard() {
                Some(c) => {
                    self.ui.toast(format!("Prop '{}' in clipboard: use Paste (Tools > Select)", c.name));
                    self.tools.clipboard = Some(c);
                }
                None => self.ui.toast("The preview is empty".into()),
            }
        }
        if let Some(id) = block_act.select {
            self.ui.selected_block = id;
            self.tools.kind = crate::tools::ToolKind::Block;
        }
        if let Some(id) = block_act.place {
            // a few blocks in front of the camera (the mouse is over the editor, not the world)
            let f = self.camera.forward();
            let at = (self.camera.pos + f.as_dvec3() * 4.0).floor().as_ivec3();
            let v = self.blocks.oriented(id, IVec3::Y, f);
            let touched = self.world.overlay.apply("Place", &[(at, v)]);
            self.world.invalidate_blocks(&touched);
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
            self.fluid.reset();
            let touched = self.world.overlay.clear();
            self.world.invalidate_blocks(&touched);
        }
        if a.screenshot {
            self.screenshot_request = Some(PathBuf::from(format!("screenshots/shot_{}.png", chrono_stamp())));
        }
        if let Some(v) = a.set_vsync
            && let Some(sc) = &mut self.swapchain {
                self.ctx.wait_idle();
                let e = sc.extent;
                sc.recreate(&self.ctx, e.width, e.height, v)?;
            }
        if a.undo
            && let Some((_, touched)) = self.world.overlay.undo() {
                self.world.invalidate_blocks(&touched);
            }
        if a.redo
            && let Some((_, touched)) = self.world.overlay.redo() {
                self.world.invalidate_blocks(&touched);
            }
        if a.quit {
            self.quit = true;
        }
        if let Some(j) = a.layer_jump {
            self.start_layer_jump(j);
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
            match self.world_path.clone() {
                Some(p) => self.save_world_ui(&p),
                None => self.ui.show_worlds = true,
            }
        }
        if let Some(p) = &a.save_world_as {
            self.save_world_ui(p);
        }
        if let Some(p) = &a.open_world {
            match self.load_world(p) {
                Ok(unknown) if unknown.is_empty() => self.ui.toast(format!("Opened {}", p.display())),
                Ok(unknown) => self.ui.toast(format!("Opened {} (unknown blocks: {})", p.display(), unknown.join(", "))),
                Err(e) => self.ui.toast(format!("Open failed: {e:#}")),
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
        self.taa.resize(&self.ctx, &mut self.bindless, extent)?;
        self.bloom.resize(&self.ctx, &mut self.bindless, extent)?;
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
        self.light.destroy(&self.ctx, &mut self.bindless);
        self.taa.destroy(&self.ctx, &mut self.bindless);
        self.particles.destroy(&self.ctx);
        self.bloom.destroy(&self.ctx, &mut self.bindless);
        self.wg.destroy(&self.ctx);
        self.blockgpu.destroy(&self.ctx);
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

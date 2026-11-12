mod atmosphere;
mod blocks;
mod camera;
mod engine;
mod input;
mod renderer;
mod shaders;
mod textures;
mod ui;
mod world;
mod worldgen_default;

use std::sync::Arc;

use anyhow::Result;
use engine::{Engine, Options};
use glam::DVec3;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

struct App {
    opts: Options,
    engine: Option<Engine>,
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.engine.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Voxel Engine")
            .with_inner_size(winit::dpi::PhysicalSize::new(self.opts.width, self.opts.height));
        let window = match el.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.error = Some(e.into());
                el.exit();
                return;
            }
        };
        match Engine::new(self.opts.clone(), Some(window)) {
            Ok(e) => self.engine = Some(e),
            Err(e) => {
                self.error = Some(e);
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(engine) = &mut self.engine else { return };
        match &event {
            WindowEvent::CloseRequested => {
                el.exit();
                return;
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = engine.frame() {
                    self.error = Some(e);
                    el.exit();
                    return;
                }
                if engine.quit {
                    el.exit();
                    return;
                }
            }
            _ => {}
        }
        engine.handle_window_event(&event);
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let (Some(engine), DeviceEvent::MouseMotion { delta }) = (&mut self.engine, event) {
            engine.handle_mouse_motion(delta);
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(e) = &self.engine {
            if let Some(w) = &e.window {
                w.request_redraw();
            }
        }
    }
}

fn parse_args() -> Options {
    let mut o = Options { width: 1600, height: 900, validation: cfg!(debug_assertions), ..Default::default() };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    let next = |i: &mut usize| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_default()
    };
    while i < args.len() {
        match args[i].as_str() {
            "--headless" => o.headless = true,
            "--validate" => o.validation = true,
            "--no-validate" => o.validation = false,
            "--no-ui" => o.no_ui = true,
            "--frames" => o.frames = next(&mut i).parse().ok(),
            "--screenshot" => o.screenshot = Some(next(&mut i).into()),
            "--size" => {
                let s = next(&mut i);
                if let Some((w, h)) = s.split_once('x') {
                    o.width = w.parse().unwrap_or(1600);
                    o.height = h.parse().unwrap_or(900);
                }
            }
            "--pos" => {
                let v: Vec<f64> = next(&mut i).split(',').filter_map(|s| s.parse().ok()).collect();
                if v.len() == 3 {
                    o.cam_pos = Some(DVec3::new(v[0], v[1], v[2]));
                }
            }
            "--look" => {
                let v: Vec<f32> = next(&mut i).split(',').filter_map(|s| s.parse().ok()).collect();
                if v.len() == 2 {
                    o.cam_angles = Some((v[0], v[1]));
                }
            }
            "--time" => o.time_of_day = next(&mut i).parse().ok(),
            "--debug-view" => o.debug_view = next(&mut i).parse().ok(),
            "--seed" => o.seed = next(&mut i).parse().ok(),
            other => eprintln!("unknown argument {other}"),
        }
        i += 1;
    }
    o
}

/// Write every block texture as `<dir>/<name>.png` plus an atlas preview (texture pack template).
fn export_textures(dir: &str) -> Result<()> {
    let reg = blocks::BlockRegistry::new();
    std::fs::create_dir_all(dir)?;
    let n = reg.textures.len();
    let cols = 12usize;
    let rows = n.div_ceil(cols);
    let t = textures::TEX;
    let mut atlas = image::RgbaImage::new((cols * t) as u32, (rows * t * 2) as u32);
    for (i, name) in reg.textures.iter().enumerate() {
        let tex = textures::generate(name);
        let albedo = tex.albedo_rgba8();
        let normal = tex.normal_rgba8();
        image::save_buffer(format!("{dir}/{name}.png"), &albedo, t as u32, t as u32, image::ColorType::Rgba8)?;
        let (cx, cy) = ((i % cols) * t, (i / cols) * t * 2);
        for y in 0..t {
            for x in 0..t {
                let o = (y * t + x) * 4;
                atlas.put_pixel((cx + x) as u32, (cy + y) as u32, image::Rgba([albedo[o], albedo[o + 1], albedo[o + 2], 255]));
                atlas.put_pixel((cx + x) as u32, (cy + t + y) as u32, image::Rgba([normal[o], normal[o + 1], normal[o + 2], 255]));
            }
        }
    }
    atlas.save(format!("{dir}/_atlas.png"))?;
    println!("exported {n} textures to {dir}");
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Some(pos) = std::env::args().position(|a| a == "--export-textures") {
        let dir = std::env::args().nth(pos + 1).unwrap_or_else(|| "assets/textures_export".into());
        return export_textures(&dir);
    }
    let opts = parse_args();
    if opts.headless {
        let mut opts = opts;
        if opts.frames.is_none() {
            opts.frames = Some(120);
        }
        let mut engine = Engine::new(opts, None)?;
        while !engine.quit {
            engine.frame()?;
        }
        let errors = vkcore::context::VALIDATION_ERRORS.load(std::sync::atomic::Ordering::Relaxed);
        log::info!("headless run finished: {} frames, {} validation errors", engine.frame_counter, errors);
        return Ok(());
    }
    let el = EventLoop::new()?;
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App { opts, engine: None, error: None };
    el.run_app(&mut app)?;
    drop(app.engine.take());
    if let Some(e) = app.error {
        return Err(e);
    }
    Ok(())
}

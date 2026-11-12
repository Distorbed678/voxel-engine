//! World-generation system: owns the node graph project, compiles it to GLSL, builds the GPU
//! pipelines on a background thread and keeps live parameters / biome tables in sync.
//!
//! Change flow:
//!   graph edited -> `mark_dirty()` -> next frame `update()` runs the (fast, CPU) graph compiler
//!     * same GLSL as running   -> only the parameter buffer changes -> regenerate chunks now
//!     * different GLSL         -> background thread: shaderc + vkCreateComputePipelines
//!                                 -> swap pipelines -> regenerate chunks

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use vkcore::{Buffer, ShaderCompiler, ShaderKind, VkContext, vk};
use worldgen::{Compiled, Project};

use crate::engine::compose_source;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuBiome {
    pub grass: [f32; 4],
    pub foliage: [f32; 4],
    pub water: [f32; 4],
    pub map: [f32; 4],
    pub zone: [f32; 4],
}

pub const MAX_BIOMES: usize = 128;

struct Job {
    id: u64,
    terrain: String,
    props: String,
    probe: String,
    /// skip the quick tier and go straight to the optimised build
    optimized_only: bool,
}

pub struct BuiltPipelines {
    pub id: u64,
    /// fully inlined (fast generation) build; quick builds are followed by an optimised one
    pub optimized: bool,
    pub terrain: vk::Pipeline,
    pub props: vk::Pipeline,
    pub probe: vk::Pipeline,
    pub terrain_source: String,
    pub props_source: String,
    pub probe_source: String,
    pub deps: Vec<PathBuf>,
    pub ms: f32,
}

pub struct WorldgenSystem {
    pub project: Project,
    pub dir: PathBuf,
    pub compiled: Compiled,
    /// generated code the running pipelines were built from
    pub active_source: String,
    pending_source: Option<String>,
    next_id: u64,
    pending_id: Option<u64>,
    optimizing_id: Option<u64>,
    tx: Sender<Job>,
    rx: Receiver<std::result::Result<BuiltPipelines, (u64, String)>>,
    /// The background shader build thread (joined on destroy: it uses the device).
    thread: Option<std::thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    pub dirty: bool,
    pub last_error: Option<String>,
    pub last_compile_ms: f32,
    pub last_build_ms: f32,
    pub biome_table: Buffer,
    pub block_ids: HashMap<String, u32>,
    /// Block defines and tables (`BlockRegistry::glsl_header`), put in front of the compiled
    /// graph code, so a block change rebuilds the generator like a graph edit.
    pub blocks_header: String,
    pub prefab_info: HashMap<String, (u32, [i32; 3])>,
    pub prefab_names: Vec<String>,
    pub unsaved: bool,
    last_change: Instant,
    pub autosave: bool,
    pub revision: u64,
}

impl WorldgenSystem {
    pub fn new(
        ctx: &VkContext,
        shader_root: &Path,
        layout: vk::PipelineLayout,
        block_ids: HashMap<String, u32>,
        blocks_header: String,
        prefab_info: HashMap<String, (u32, [i32; 3])>,
        dir: PathBuf,
    ) -> Result<Self> {
        let project = match Project::load_dir(&dir) {
            Ok(p) => {
                log::info!("loaded world generator from {}", dir.display());
                p
            }
            Err(e) => {
                log::info!("using built-in world generator ({e})");
                let p = worldgen::defaults::default_project();
                if let Err(e) = p.save_dir(&dir) {
                    log::warn!("could not save default generator: {e}");
                }
                p
            }
        };
        let (tx, job_rx) = channel::<Job>();
        let (res_tx, rx) = channel();
        let device = ctx.device.clone();
        let root = shader_root.to_path_buf();
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let thread = std::thread::Builder::new()
            .name("worldgen-compiler".into())
            .spawn(move || {
                let compiler = match ShaderCompiler::new(root) {
                    Ok(c) => c,
                    Err(e) => {
                        log::error!("worldgen compiler thread: {e}");
                        return;
                    }
                };
                let mut carry: Option<Job> = None;
                loop {
                    let mut job = match carry.take() {
                        Some(j) => j,
                        None => match job_rx.recv() {
                            Ok(j) => j,
                            Err(_) => return,
                        },
                    };
                    // only the newest job matters
                    while let Ok(newer) = job_rx.try_recv() {
                        job = newer;
                    }
                    let tiers: &[bool] = if job.optimized_only { &[true] } else { &[false, true] };
                    for &optimized in tiers {
                        if thread_stop.load(Ordering::Relaxed) {
                            return;
                        }
                        let t0 = Instant::now();
                        let build = || -> Result<BuiltPipelines> {
                            let mut a = compiler.compile_source("worldgen_terrain.comp", &job.terrain, ShaderKind::Compute, &[])?;
                            let mut b = compiler.compile_source("worldgen_props.comp", &job.props, ShaderKind::Compute, &[])?;
                            let mut c = compiler.compile_source("worldgen_probe.comp", &job.probe, ShaderKind::Compute, &[])?;
                            let t_glsl = t0.elapsed().as_secs_f32() * 1000.0;
                            if !optimized {
                                vkcore::shader::spirv_dont_inline(&mut a.spirv, crate::shaders::worldgen_noinline);
                                vkcore::shader::spirv_dont_inline(&mut b.spirv, crate::shaders::worldgen_noinline);
                                vkcore::shader::spirv_dont_inline(&mut c.spirv, crate::shaders::worldgen_noinline);
                            }
                            // the driver compile dominates: build the pipelines in parallel
                            let (terrain, props, probe) = std::thread::scope(|sc| {
                                let h = sc.spawn(|| create_compute(&device, layout, &b.spirv));
                                let hp = sc.spawn(|| create_compute(&device, layout, &c.spirv));
                                let t = create_compute(&device, layout, &a.spirv);
                                (
                                    t,
                                    h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("props build thread panicked"))),
                                    hp.join().unwrap_or_else(|_| Err(anyhow::anyhow!("probe build thread panicked"))),
                                )
                            });
                            let probe = match probe {
                                Ok(p) => p,
                                Err(e) => {
                                    for p in [&terrain, &props].into_iter().flatten() {
                                        unsafe { device.destroy_pipeline(*p, None) };
                                    }
                                    return Err(e);
                                }
                            };
                            let terrain = match terrain {
                                Ok(t) => t,
                                Err(e) => {
                                    if let Ok(p) = props {
                                        unsafe { device.destroy_pipeline(p, None) };
                                    }
                                    unsafe { device.destroy_pipeline(probe, None) };
                                    return Err(e);
                                }
                            };
                            let props = match props {
                                Ok(p) => p,
                                Err(e) => {
                                    unsafe {
                                        device.destroy_pipeline(terrain, None);
                                        device.destroy_pipeline(probe, None);
                                    }
                                    return Err(e);
                                }
                            };
                            log::info!(
                                "{} generator: GLSL -> SPIR-V {:.0} ms, driver {:.0} ms",
                                if optimized { "optimised" } else { "quick" },
                                t_glsl,
                                t0.elapsed().as_secs_f32() * 1000.0 - t_glsl
                            );
                            let mut deps = a.deps;
                            deps.extend(b.deps);
                            Ok(BuiltPipelines {
                                id: job.id,
                                optimized,
                                terrain,
                                props,
                                probe,
                                terrain_source: job.terrain.clone(),
                                props_source: job.props.clone(),
                                probe_source: job.probe.clone(),
                                deps,
                                ms: t0.elapsed().as_secs_f32() * 1000.0,
                            })
                        };
                        let msg = build().map_err(|e| (job.id, format!("{e:#}")));
                        let failed = msg.is_err();
                        if res_tx.send(msg).is_err() {
                            return;
                        }
                        // a newer edit arrived: don't waste time optimising this one
                        if failed {
                            break;
                        }
                        if let Ok(newer) = job_rx.try_recv() {
                            carry = Some(newer);
                            break;
                        }
                    }
                }
            })?;
        let biome_table = Buffer::upload(ctx, "biome_table", (MAX_BIOMES * std::mem::size_of::<GpuBiome>()) as u64)?;
        let mut s = Self {
            project,
            dir,
            compiled: Compiled::default(),
            active_source: String::new(),
            pending_source: None,
            next_id: 1,
            pending_id: None,
            optimizing_id: None,
            tx,
            rx,
            thread: Some(thread),
            stop,
            dirty: true,
            last_error: None,
            last_compile_ms: 0.0,
            last_build_ms: 0.0,
            biome_table,
            block_ids,
            blocks_header,
            prefab_names: {
                let mut v: Vec<String> = prefab_info.keys().cloned().collect();
                v.sort();
                v
            },
            prefab_info,
            unsaved: false,
            last_change: Instant::now(),
            autosave: true,
            revision: 0,
        };
        s.recompile_cpu()?;
        Ok(s)
    }

    /// The block list changed (block editor): recompile with the new ids and header. The shaders
    /// are rebuilt in the background only if the generated code changed.
    pub fn set_blocks(&mut self, block_ids: HashMap<String, u32>, header: String) {
        if block_ids != self.block_ids || header != self.blocks_header {
            self.block_ids = block_ids;
            self.blocks_header = header;
            self.dirty = true;
        }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
        self.unsaved = true;
        self.last_change = Instant::now();
        self.revision += 1;
    }

    fn recompile_cpu(&mut self) -> Result<()> {
        let t0 = Instant::now();
        let mut c = worldgen::compile(&self.project, &self.block_ids, &self.prefab_info)?;
        c.source = format!("{}\n{}", self.blocks_header, c.source);
        self.last_compile_ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.compiled = c;
        self.write_biome_table();
        Ok(())
    }

    fn write_biome_table(&mut self) {
        let mut t = vec![GpuBiome::default(); MAX_BIOMES];
        // first zone that references each biome (for the zone overlay)
        let mut zone_of: HashMap<String, usize> = HashMap::new();
        for (zi, z) in self.compiled.zones.iter().enumerate() {
            if let Some(g) = self.project.get(worldgen::GraphKind::Zone, &z.name) {
                for n in &g.nodes {
                    if let Some(worldgen::Value::Ref(b)) = n.params.get("biome") {
                        zone_of.entry(b.clone()).or_insert(zi);
                    }
                }
            }
        }
        for (i, b) in self.compiled.biomes.iter().enumerate().take(MAX_BIOMES) {
            let zi = zone_of.get(&b.name).copied().unwrap_or(0);
            let zc = self.compiled.zones.get(zi).map(|z| z.map_color).unwrap_or([1.0; 3]);
            t[i] = GpuBiome {
                grass: [b.grass_tint[0], b.grass_tint[1], b.grass_tint[2], b.fog_density],
                foliage: [b.foliage_tint[0], b.foliage_tint[1], b.foliage_tint[2], zi as f32],
                water: [b.water_tint[0], b.water_tint[1], b.water_tint[2], 1.0],
                map: [b.map_color[0], b.map_color[1], b.map_color[2], 1.0],
                zone: [zc[0], zc[1], zc[2], 1.0],
            };
        }
        self.biome_table.write(0, &t);
    }

    pub fn sources(&self) -> (String, String, String) {
        compose_source(&self.compiled.source)
    }

    /// Call once per frame. Returns (params_changed, finished builds as (is_current, pipelines)).
    pub fn update(&mut self) -> (bool, Vec<(bool, BuiltPipelines)>) {
        let mut params_changed = false;
        if self.dirty {
            self.dirty = false;
            let (old_params, old_source) = (self.compiled.params.clone(), self.compiled.source.clone());
            match self.recompile_cpu() {
                Ok(()) => {
                    // a recompile that changes nothing (e.g. a new block no graph names yet)
                    // must not regenerate the world
                    params_changed = self.compiled.params != old_params || self.compiled.source != old_source;
                    let src = self.compiled.source.clone();
                    let already = self.pending_source.as_ref() == Some(&src) || (self.pending_source.is_none() && self.active_source == src);
                    if !already {
                        let (terrain, props, probe) = self.sources();
                        let id = self.next_id;
                        self.next_id += 1;
                        self.pending_id = Some(id);
                        self.pending_source = Some(src);
                        self.optimizing_id = None;
                        let _ = self.tx.send(Job { id, terrain, props, probe, optimized_only: false });
                    }
                    self.last_error = None;
                }
                Err(e) => self.last_error = Some(format!("{e:#}")),
            }
        }
        let mut built = Vec::new();
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Ok(b) => {
                    let current = Some(b.id) == self.pending_id || Some(b.id) == self.optimizing_id;
                    if current {
                        if Some(b.id) == self.pending_id {
                            self.pending_id = None;
                            self.active_source = self.pending_source.take().unwrap_or_default();
                            self.optimizing_id = if b.optimized { None } else { Some(b.id) };
                        } else if b.optimized {
                            self.optimizing_id = None;
                        }
                        self.last_build_ms = b.ms;
                    }
                    // stale builds are returned too so the caller can destroy them
                    built.push((current, b));
                }
                Err((id, e)) => {
                    if Some(id) == self.pending_id {
                        self.pending_id = None;
                        self.pending_source = None;
                    }
                    if Some(id) == self.optimizing_id {
                        self.optimizing_id = None;
                    }
                    log::error!("world generator shader failed:\n{e}");
                    self.last_error = Some(e);
                }
            }
        }
        if self.autosave && self.unsaved && self.last_change.elapsed().as_secs_f32() > 2.0 {
            self.save();
        }
        (params_changed, built)
    }

    pub fn is_building(&self) -> bool {
        self.pending_id.is_some()
    }

    pub fn is_optimizing(&self) -> bool {
        self.optimizing_id.is_some()
    }

    /// Build the fully optimised pipelines for the currently active source (used at startup,
    /// which boots with the quick build).
    pub fn request_optimized(&mut self) {
        let (terrain, props, probe) = self.sources();
        let id = self.next_id;
        self.next_id += 1;
        self.optimizing_id = Some(id);
        let _ = self.tx.send(Job { id, terrain, props, probe, optimized_only: true });
    }

    pub fn save(&mut self) {
        match self.project.save_dir(&self.dir) {
            Ok(()) => self.unsaved = false,
            Err(e) => log::error!("saving generator failed: {e:#}"),
        }
    }

    pub fn reload_from_disk(&mut self) -> Result<()> {
        self.project = Project::load_dir(&self.dir)?;
        self.mark_dirty();
        self.unsaved = false;
        Ok(())
    }

    pub fn reset_to_defaults(&mut self) {
        self.project = worldgen::defaults::default_project();
        self.mark_dirty();
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        // stop the build thread (it finishes the pipeline it's compiling, if any), then destroy
        // whatever it delivered that was never used
        self.stop.store(true, Ordering::Relaxed);
        drop(std::mem::replace(&mut self.tx, channel().0));
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        while let Ok(msg) = self.rx.try_recv() {
            if let Ok(b) = msg {
                unsafe {
                    ctx.device.destroy_pipeline(b.terrain, None);
                    ctx.device.destroy_pipeline(b.props, None);
                    ctx.device.destroy_pipeline(b.probe, None);
                }
            }
        }
        self.biome_table.destroy(ctx);
    }
}

fn create_compute(device: &ash::Device, layout: vk::PipelineLayout, spirv: &[u32]) -> Result<vk::Pipeline> {
    unsafe {
        let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let r = device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None);
        device.destroy_shader_module(module, None);
        Ok(r.map_err(|(_, e)| e)?[0])
    }
}

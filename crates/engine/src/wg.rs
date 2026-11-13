//! World-generation system: owns the node graph project, compiles it to GLSL, builds the GPU
//! pipelines on a background thread and keeps live parameters / biome tables in sync.
//!
//! Every layer of the world (surface, underground bands, sky) has its own generated code and its
//! own pipelines (terrain, props, probe), so an edit rebuilds only the layers whose code changed.
//!
//! Change flow:
//!   graph edited -> `mark_dirty()` -> next frame `update()` runs the (fast, CPU) graph compiler
//!     * same GLSL as running   -> only the parameter buffer changes -> regenerate chunks now
//!     * different GLSL         -> background thread: shaderc + vkCreateComputePipelines for the
//!                                 changed layers -> swap pipelines -> regenerate chunks

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

/// Biome ids are 12 bits (quads, model instances, the column buffer).
pub const MAX_BIOMES: usize = 4096;

/// The generator passes of a layer, in this order everywhere: terrain, props, probe.
pub const PASS_NAMES: [&str; 3] = ["worldgen_terrain", "worldgen_props", "worldgen_probe"];

struct Job {
    id: u64,
    /// the layers to build: (layer index, sources of its passes)
    layers: Vec<(usize, [String; 3])>,
    /// skip the quick tier and go straight to the optimised build
    optimized_only: bool,
}

/// A layer's built passes (terrain, props, probe) and their sources.
pub struct BuiltLayer {
    pub index: usize,
    pub pipelines: [vk::Pipeline; 3],
    pub sources: [String; 3],
}

pub struct BuiltPipelines {
    pub id: u64,
    /// fully inlined (fast generation) build; quick builds are followed by an optimised one
    pub optimized: bool,
    pub layers: Vec<BuiltLayer>,
    pub deps: Vec<PathBuf>,
    pub ms: f32,
}

impl BuiltPipelines {
    pub fn destroy(self, device: &ash::Device) {
        for l in self.layers {
            for p in l.pipelines {
                unsafe { device.destroy_pipeline(p, None) };
            }
        }
    }
}

pub struct WorldgenSystem {
    pub project: Project,
    pub dir: PathBuf,
    pub compiled: Compiled,
    /// per layer, the generated code its running pipelines were built from
    pub active: Vec<String>,
    /// per layer, the code being built (all layers, built or not)
    pending: Option<Vec<String>>,
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
                            let mut spirv = Vec::new();
                            let mut deps = Vec::new();
                            for (_, sources) in &job.layers {
                                for (name, src) in PASS_NAMES.iter().zip(sources) {
                                    let mut out = compiler.compile_source(&format!("{name}.comp"), src, ShaderKind::Compute, &[])?;
                                    if !optimized {
                                        vkcore::shader::spirv_dont_inline(&mut out.spirv, crate::shaders::worldgen_noinline);
                                    }
                                    deps.extend(out.deps);
                                    spirv.push(out.spirv);
                                }
                            }
                            let t_glsl = t0.elapsed().as_secs_f32() * 1000.0;
                            // the driver compile dominates: build the pipelines in parallel
                            let built: Vec<Result<vk::Pipeline>> = std::thread::scope(|sc| {
                                let hs: Vec<_> = spirv.iter().map(|code| sc.spawn(|| create_compute(&device, layout, code))).collect();
                                hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("pipeline build thread panicked")))).collect()
                            });
                            if let Some(i) = built.iter().position(|b| b.is_err()) {
                                let mut built = built;
                                let err = built.remove(i).unwrap_err();
                                for p in built.into_iter().flatten() {
                                    unsafe { device.destroy_pipeline(p, None) };
                                }
                                return Err(err);
                            }
                            let mut built = built.into_iter().map(|b| b.unwrap());
                            let layers = job
                                .layers
                                .iter()
                                .map(|(index, sources)| BuiltLayer {
                                    index: *index,
                                    pipelines: [built.next().unwrap(), built.next().unwrap(), built.next().unwrap()],
                                    sources: sources.clone(),
                                })
                                .collect();
                            log::info!(
                                "{} generator ({} layer(s)): GLSL -> SPIR-V {:.0} ms, driver {:.0} ms",
                                if optimized { "optimised" } else { "quick" },
                                job.layers.len(),
                                t_glsl,
                                t0.elapsed().as_secs_f32() * 1000.0 - t_glsl
                            );
                            deps.sort();
                            deps.dedup();
                            Ok(BuiltPipelines { id: job.id, optimized, layers, deps, ms: t0.elapsed().as_secs_f32() * 1000.0 })
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
            active: Vec::new(),
            pending: None,
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
        if let Err(e) = s.recompile_cpu() {
            // e.g. a generator saved before layers existed
            log::error!("world generator in {} does not compile ({e:#}); using the built-in one", s.dir.display());
            s.project = worldgen::defaults::default_project();
            s.recompile_cpu()?;
        }
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
        for l in &mut c.layers {
            l.source = format!("{}\n{}", self.blocks_header, l.source);
        }
        c.source = c.layers.iter().map(|l| l.source.as_str()).collect();
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

    /// Each layer's generated code (the blocks header included).
    pub fn layer_sources(&self) -> Vec<String> {
        self.compiled.layers.iter().map(|l| l.source.clone()).collect()
    }

    /// The pass sources (terrain, props, probe) of every layer.
    pub fn sources(&self) -> Vec<[String; 3]> {
        self.compiled.layers.iter().map(|l| compose_source(&l.source).into()).collect()
    }

    /// Whether the running pipelines were built from the current code.
    pub fn is_current(&self) -> bool {
        self.pending.is_none() && self.active == self.layer_sources()
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
                    let target = self.layer_sources();
                    // what the running pipelines (plus the build under way) will be
                    let base = self.pending.clone().unwrap_or_else(|| self.active.clone());
                    if target != base {
                        // build the layers whose code differs from what's running
                        let changed: Vec<(usize, [String; 3])> = target
                            .iter()
                            .enumerate()
                            .filter(|(i, src)| self.active.get(*i) != Some(*src))
                            .map(|(i, src)| (i, compose_source(src).into()))
                            .collect();
                        let id = self.next_id;
                        self.next_id += 1;
                        self.optimizing_id = None;
                        if changed.is_empty() {
                            // back to what's running (an edit undone, or layers removed)
                            self.pending_id = None;
                            self.pending = None;
                            self.active = target;
                        } else {
                            self.pending_id = Some(id);
                            self.pending = Some(target);
                            let _ = self.tx.send(Job { id, layers: changed, optimized_only: false });
                        }
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
                            self.active = self.pending.take().unwrap_or_default();
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
                        self.pending = None;
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
        let layers = self.sources().into_iter().enumerate().collect();
        let id = self.next_id;
        self.next_id += 1;
        self.optimizing_id = Some(id);
        let _ = self.tx.send(Job { id, layers, optimized_only: true });
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
                b.destroy(&ctx.device);
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

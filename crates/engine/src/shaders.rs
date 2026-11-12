//! Pipeline registry with GLSL hot reload (file watcher on `shaders/`).

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use notify::{RecursiveMode, Watcher};
use vkcore::pipeline::{self, BlendMode, GraphicsDesc};
use vkcore::{ShaderCompiler, ShaderKind, VkContext, vk};

#[derive(Clone)]
pub struct GraphicsState {
    pub color_formats: Vec<vk::Format>,
    pub blend: Vec<BlendMode>,
    pub depth_format: Option<vk::Format>,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: vk::CompareOp,
    pub cull: vk::CullModeFlags,
    pub polygon_mode: vk::PolygonMode,
    pub depth_bias: bool,
    pub depth_clamp: bool,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            color_formats: vec![],
            blend: vec![],
            depth_format: None,
            depth_test: false,
            depth_write: false,
            depth_compare: vk::CompareOp::GREATER_OR_EQUAL,
            cull: vk::CullModeFlags::NONE,
            polygon_mode: vk::PolygonMode::FILL,
            depth_bias: false,
            depth_clamp: false,
        }
    }
}

#[derive(Clone)]
pub enum PipelineSource {
    Compute { path: String, defines: Vec<(String, String)> },
    /// Compute shader whose main source is provided in memory (e.g. generated worldgen code).
    ComputeSource { name: String, source: String, defines: Vec<(String, String)> },
    Graphics { stages: Vec<(String, Vec<(String, String)>)>, state: GraphicsState },
}

struct Entry {
    name: String,
    source: PipelineSource,
    pipeline: vk::Pipeline,
    deps: Vec<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipelineHandle(usize);

pub struct PipelineManager {
    pub compiler: ShaderCompiler,
    pub layout: vk::PipelineLayout,
    entries: Vec<Entry>,
    changed: Arc<Mutex<HashSet<PathBuf>>>,
    _watcher: Option<notify::RecommendedWatcher>,
    /// Pipelines replaced by a reload, destroyed once `frame_value` completed.
    retired: Vec<(u64, vk::Pipeline)>,
    pub last_errors: Vec<String>,
    pub reload_count: u32,
}

/// Prop graph building (props pass): one thread runs it once per placed instance, so the quick
/// tier keeps it out of line. (The optimised tier inlines it too: out of line, the NVIDIA driver
/// compile took 30 s instead of 9.5 s.) The per-cell evaluation (`prop_eval`) is hot.
pub fn worldgen_cold(name: &str) -> bool {
    // (not the transform helpers xf_* / rot_*: with them and the build functions both out of line,
    // the NVIDIA driver miscompiled the quick tier and lost parts of props)
    const PREFIXES: &[&str] = &["prop_build", "tree_", "pb_", "sk_", "leaf_cluster", "quat_from_mat", "orient_column", "any_perp"];
    if PREFIXES.iter().any(|p| name.starts_with(p)) {
        return true;
    }
    // generated build functions: p3_..., prop3_build
    name.strip_prefix("prop").and_then(|r| r.split('_').next()).is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
}

/// Functions of the generated world generator that should stay out-of-line (compile time).
pub fn worldgen_noinline(name: &str) -> bool {
    if worldgen_cold(name) {
        return true;
    }
    // Tier-1 ("quick") builds keep every sizeable function out of line: ~3x faster driver
    // compile, ~5x slower generation. The tier-2 build inlines everything and replaces it.
    const PREFIXES: &[&str] = &[
        "simplex", "fbm", "ridged", "billow", "worley", "value3", "cave_", "voronoi2", "strata_pick", "ground_cover",
        "wg_", "biome_density", "zone_", "shape_", "prop_shape", "fn_",
    ];
    if PREFIXES.iter().any(|p| name.starts_with(p)) {
        return true;
    }
    // per-biome / per-zone generated functions: b3_den(...), z1_carve(...)
    let mut ch = name.chars();
    matches!(ch.next(), Some('b') | Some('z')) && ch.next().map(|c| c.is_ascii_digit()).unwrap_or(false)
}

fn defines_ref(d: &[(String, String)]) -> Vec<(&str, &str)> {
    d.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect()
}

impl PipelineManager {
    pub fn new(root: PathBuf, layout: vk::PipelineLayout) -> Result<Self> {
        let changed = Arc::new(Mutex::new(HashSet::new()));
        let changed_w = changed.clone();
        let watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(ev) = res
                && matches!(ev.kind, notify::EventKind::Modify(_) | notify::EventKind::Create(_)) {
                    let mut c = changed_w.lock().unwrap();
                    for p in ev.paths {
                        c.insert(p.canonicalize().unwrap_or(p));
                    }
                }
        })
        .ok()
        .and_then(|mut w| w.watch(&root, RecursiveMode::Recursive).ok().map(|_| w));
        if watcher.is_none() {
            log::warn!("shader hot reload disabled (could not watch {})", root.display());
        }
        Ok(Self {
            compiler: ShaderCompiler::new(root)?,
            layout,
            entries: Vec::new(),
            changed,
            _watcher: watcher,
            retired: Vec::new(),
            last_errors: Vec::new(),
            reload_count: 0,
        })
    }

    fn build(&self, ctx: &VkContext, name: &str, source: &PipelineSource) -> Result<(vk::Pipeline, Vec<PathBuf>)> {
        match source {
            PipelineSource::Compute { path, defines } => {
                let out = self.compiler.compile_file(path, &defines_ref(defines))?;
                Ok((pipeline::compute(ctx, self.layout, &out.spirv, name)?, out.deps))
            }
            PipelineSource::ComputeSource { name: n, source, defines } => {
                let t0 = std::time::Instant::now();
                let mut out = self.compiler.compile_source(n, source, ShaderKind::Compute, &defines_ref(defines))?;
                if n.starts_with("worldgen_") && std::env::var("VE_INLINE_ALL").is_err() {
                    vkcore::shader::spirv_dont_inline(&mut out.spirv, worldgen_noinline);
                }
                let t1 = t0.elapsed().as_secs_f32() * 1000.0;
                let p = pipeline::compute(ctx, self.layout, &out.spirv, name)?;
                log::debug!("{name}: glsl->spirv {t1:.0} ms ({} words), pipeline {:.0} ms", out.spirv.len(), t0.elapsed().as_secs_f32() * 1000.0 - t1);
                Ok((p, out.deps))
            }
            PipelineSource::Graphics { stages, state } => {
                let mut compiled = Vec::new();
                let mut deps = Vec::new();
                for (path, defines) in stages {
                    let kind = ShaderKind::from_path(path).unwrap();
                    let out = self.compiler.compile_file(path, &defines_ref(defines))?;
                    deps.extend(out.deps);
                    compiled.push((kind.stage(), out.spirv));
                }
                let desc = GraphicsDesc {
                    stages: compiled.iter().map(|(s, c)| (*s, c.as_slice())).collect(),
                    color_formats: state.color_formats.clone(),
                    blend: state.blend.clone(),
                    depth_format: state.depth_format,
                    depth_test: state.depth_test,
                    depth_write: state.depth_write,
                    depth_compare: state.depth_compare,
                    cull: state.cull,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    polygon_mode: state.polygon_mode,
                    depth_bias: state.depth_bias,
                    depth_clamp: state.depth_clamp,
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                };
                Ok((pipeline::graphics(ctx, self.layout, &desc, name)?, deps))
            }
        }
    }

    pub fn create(&mut self, ctx: &VkContext, name: &str, source: PipelineSource) -> Result<PipelineHandle> {
        let (pipeline, deps) = self.build(ctx, name, &source)?;
        let deps = deps.into_iter().map(|p| p.canonicalize().unwrap_or(p)).collect();
        self.entries.push(Entry { name: name.to_string(), source, pipeline, deps });
        Ok(PipelineHandle(self.entries.len() - 1))
    }

    /// Create several in-memory compute pipelines at once: GLSL -> SPIR-V one after another (fast),
    /// then the driver compiles (slow) run in parallel. Used for the generated world generator.
    pub fn create_sources_parallel(&mut self, ctx: &VkContext, items: Vec<(&str, PipelineSource)>) -> Result<Vec<PipelineHandle>> {
        let mut spirvs = Vec::new();
        for (name, source) in &items {
            let PipelineSource::ComputeSource { name: n, source: src, defines } = source else {
                anyhow::bail!("{name}: create_sources_parallel takes ComputeSource pipelines");
            };
            let mut out = self.compiler.compile_source(n, src, ShaderKind::Compute, &defines_ref(defines))?;
            if n.starts_with("worldgen_") && std::env::var("VE_INLINE_ALL").is_err() {
                vkcore::shader::spirv_dont_inline(&mut out.spirv, worldgen_noinline);
            }
            spirvs.push(out);
        }
        let layout = self.layout;
        let built: Vec<Result<vk::Pipeline>> = std::thread::scope(|sc| {
            let handles: Vec<_> = items.iter().zip(&spirvs).map(|((name, _), out)| sc.spawn(move || pipeline::compute(ctx, layout, &out.spirv, name))).collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("pipeline build thread panicked")))).collect()
        });
        if built.iter().any(|b| b.is_err()) {
            let mut err = None;
            for b in built {
                match b {
                    Ok(p) => unsafe { ctx.device.destroy_pipeline(p, None) },
                    Err(e) => err = err.or(Some(e)),
                }
            }
            return Err(err.unwrap());
        }
        let mut handles = Vec::new();
        for (((name, source), out), pipeline) in items.into_iter().zip(spirvs).zip(built) {
            let deps = out.deps.into_iter().map(|p| p.canonicalize().unwrap_or(p)).collect();
            self.entries.push(Entry { name: name.to_string(), source, pipeline: pipeline?, deps });
            handles.push(PipelineHandle(self.entries.len() - 1));
        }
        Ok(handles)
    }

    /// Replace the source of an existing pipeline (e.g. new worldgen code) with an already-built
    /// pipeline object.
    pub fn replace(&mut self, h: PipelineHandle, source: PipelineSource, pipeline: vk::Pipeline, deps: Vec<PathBuf>, retire_after: u64) {
        let e = &mut self.entries[h.0];
        self.retired.push((retire_after, e.pipeline));
        e.pipeline = pipeline;
        e.source = source;
        e.deps = deps.into_iter().map(|p| p.canonicalize().unwrap_or(p)).collect();
    }

    pub fn get(&self, h: PipelineHandle) -> vk::Pipeline {
        self.entries[h.0].pipeline
    }

    /// Rebuild every pipeline that depends on a changed file. Keeps the old pipeline on error.
    pub fn hot_reload(&mut self, ctx: &VkContext, frame_value: u64, force_all: bool) -> usize {
        let changed: HashSet<PathBuf> = std::mem::take(&mut *self.changed.lock().unwrap());
        if changed.is_empty() && !force_all {
            return 0;
        }
        let mut n = 0;
        let mut errors = Vec::new();
        for i in 0..self.entries.len() {
            let hit = force_all || self.entries[i].deps.iter().any(|d| changed.contains(d));
            if !hit {
                continue;
            }
            let name = self.entries[i].name.clone();
            let source = self.entries[i].source.clone();
            match self.build(ctx, &name, &source) {
                Ok((p, deps)) => {
                    let e = &mut self.entries[i];
                    self.retired.push((frame_value, e.pipeline));
                    e.pipeline = p;
                    e.deps = deps.into_iter().map(|p| p.canonicalize().unwrap_or(p)).collect();
                    n += 1;
                    log::info!("reloaded pipeline {name}");
                }
                Err(err) => {
                    log::error!("{err:#}");
                    errors.push(format!("{name}: {err:#}"));
                }
            }
        }
        if n > 0 || !errors.is_empty() {
            self.last_errors = errors;
            self.reload_count += n as u32;
        }
        n
    }

    pub fn collect_garbage(&mut self, ctx: &VkContext, completed: u64) {
        self.retired.retain(|(v, p)| {
            if *v <= completed {
                unsafe { ctx.device.destroy_pipeline(*p, None) };
                false
            } else {
                true
            }
        });
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        unsafe {
            for e in self.entries.drain(..) {
                ctx.device.destroy_pipeline(e.pipeline, None);
            }
            for (_, p) in self.retired.drain(..) {
                ctx.device.destroy_pipeline(p, None);
            }
            ctx.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

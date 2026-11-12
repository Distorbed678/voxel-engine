use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::{Context as _, Result, anyhow};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderKind {
    Compute,
    Task,
    Mesh,
    Vertex,
    Fragment,
}

impl ShaderKind {
    fn to_shaderc(self) -> shaderc::ShaderKind {
        match self {
            ShaderKind::Compute => shaderc::ShaderKind::Compute,
            ShaderKind::Task => shaderc::ShaderKind::Task,
            ShaderKind::Mesh => shaderc::ShaderKind::Mesh,
            ShaderKind::Vertex => shaderc::ShaderKind::Vertex,
            ShaderKind::Fragment => shaderc::ShaderKind::Fragment,
        }
    }

    pub fn stage(self) -> ash::vk::ShaderStageFlags {
        use ash::vk::ShaderStageFlags as S;
        match self {
            ShaderKind::Compute => S::COMPUTE,
            ShaderKind::Task => S::TASK_EXT,
            ShaderKind::Mesh => S::MESH_EXT,
            ShaderKind::Vertex => S::VERTEX,
            ShaderKind::Fragment => S::FRAGMENT,
        }
    }

    pub fn from_path(path: &str) -> Option<Self> {
        let ext = Path::new(path).extension()?.to_str()?;
        Some(match ext {
            "comp" => ShaderKind::Compute,
            "task" => ShaderKind::Task,
            "mesh" => ShaderKind::Mesh,
            "vert" => ShaderKind::Vertex,
            "frag" => ShaderKind::Fragment,
            _ => return None,
        })
    }
}

pub struct ShaderOutput {
    pub spirv: Vec<u32>,
    /// Every file that contributed to this shader (for hot reload).
    pub deps: Vec<PathBuf>,
}

/// Runtime GLSL -> SPIR-V compiler (system libshaderc). Not `Sync`; create one per thread.
pub struct ShaderCompiler {
    compiler: shaderc::Compiler,
    pub root: PathBuf,
    pub debug_info: bool,
}

impl ShaderCompiler {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            compiler: shaderc::Compiler::new().map_err(|e| anyhow!("shaderc init: {e}"))?,
            root: root.into(),
            debug_info: false,
        })
    }

    pub fn compile_file(
        &self,
        rel_path: &str,
        defines: &[(&str, &str)],
    ) -> Result<ShaderOutput> {
        let kind = ShaderKind::from_path(rel_path)
            .ok_or_else(|| anyhow!("unknown shader stage for {rel_path}"))?;
        let path = self.root.join(rel_path);
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("reading shader {}", path.display()))?;
        let mut out = self.compile_source(rel_path, &source, kind, defines)?;
        out.deps.push(path);
        Ok(out)
    }

    pub fn compile_source(
        &self,
        name: &str,
        source: &str,
        kind: ShaderKind,
        defines: &[(&str, &str)],
    ) -> Result<ShaderOutput> {
        let deps = Rc::new(RefCell::new(Vec::<PathBuf>::new()));
        let mut options =
            shaderc::CompileOptions::new().map_err(|e| anyhow!("shaderc options: {e}"))?;
        options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);
        options.set_target_spirv(shaderc::SpirvVersion::V1_6);
        options.set_source_language(shaderc::SourceLanguage::GLSL);
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        if self.debug_info {
            options.set_generate_debug_info();
        }
        for (k, v) in defines {
            options.add_macro_definition(k, Some(v));
        }
        let root = self.root.clone();
        let deps_cb = deps.clone();
        options.set_include_callback(move |requested, ty, requester, _depth| {
            let candidates = match ty {
                shaderc::IncludeType::Relative => {
                    let base = root.join(requester);
                    let dir = base.parent().map(Path::to_path_buf).unwrap_or(root.clone());
                    vec![dir.join(requested), root.join(requested)]
                }
                shaderc::IncludeType::Standard => vec![root.join(requested)],
            };
            for c in candidates {
                if let Ok(content) = std::fs::read_to_string(&c) {
                    deps_cb.borrow_mut().push(c.clone());
                    let resolved = c
                        .strip_prefix(&root)
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| c.to_string_lossy().into_owned());
                    return Ok(shaderc::ResolvedInclude { resolved_name: resolved, content });
                }
            }
            Err(format!("include not found: {requested} (from {requester})"))
        });
        let artifact = self
            .compiler
            .compile_into_spirv(source, kind.to_shaderc(), name, "main", Some(&options))
            .map_err(|e| anyhow!("shader compile failed ({name}):\n{e}"))?;
        if artifact.get_num_warnings() > 0 {
            log::debug!("shader warnings ({name}): {}", artifact.get_warning_messages());
        }
        let spirv = artifact.as_binary().to_vec();
        drop(options);
        let deps = Rc::try_unwrap(deps).map(|c| c.into_inner()).unwrap_or_default();
        Ok(ShaderOutput { spirv, deps })
    }
}

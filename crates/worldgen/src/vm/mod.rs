//! The world generator's interpreter (roadmap item 10).
//!
//! A layer's generated GLSL is compiled once more, here, into bytecode for a small register VM
//! (`shaders/worldgen/vm_core.glsl`). Interpreter pipelines (terrain, props, probe) run it while
//! the layer's compiled pipelines build after a structural edit, so edits apply at once. Both run
//! the same source: the VM interprets the generated code, and every helper it calls (noise, caves,
//! prop builders, ...) is the real GLSL function, called as a native. The interpreter shaders
//! don't depend on the graph: they are built once.
//!
//! Registers are 32-bit words (`uint R[]`); floats, ints, uints and bools are stored as bits, a
//! vecN takes N words, a mat3 nine, arrays and structs their elements in order. Each function
//! call gets a frame: [return value][parameters][locals and temporaries]. `out`/`inout`
//! parameters and array or struct parameters are passed by reference (a word holding an
//! absolute register address). The column (`WgColumn`) of the API calls lives at address 0, the
//! program's globals after it, frames after those.

pub mod codegen;
pub mod glsl;
pub mod lex;
pub mod parse;

use std::collections::HashMap;

use anyhow::{Result, bail};

pub use codegen::compile_program;

/// Operand flag: an absolute register address (else relative to the frame).
pub const ABS: u32 = 1 << 31;
/// Instruction size in words: [op | w << 8, a, b, c, d].
pub const INSN: usize = 5;
/// Words of VM registers per thread (`VM_REGS` in the shader).
pub const VM_REGS: u32 = 448;

macro_rules! ops {
    ($($name:ident),* $(,)?) => {
        /// VM opcodes (`VM_<NAME>` in the shader, same values).
        #[allow(clippy::upper_case_acronyms, non_camel_case_types)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u32)]
        pub enum Op { $($name),* }
        impl Op {
            pub const ALL: &'static [Op] = &[$(Op::$name),*];
            pub fn name(self) -> &'static str {
                match self { $(Op::$name => stringify!($name)),* }
            }
        }
    };
}

ops! {
    NOP,
    // data: MOV a <- b (w words); LDK a <- inline words b, c, d (w <= 3)
    MOV, LDK,
    // LDI a <- [R[b] + c] (d words); STI [R[a] + c] <- b (d words)
    LDI, STI,
    // ADDR a <- absolute address of operand b; LEA a <- R[b] + R[c] * d
    ADDR, LEA,
    // SWZ a <- b swizzled (w comps, selector in c: 2 bits each); SPLAT a <- b (w comps)
    SWZ, SPLAT,
    // float, component-wise over w: a = b op c
    FADD, FSUB, FMUL, FDIV, FMOD, FMIN, FMAX, FPOW, FATAN2, FSTEP,
    // float unary: a = op(b)
    FNEG, FABS, FSIGN, FFLOOR, FCEIL, FFRACT, FSQRT, FINVSQRT, FSIN, FCOS, FTAN, FASIN, FACOS, FATAN, FEXP, FLOG, FEXP2, FLOG2, FROUND, FTRUNC,
    // float ternary: a = op(b, c, d)
    FMIX, FCLAMP, FSMOOTH,
    // vectors of width w: a = op(b[, c])
    FDOT, FLEN, FDIST, FNORM, FCROSS,
    // int / uint, component-wise over w (XADD/XSUB/XMUL wrap: the same for int and uint)
    XADD, XSUB, XMUL, IDIV, UDIV, IMOD, UMOD, AND, OR, XOR, SHL, ISHR, USHR, IMIN, IMAX, UMIN, UMAX, ICLAMP, UCLAMP,
    INEG, INOT, IABS, ISIGN,
    // conversions over w components
    F2I, F2U, I2F, U2F, F2B, X2B,
    // comparisons over w components (== and != over all of them), result: bool
    FLT, FLE, FEQ, FNE, ILT, ILE, ULT, ULE, XEQ, XNE,
    LNOT,
    // control: JMP b; JZ/JNZ a, b; CALL b (code address), c (callee frame offset); RET
    JMP, JZ, JNZ, CALL, RET,
    // NAT: native c... a = result, b = native id, c = first argument (frame-relative)
    NAT,
}

/// Scalar kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sc {
    F,
    I,
    U,
    B,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Void,
    /// a scalar (n = 1) or a vector (n = 2..4)
    V(Sc, u8),
    Mat3,
    Array(Box<Ty>, u32),
    Struct(usize),
}

impl Ty {
    pub const FLOAT: Ty = Ty::V(Sc::F, 1);
    pub const INT: Ty = Ty::V(Sc::I, 1);
    pub const UINT: Ty = Ty::V(Sc::U, 1);
    pub const BOOL: Ty = Ty::V(Sc::B, 1);
    pub fn is_scalar(&self) -> bool {
        matches!(self, Ty::V(_, 1))
    }
    pub fn sc(&self) -> Option<Sc> {
        match self {
            Ty::V(s, _) => Some(*s),
            _ => None,
        }
    }
    pub fn n(&self) -> u32 {
        match self {
            Ty::V(_, n) => *n as u32,
            _ => 0,
        }
    }
    /// passed by reference when a parameter
    pub fn aggregate(&self) -> bool {
        matches!(self, Ty::Array(..) | Ty::Struct(_))
    }
}

/// A struct: fields with their types and word offsets.
#[derive(Clone, Debug)]
pub struct StructInfo {
    pub name: String,
    pub fields: Vec<(String, Ty, u32)>,
    pub size: u32,
    /// a library struct of the props pass only
    pub props_only: bool,
}

/// A native: a GLSL function of the shader library (or a library global's getter / setter, or a
/// push constant field) the bytecode calls by id.
#[derive(Clone, Debug)]
pub struct Native {
    pub name: String,
    pub ret: Ty,
    /// (is out/inout, type)
    pub params: Vec<(parse::Qual, Ty)>,
    pub kind: NativeKind,
    /// only in the props pass (`#ifdef PROPS_PASS`)
    pub props_only: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NativeKind {
    Func,
    /// read / write a library global
    Get(String),
    Set(String),
    /// read a push constant field
    Push(String),
}

/// What the interpreter knows of the shader library: structs, natives and `#define`s.
#[derive(Clone, Debug, Default)]
pub struct Library {
    pub structs: Vec<StructInfo>,
    pub natives: Vec<Native>,
    pub defines: HashMap<String, String>,
    /// library globals (name -> type)
    pub globals: HashMap<String, Ty>,
}

/// Library files the interpreter calls into, in include order, and whether each is only in the
/// props pass.
pub const LIBRARY_FILES: &[(&str, bool)] = &[
    ("common.glsl", false),
    ("noise.glsl", false),
    ("voxel.glsl", false),
    ("worldgen/prelude.glsl", false),
    ("worldgen/props_lib.glsl", true),
];

/// Push constant fields the generated code reads (`pc.<field>`).
pub const PUSH_FIELDS: &[(&str, Sc)] = &[("soloZone", Sc::F), ("soloBiome", Sc::F)];

impl Library {
    /// Scan the library sources (`LIBRARY_FILES`, read by the caller): structs, globals, function
    /// signatures and `#define`s.
    pub fn new(files: &[(&str, &str)]) -> Result<Self> {
        let mut lib = Library::default();
        let mut defines: HashMap<String, String> = HashMap::new();
        defines.insert("PROPS_PASS".into(), String::new());
        let mut struct_names: Vec<String> = Vec::new();
        let mut struct_defs: Vec<(parse::StructDef, bool)> = Vec::new();
        let mut sigs: Vec<(String, bool)> = Vec::new();
        let mut globals: Vec<(String, bool)> = Vec::new();
        for (path, text) in files {
            let props_only = LIBRARY_FILES.iter().find(|(p, _)| p == path).is_some_and(|(_, p)| *p);
            let src = lex::preprocess(text, &mut defines);
            // structs: `struct Name { ... };` blocks
            let mut rest = src.as_str();
            while let Some(i) = rest.find("struct ") {
                let at = &rest[i..];
                let line_start = rest[..i].rfind('\n').map_or(0, |k| k + 1);
                if !rest[line_start..i].trim().is_empty() {
                    rest = &at[7..];
                    continue;
                }
                let Some(end) = at.find("};") else { break };
                let block = &at[..end + 2];
                let toks = lex::tokenize(block, &HashMap::new())?;
                let mut p = parse::Parser::new(&toks, struct_names.clone());
                if let Ok(items) = p.items() {
                    for it in items {
                        if let parse::Item::Struct(s) = it {
                            struct_names.push(s.name.clone());
                            struct_defs.push((s, props_only));
                        }
                    }
                }
                rest = &at[end + 2..];
            }
            for line in src.lines() {
                if line.starts_with(char::is_whitespace) || line.is_empty() {
                    continue;
                }
                let t = line.trim_end();
                // function definitions: `type name(params) {` (the body may follow on the line)
                if let Some(open) = t.find('(')
                    && !t.starts_with("layout")
                    && let Some(close) = t[open..].find(')').map(|k| open + k)
                    && t[close + 1..].trim_start().starts_with('{')
                {
                    let head: Vec<&str> = t[..open].split_whitespace().collect();
                    if head.len() == 2 && !head[0].contains('=') {
                        sigs.push((format!("{} {}({});", head[0], head[1], &t[open + 1..close]), props_only));
                    }
                    continue;
                }
                // globals: `type gName;` / `type gA; type gB;` (no initialiser)
                for decl in t.split(';').map(str::trim).filter(|d| !d.is_empty()) {
                    let w: Vec<&str> = decl.split_whitespace().collect();
                    if w.len() >= 2 && w[1].starts_with('g') && w[1].chars().nth(1).is_some_and(|c| c.is_ascii_uppercase()) && !decl.contains('(') && !decl.contains('=') {
                        for name in decl[w[0].len()..].split(',') {
                            globals.push((format!("{} {};", w[0], name.trim()), props_only));
                        }
                    }
                }
            }
        }
        // structs, in order (fields may use earlier structs)
        let mut ids: HashMap<String, usize> = HashMap::new();
        for (s, props_only) in &struct_defs {
            let mut fields = Vec::new();
            let mut off = 0;
            let mut ok = true;
            for (ty, name, dims) in &s.fields {
                let Some(t) = codegen::type_of_name(ty, &ids) else {
                    ok = false;
                    break;
                };
                let Some(t) = codegen::array_of(t, dims, &defines) else {
                    ok = false;
                    break;
                };
                let size = codegen::size_of(&t, &lib.structs);
                fields.push((name.clone(), t, off));
                off += size;
            }
            if ok {
                ids.insert(s.name.clone(), lib.structs.len());
                lib.structs.push(StructInfo { name: s.name.clone(), fields, size: off, props_only: *props_only });
            }
        }
        // natives: functions whose parameter and return types the VM can hold
        for (sig, props_only) in &sigs {
            let toks = lex::tokenize(sig, &HashMap::new())?;
            let mut p = parse::Parser::new(&toks, ids.keys().cloned());
            let Ok(items) = p.items() else { continue };
            for it in items {
                let parse::Item::Func(f) = it else { continue };
                let Some(ret) = codegen::type_of_name(&f.ret, &ids) else { continue };
                let mut params = Vec::new();
                for prm in &f.params {
                    match codegen::type_of_name(&prm.ty, &ids).and_then(|t| codegen::array_of(t, &prm.dims, &defines)) {
                        Some(t) if !matches!(t, Ty::Array(..)) => params.push((prm.qual, t)),
                        _ => {
                            params.clear();
                            params.push((parse::Qual::In, Ty::Void));
                            break;
                        }
                    }
                }
                if params.iter().any(|(_, t)| *t == Ty::Void) {
                    continue;
                }
                lib.natives.push(Native { name: f.name.clone(), ret, params, kind: NativeKind::Func, props_only: *props_only });
            }
        }
        // library globals: a getter and a setter each
        for (decl, props_only) in &globals {
            let w: Vec<&str> = decl.trim_end_matches(';').split_whitespace().collect();
            let Some(ty) = codegen::type_of_name(w[0], &ids) else { continue };
            let name = w[1].to_string();
            lib.globals.insert(name.clone(), ty.clone());
            lib.natives.push(Native { name: format!("__get_{name}"), ret: ty.clone(), params: vec![], kind: NativeKind::Get(name.clone()), props_only: *props_only });
            lib.natives.push(Native { name: format!("__set_{name}"), ret: Ty::Void, params: vec![(parse::Qual::In, ty)], kind: NativeKind::Set(name), props_only: *props_only });
        }
        for (f, sc) in PUSH_FIELDS {
            lib.natives.push(Native { name: format!("__pc_{f}"), ret: Ty::V(*sc, 1), params: vec![], kind: NativeKind::Push(f.to_string()), props_only: false });
        }
        defines.remove("PROPS_PASS");
        lib.defines = defines;
        if lib.natives.is_empty() {
            bail!("the shader library has no functions the interpreter can call");
        }
        Ok(lib)
    }

    pub fn struct_id(&self, name: &str) -> Option<usize> {
        self.structs.iter().position(|s| s.name == name)
    }
}

/// The functions the main templates call (`terrain_main.glsl`, ...): each program has an entry
/// for every one, in this order (the program header), and the interpreter's glue implements
/// them by running the VM. Signatures as the generator emits them.
pub const API: &[&str] = &[
    "void wg_init()",
    "void wg_column(vec2 xz, out WgColumn c)",
    "uint wg_main_biome(in WgColumn c)",
    "float wg_density_c(vec3 p, in WgColumn c, out bool carved)",
    "float wg_density(vec3 p, in WgColumn c)",
    "float wg_depth_max(in WgColumn c)",
    "uint wg_material(vec3 p, float depth, float airAbove, bool skyOpen, in WgColumn c)",
    "uint wg_fluid(vec3 p, bool skyOpen, in WgColumn c)",
    "uint wg_decorate(vec3 p, uint ground, in WgColumn c)",
    "uint wg_cave_biome(vec3 p, in WgColumn c)",
    "uint wg_cave_material(uint b, vec3 p, float depth, float airAbove)",
    "uint wg_cave_decorate(uint b, vec3 p, uint ground)",
    "uint wg_col_biome(in WgColumn c)",
    "float wg_col_weight(in WgColumn c)",
    "uint wg_col_zone(in WgColumn c, uint k)",
    "void prop_layer(uint i, inout PropLayer L)",
    "float prop_mask(uint i, vec3 p, in WgColumn c)",
    "void prop_build(uint g)",
];

/// `#define`s of the generated code the main templates read; the program header carries them.
pub const HEADER_DEFINES: &[&str] = &["NUM_PROP_LAYERS", "BIOME_BASE", "ZONE_BASE", "NUM_BIOMES", "NUM_ZONES"];

/// Program header (words): magic, frame base (first word after the globals), the address of
/// the generated global `gSeaLevel` (read by the main templates), the init routine (global
/// initialisers), then one entry address per `API` function (0xFFFFFFFF = missing), then the
/// `HEADER_DEFINES` values.
pub const MAGIC: u32 = 0x564D_0001;
pub const H_FRAME: usize = 1;
pub const H_SEA: usize = 2;
pub const H_INIT: usize = 3;
pub const H_ENTRIES: usize = 4;
pub fn header_len() -> usize {
    H_ENTRIES + API.len() + HEADER_DEFINES.len()
}

/// A compiled layer.
#[derive(Clone, Debug, Default)]
pub struct Program {
    /// header and code
    pub words: Vec<u32>,
    /// the deepest the frames go (words above the frame base)
    pub stack: u32,
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader library, read from the repository.
    pub(crate) fn library() -> Library {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders");
        let texts: Vec<(String, String)> = LIBRARY_FILES.iter().map(|(p, _)| (p.to_string(), std::fs::read_to_string(root.join(p)).unwrap())).collect();
        let refs: Vec<(&str, &str)> = texts.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        Library::new(&refs).unwrap()
    }

    #[test]
    fn library_has_the_natives_the_generator_calls() {
        let lib = library();
        for n in ["fbm3", "P", "cave_spaghetti", "voronoi2", "ground_cover", "tree_trunk", "xf_mul", "__get_gSeed", "__set_gXf", "__pc_soloZone"] {
            assert!(lib.natives.iter().any(|x| x.name == n), "no native {n}");
        }
        assert!(lib.struct_id("Xf").is_some() && lib.struct_id("PropLayer").is_some());
        assert!(lib.defines.contains_key("NO_CAVE_BIOME"));
    }

    #[test]
    fn default_layers_compile_to_bytecode() {
        let lib = library();
        let p = crate::defaults::default_project();
        let blocks: HashMap<String, u32> = ["air", "stone", "dirt", "grass", "snow", "gravel", "water", "lava", "ice", "sand", "sandstone"].iter().enumerate().map(|(i, n)| (n.to_string(), i as u32)).collect();
        let header: String = blocks.iter().map(|(n, i)| format!("#define B_{} {i}u\n", n.to_uppercase())).collect();
        let c = crate::compile::compile(&p, &blocks, &HashMap::new()).unwrap();
        for l in &c.layers {
            let prog = compile_program(&lib, &format!("{header}\n{}", l.source)).unwrap_or_else(|e| panic!("layer {}: {e:#}", l.name));
            assert_eq!(prog.words[0], MAGIC);
            assert!(prog.warnings.is_empty(), "layer {}: {:?}", l.name, prog.warnings);
            for (i, api) in API.iter().enumerate() {
                assert!(prog.entry(i).is_some(), "layer {}: no entry for {api}", l.name);
            }
            eprintln!("layer {}: {} words, frame base {}, stack {}", l.name, prog.words.len(), prog.words[H_FRAME], prog.stack);
        }
        let src = glsl::interpreter_source(&lib).unwrap();
        if let Ok(d) = std::env::var("VE_DUMP_LAYERS") {
            std::fs::write(format!("{d}/interpreter.glsl"), &src).unwrap();
        }
        assert!(src.contains("void vm_native(") && src.contains("float wg_density_c(vec3 p, in WgColumn c, out bool carved)"));
    }
}

//! Type checking and code generation: the generated GLSL of a layer -> VM bytecode.

use std::collections::HashMap;

use anyhow::{Context as _, Result, anyhow, bail};

use super::parse::{Expr, Func, Item, Param, Parser, Qual, Stmt};
use super::{ABS, API, HEADER_DEFINES, Library, Native, Op, Program, Sc, StructInfo, Ty};

pub fn type_of_name(name: &str, ids: &HashMap<String, usize>) -> Option<Ty> {
    let v = |s: Sc, n: u8| Some(Ty::V(s, n));
    match name {
        "void" => Some(Ty::Void),
        "float" => v(Sc::F, 1),
        "int" => v(Sc::I, 1),
        "uint" => v(Sc::U, 1),
        "bool" => v(Sc::B, 1),
        "vec2" => v(Sc::F, 2),
        "vec3" => v(Sc::F, 3),
        "vec4" => v(Sc::F, 4),
        "ivec2" => v(Sc::I, 2),
        "ivec3" => v(Sc::I, 3),
        "ivec4" => v(Sc::I, 4),
        "uvec2" => v(Sc::U, 2),
        "uvec3" => v(Sc::U, 3),
        "uvec4" => v(Sc::U, 4),
        "bvec2" => v(Sc::B, 2),
        "bvec3" => v(Sc::B, 3),
        "bvec4" => v(Sc::B, 4),
        "mat3" => Some(Ty::Mat3),
        s => ids.get(s).map(|&i| Ty::Struct(i)),
    }
}

/// A constant integer expression (array sizes).
pub fn const_int(e: &Expr) -> Option<i64> {
    match e {
        Expr::Int(i) => Some(*i),
        Expr::Uint(u) => Some(*u as i64),
        Expr::Bin(op, a, b) => {
            let (a, b) = (const_int(a)?, const_int(b)?);
            Some(match *op {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" if b != 0 => a / b,
                _ => return None,
            })
        }
        Expr::Unary("-", a) => Some(-const_int(a)?),
        _ => None,
    }
}

/// `ty name[d0][d1]` -> array of d0 arrays of d1. Unsized dimensions (`[]`) are size 0.
pub fn array_of(ty: Ty, dims: &[Expr], _defines: &HashMap<String, String>) -> Option<Ty> {
    let mut t = ty;
    for d in dims.iter().rev() {
        let n = const_int(d)?;
        t = Ty::Array(Box::new(t), n.max(0) as u32);
    }
    Some(t)
}

pub fn size_of(ty: &Ty, structs: &[StructInfo]) -> u32 {
    match ty {
        Ty::Void => 0,
        Ty::V(_, n) => *n as u32,
        Ty::Mat3 => 9,
        Ty::Array(t, n) => size_of(t, structs) * n,
        Ty::Struct(i) => structs[*i].size,
    }
}

/// A value: an operand (frame-relative, or absolute with `ABS`) holding it.
#[derive(Clone, Debug)]
struct Val {
    op: u32,
    ty: Ty,
}

/// Where a variable (or part of one) lives.
#[derive(Clone, Debug)]
enum Place {
    /// registers addressed directly: a frame-relative or absolute operand
    Direct { op: u32, ty: Ty },
    /// through a pointer: frame register `reg` holds an absolute address, plus `off` words
    Ptr { reg: u32, off: u32, ty: Ty },
    /// components of a vector (a non-contiguous swizzle)
    Comps { base: Box<Place>, comps: Vec<u32>, ty: Ty },
    /// a global of the shader library: read and written through natives
    Lib { name: String, ty: Ty },
}

impl Place {
    fn ty(&self) -> &Ty {
        match self {
            Place::Direct { ty, .. } | Place::Ptr { ty, .. } | Place::Comps { ty, .. } | Place::Lib { ty, .. } => ty,
        }
    }
    /// The same place `off` words further, typed `ty`.
    fn sub(&self, off: u32, ty: Ty) -> Option<Place> {
        match self {
            Place::Direct { op, .. } => Some(Place::Direct { op: op + off, ty }),
            Place::Ptr { reg, off: o, .. } => Some(Place::Ptr { reg: *reg, off: o + off, ty }),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct FuncSig {
    ret: Ty,
    /// (qualifier, type, passed by reference)
    params: Vec<(Qual, Ty, bool)>,
}

struct FuncState {
    scopes: Vec<HashMap<String, Place>>,
    top: u32,
    max: u32,
    ret: Ty,
    breaks: Vec<Vec<usize>>,
    conts: Vec<Vec<usize>>,
    /// (call site's callee frame offset, callee)
    calls: Vec<(u32, String)>,
    name: String,
}

struct Gen<'a> {
    lib: &'a Library,
    structs: Vec<StructInfo>,
    struct_ids: HashMap<String, usize>,
    sigs: HashMap<String, FuncSig>,
    globals: HashMap<String, (u32, Ty)>,
    global_top: u32,
    code: Vec<u32>,
    /// call instructions to patch: (instruction index, callee)
    call_patches: Vec<(usize, String)>,
    addrs: HashMap<String, u32>,
    /// per function: frame size and calls (for the stack depth)
    frames: HashMap<String, (u32, Vec<(u32, String)>)>,
    f: Option<FuncState>,
    natives: HashMap<String, Vec<usize>>,
}

/// Compile a layer's generated source (with the blocks header). `PROPS_PASS` is defined: one
/// program serves every interpreter pass.
pub fn compile_program(lib: &Library, source: &str) -> Result<Program> {
    let mut defines = lib.defines.clone();
    defines.insert("PROPS_PASS".into(), String::new());
    let src = super::lex::preprocess(source, &mut defines);
    defines.remove("PROPS_PASS");
    let toks = super::lex::tokenize(&src, &defines)?;
    let mut parser = Parser::new(&toks, lib.structs.iter().map(|s| s.name.clone()));
    let items = parser.items()?;
    let mut g = Gen {
        lib,
        structs: lib.structs.clone(),
        struct_ids: lib.structs.iter().enumerate().map(|(i, s)| (s.name.clone(), i)).collect(),
        sigs: HashMap::new(),
        globals: HashMap::new(),
        global_top: 0,
        code: Vec::new(),
        call_patches: Vec::new(),
        addrs: HashMap::new(),
        frames: HashMap::new(),
        f: None,
        natives: HashMap::new(),
    };
    for (i, n) in lib.natives.iter().enumerate() {
        g.natives.entry(n.name.clone()).or_default().push(i);
    }
    g.program(&items, &defines)
}

impl<'a> Gen<'a> {
    fn ty(&self, name: &str) -> Result<Ty> {
        type_of_name(name, &self.struct_ids).ok_or_else(|| anyhow!("unknown type '{name}'"))
    }
    fn size(&self, t: &Ty) -> u32 {
        size_of(t, &self.structs)
    }

    // ---- code emission ----
    fn emit(&mut self, op: Op, w: u32, a: u32, b: u32, c: u32, d: u32) -> usize {
        let at = self.code.len();
        self.code.extend_from_slice(&[op as u32 | (w << 8), a, b, c, d]);
        at
    }
    fn here(&self) -> u32 {
        self.code.len() as u32
    }
    fn patch(&mut self, insn: usize, target: u32) {
        self.code[insn + 2] = target;
    }

    fn fs(&mut self) -> &mut FuncState {
        self.f.as_mut().expect("in a function")
    }
    fn tmp(&mut self, words: u32) -> u32 {
        let f = self.fs();
        let at = f.top;
        f.top += words.max(1);
        f.max = f.max.max(f.top);
        at
    }
    fn mark(&self) -> u32 {
        self.f.as_ref().unwrap().top
    }
    fn reset(&mut self, m: u32) {
        self.fs().top = m;
    }

    fn program(&mut self, items: &[Item], defines: &HashMap<String, String>) -> Result<Program> {
        // structs of the program
        for it in items {
            if let Item::Struct(s) = it {
                let mut fields = Vec::new();
                let mut off = 0;
                for (ty, name, dims) in &s.fields {
                    let t = array_of(self.ty(ty)?, dims, defines).ok_or_else(|| anyhow!("struct {}: bad array size", s.name))?;
                    let n = self.size(&t);
                    fields.push((name.clone(), t, off));
                    off += n;
                }
                self.struct_ids.insert(s.name.clone(), self.structs.len());
                self.structs.push(StructInfo { name: s.name.clone(), fields, size: off, props_only: false });
            }
        }
        // the column (WgColumn) lives at address 0, globals after it
        let col = self.struct_ids.get("WgColumn").copied().ok_or_else(|| anyhow!("the program has no WgColumn"))?;
        self.global_top = self.structs[col].size;
        let mut inits: Vec<(u32, Ty, Expr)> = Vec::new();
        for it in items {
            if let Item::Global { ty, vars, .. } = it {
                let base = self.ty(ty)?;
                for v in vars {
                    let mut t = array_of(base.clone(), &v.dims, defines).ok_or_else(|| anyhow!("global {}: bad array size", v.name))?;
                    // unsized array: its size from the initialiser
                    if let (Ty::Array(e, 0), Some(Expr::ArrayCtor(_, args))) = (&t, &v.init) {
                        t = Ty::Array(e.clone(), args.len() as u32);
                    }
                    let addr = self.global_top;
                    self.global_top += self.size(&t);
                    self.globals.insert(v.name.clone(), (addr, t.clone()));
                    if let Some(e) = &v.init {
                        inits.push((addr, t, e.clone()));
                    }
                }
            }
        }
        // signatures
        for it in items {
            if let Item::Func(f) = it {
                let sig = self.sig(f, defines)?;
                self.sigs.insert(f.name.clone(), sig);
            }
        }
        // header placeholder
        let hl = super::header_len();
        self.code.resize(hl, 0);
        // global initialisers: a routine of their own
        let init_addr = self.here();
        self.f = Some(FuncState { scopes: vec![HashMap::new()], top: 0, max: 0, ret: Ty::Void, breaks: vec![], conts: vec![], calls: vec![], name: "__init".into() });
        for (addr, t, e) in &inits {
            let m = self.mark();
            let v = self.expr(e)?;
            let v = self.convert(v, t)?;
            let n = self.size(t);
            self.emit(Op::MOV, n, addr | ABS, v.op, 0, 0);
            self.reset(m);
        }
        self.emit(Op::RET, 0, 0, 0, 0, 0);
        let fstate = self.f.take().unwrap();
        self.frames.insert("__init".into(), (fstate.max, fstate.calls));
        // functions
        for it in items {
            if let Item::Func(f) = it
                && let Some(body) = &f.body
            {
                self.function(f, body, defines).with_context(|| format!("function {} (line {})", f.name, f.line))?;
            }
        }
        for (insn, callee) in std::mem::take(&mut self.call_patches) {
            let a = *self.addrs.get(&callee).ok_or_else(|| anyhow!("function {callee} is declared but not defined"))?;
            self.patch(insn, a);
        }
        // header
        let frame0 = self.global_top.next_multiple_of(4);
        self.code[0] = super::MAGIC;
        self.code[super::H_FRAME] = frame0;
        self.code[super::H_SEA] = self.globals.get("gSeaLevel").map_or(u32::MAX, |g| g.0);
        self.code[super::H_INIT] = init_addr;
        let mut warnings = Vec::new();
        for (i, api) in API.iter().enumerate() {
            let name = api.split('(').next().unwrap().split_whitespace().last().unwrap();
            let addr = match (self.addrs.get(name), self.sigs.get(name)) {
                (Some(a), Some(sig)) => {
                    let want = self.api_sig(api)?;
                    if want.ret != sig.ret || want.params.len() != sig.params.len() || want.params.iter().zip(&sig.params).any(|(a, b)| a.0 != b.0 || a.1 != b.1) {
                        bail!("{name}: signature differs from the interpreter's API ({api})");
                    }
                    *a
                }
                _ => {
                    warnings.push(format!("no {name}"));
                    u32::MAX
                }
            };
            self.code[super::H_ENTRIES + i] = addr;
        }
        for (i, d) in HEADER_DEFINES.iter().enumerate() {
            let v = defines.get(*d).and_then(|v| v.trim().trim_end_matches(['u', 'U']).parse::<i64>().ok()).unwrap_or(0);
            self.code[super::H_ENTRIES + API.len() + i] = v as u32;
        }
        // stack depth: frames along the deepest call chain
        let mut memo = HashMap::new();
        let mut stack = 0;
        for name in self.frames.keys().cloned().collect::<Vec<_>>() {
            stack = stack.max(self.depth(&name, &mut memo, 0)?);
        }
        let need = frame0 + super::glsl::ENTRY_SCRATCH + stack;
        if need > super::VM_REGS {
            bail!("the program needs {need} registers per thread (the interpreter has {})", super::VM_REGS);
        }
        Ok(Program { words: std::mem::take(&mut self.code), stack, warnings })
    }

    fn depth(&self, name: &str, memo: &mut HashMap<String, u32>, level: u32) -> Result<u32> {
        if let Some(&d) = memo.get(name) {
            return Ok(d);
        }
        if level > 64 {
            bail!("recursion through {name}");
        }
        let Some((frame, calls)) = self.frames.get(name) else { return Ok(0) };
        let mut d = *frame;
        for (at, callee) in calls {
            d = d.max(at + self.depth(callee, memo, level + 1)?);
        }
        memo.insert(name.to_string(), d);
        Ok(d)
    }

    fn api_sig(&self, api: &str) -> Result<FuncSig> {
        let toks = super::lex::tokenize(&format!("{api};"), &HashMap::new())?;
        let mut p = Parser::new(&toks, self.structs.iter().map(|s| s.name.clone()));
        let items = p.items()?;
        let Some(Item::Func(f)) = items.first() else { bail!("bad API signature {api}") };
        self.sig(f, &HashMap::new())
    }

    fn sig(&self, f: &Func, defines: &HashMap<String, String>) -> Result<FuncSig> {
        let ret = self.ty(&f.ret)?;
        let mut params = Vec::new();
        for Param { qual, ty, dims, .. } in &f.params {
            let t = array_of(self.ty(ty)?, dims, defines).ok_or_else(|| anyhow!("{}: bad array size", f.name))?;
            let by_ref = *qual != Qual::In || t.aggregate();
            params.push((*qual, t, by_ref));
        }
        Ok(FuncSig { ret, params })
    }

    fn function(&mut self, f: &Func, body: &[Stmt], _defines: &HashMap<String, String>) -> Result<()> {
        let sig = self.sigs[&f.name].clone();
        self.addrs.insert(f.name.clone(), self.here());
        let ret_size = self.size(&sig.ret);
        let mut scope = HashMap::new();
        let mut top = ret_size;
        for (p, (_, t, by_ref)) in f.params.iter().zip(&sig.params) {
            if *by_ref {
                scope.insert(p.name.clone(), Place::Ptr { reg: top, off: 0, ty: t.clone() });
                top += 1;
            } else {
                scope.insert(p.name.clone(), Place::Direct { op: top, ty: t.clone() });
                top += self.size(t);
            }
        }
        self.f = Some(FuncState { scopes: vec![scope], top, max: top, ret: sig.ret.clone(), breaks: vec![], conts: vec![], calls: vec![], name: f.name.clone() });
        for s in body {
            self.stmt(s)?;
        }
        self.emit(Op::RET, 0, 0, 0, 0, 0);
        let st = self.f.take().unwrap();
        self.frames.insert(st.name, (st.max, st.calls));
        Ok(())
    }

    // ---- statements ----
    fn stmt(&mut self, s: &Stmt) -> Result<()> {
        match s {
            Stmt::Empty => {}
            Stmt::Block(v) => {
                self.fs().scopes.push(HashMap::new());
                let m = self.mark();
                for s in v {
                    self.stmt(s)?;
                }
                self.reset(m);
                self.fs().scopes.pop();
            }
            Stmt::Decl { ty, ty_dims, vars, line } => {
                let base = array_of(self.ty(ty).with_context(|| format!("line {line}"))?, ty_dims, &HashMap::new()).ok_or_else(|| anyhow!("line {line}: bad array size"))?;
                for v in vars {
                    let mut t = array_of(base.clone(), &v.dims, &HashMap::new()).ok_or_else(|| anyhow!("line {line}: bad array size"))?;
                    if let (Ty::Array(e, 0), Some(Expr::ArrayCtor(_, args))) = (&t, &v.init) {
                        t = Ty::Array(e.clone(), args.len() as u32);
                    }
                    let n = self.size(&t);
                    let at = self.tmp(n);
                    if let Some(e) = &v.init {
                        let m = self.mark();
                        let val = self.expr(e).with_context(|| format!("line {line}"))?;
                        let val = self.convert(val, &t).with_context(|| format!("line {line}: initialising {}", v.name))?;
                        if val.op != at {
                            self.emit(Op::MOV, n, at, val.op, 0, 0);
                        }
                        self.reset(m);
                    }
                    self.fs().scopes.last_mut().unwrap().insert(v.name.clone(), Place::Direct { op: at, ty: t });
                }
            }
            Stmt::Expr(e, line) => {
                let m = self.mark();
                self.expr(e).with_context(|| format!("line {line}"))?;
                self.reset(m);
            }
            Stmt::If(c, t, e) => {
                let m = self.mark();
                let cv = self.cond(c)?;
                let jz = self.emit(Op::JZ, 0, cv, 0, 0, 0);
                self.reset(m);
                self.scoped(t)?;
                if let Some(e) = e {
                    let jmp = self.emit(Op::JMP, 0, 0, 0, 0, 0);
                    let h = self.here();
                    self.patch(jz, h);
                    self.scoped(e)?;
                    let h = self.here();
                    self.patch(jmp, h);
                } else {
                    let h = self.here();
                    self.patch(jz, h);
                }
            }
            Stmt::While(c, body) => {
                let top = self.here();
                let m = self.mark();
                let cv = self.cond(c)?;
                let jz = self.emit(Op::JZ, 0, cv, 0, 0, 0);
                self.reset(m);
                self.fs().breaks.push(vec![]);
                self.fs().conts.push(vec![]);
                self.scoped(body)?;
                self.emit(Op::JMP, 0, 0, top, 0, 0);
                let end = self.here();
                self.patch(jz, end);
                for b in self.fs().breaks.pop().unwrap() {
                    self.patch(b, end);
                }
                for c in self.fs().conts.pop().unwrap() {
                    self.patch(c, top);
                }
            }
            Stmt::For(init, cond, step, body) => {
                self.fs().scopes.push(HashMap::new());
                let m0 = self.mark();
                if let Some(i) = init {
                    self.stmt(i)?;
                }
                let top = self.here();
                let jz = match cond {
                    Some(c) => {
                        let m = self.mark();
                        let cv = self.cond(c)?;
                        let j = self.emit(Op::JZ, 0, cv, 0, 0, 0);
                        self.reset(m);
                        Some(j)
                    }
                    None => None,
                };
                self.fs().breaks.push(vec![]);
                self.fs().conts.push(vec![]);
                self.scoped(body)?;
                let cont = self.here();
                if let Some(s) = step {
                    let m = self.mark();
                    self.expr(s)?;
                    self.reset(m);
                }
                self.emit(Op::JMP, 0, 0, top, 0, 0);
                let end = self.here();
                if let Some(j) = jz {
                    self.patch(j, end);
                }
                for b in self.fs().breaks.pop().unwrap() {
                    self.patch(b, end);
                }
                for c in self.fs().conts.pop().unwrap() {
                    self.patch(c, cont);
                }
                self.reset(m0);
                self.fs().scopes.pop();
            }
            Stmt::Switch(sel, cases) => {
                let m = self.mark();
                let v = self.expr(sel)?;
                let sc = v.ty.sc().filter(|_| v.ty.is_scalar()).ok_or_else(|| anyhow!("switch on a {:?}", v.ty))?;
                let selv = self.tmp(1);
                self.emit(Op::MOV, 1, selv, v.op, 0, 0);
                // the jump table: one compare per case
                let mut jumps = Vec::new();
                for (label, _) in cases {
                    if let Some(l) = label {
                        let lv = self.expr(l)?;
                        let lv = self.convert(lv, &Ty::V(sc, 1))?;
                        let t = self.tmp(1);
                        self.emit(Op::XEQ, 1, t, selv, lv.op, 0);
                        jumps.push(Some(self.emit(Op::JNZ, 0, t, 0, 0, 0)));
                    } else {
                        jumps.push(None);
                    }
                }
                let to_default = self.emit(Op::JMP, 0, 0, 0, 0, 0);
                self.fs().breaks.push(vec![]);
                let mut default_at = None;
                for ((label, body), j) in cases.iter().zip(&jumps) {
                    let h = self.here();
                    match j {
                        Some(j) => self.patch(*j, h),
                        None => default_at = Some(h),
                    }
                    let _ = label;
                    self.fs().scopes.push(HashMap::new());
                    for s in body {
                        self.stmt(s)?;
                    }
                    self.fs().scopes.pop();
                }
                let end = self.here();
                self.patch(to_default, default_at.unwrap_or(end));
                for b in self.fs().breaks.pop().unwrap() {
                    self.patch(b, end);
                }
                self.reset(m);
            }
            Stmt::Return(e, line) => {
                if let Some(e) = e {
                    let m = self.mark();
                    let rt = self.fs().ret.clone();
                    let v = self.expr(e).with_context(|| format!("line {line}"))?;
                    let v = self.convert(v, &rt).with_context(|| format!("line {line}: return value"))?;
                    let n = self.size(&rt);
                    if v.op != 0 {
                        self.emit(Op::MOV, n, 0, v.op, 0, 0);
                    }
                    self.reset(m);
                }
                self.emit(Op::RET, 0, 0, 0, 0, 0);
            }
            Stmt::Break => {
                let j = self.emit(Op::JMP, 0, 0, 0, 0, 0);
                self.fs().breaks.last_mut().ok_or_else(|| anyhow!("break outside a loop or switch"))?.push(j);
            }
            Stmt::Continue => {
                let j = self.emit(Op::JMP, 0, 0, 0, 0, 0);
                self.fs().conts.last_mut().ok_or_else(|| anyhow!("continue outside a loop"))?.push(j);
            }
        }
        Ok(())
    }

    fn scoped(&mut self, s: &Stmt) -> Result<()> {
        self.fs().scopes.push(HashMap::new());
        let m = self.mark();
        self.stmt(s)?;
        self.reset(m);
        self.fs().scopes.pop();
        Ok(())
    }

    /// A condition: a bool operand.
    fn cond(&mut self, e: &Expr) -> Result<u32> {
        let v = self.expr(e)?;
        if v.ty != Ty::BOOL {
            bail!("condition is a {:?}, not a bool", v.ty);
        }
        Ok(v.op)
    }

    // ---- places ----
    fn lookup(&self, name: &str) -> Option<Place> {
        if let Some(f) = &self.f {
            for s in f.scopes.iter().rev() {
                if let Some(p) = s.get(name) {
                    return Some(p.clone());
                }
            }
        }
        if let Some((a, t)) = self.globals.get(name) {
            return Some(Place::Direct { op: a | ABS, ty: t.clone() });
        }
        self.lib.globals.get(name).map(|t| Place::Lib { name: name.to_string(), ty: t.clone() })
    }

    fn place(&mut self, e: &Expr) -> Result<Place> {
        match e {
            Expr::Var(n) => self.lookup(n).ok_or_else(|| anyhow!("unknown variable '{n}'")),
            Expr::Index(a, i) => {
                let base = self.place(a)?;
                let (elem, n) = match base.ty() {
                    Ty::Array(t, n) => ((**t).clone(), *n),
                    Ty::V(s, k) => (Ty::V(*s, 1), *k as u32),
                    Ty::Mat3 => (Ty::V(Sc::F, 3), 3),
                    t => bail!("indexing a {t:?}"),
                };
                let es = self.size(&elem);
                if let Some(k) = const_int(i) {
                    if k < 0 || k as u32 >= n {
                        bail!("index {k} out of bounds ({n})");
                    }
                    return base.sub(k as u32 * es, elem).ok_or_else(|| anyhow!("indexing this value"));
                }
                // dynamic: an absolute address in a register
                let iv = self.expr(i)?;
                if !matches!(iv.ty, Ty::V(Sc::I | Sc::U, 1)) {
                    bail!("index is a {:?}", iv.ty);
                }
                let addr = self.address(&base)?;
                let r = self.tmp(1);
                self.emit(Op::LEA, 0, r, addr, iv.op, es);
                Ok(Place::Ptr { reg: r, off: 0, ty: elem })
            }
            Expr::Field(a, f) => {
                if let Expr::Var(v) = &**a
                    && v == "pc"
                {
                    bail!("pc.{f} is not an lvalue");
                }
                let base = self.place(a)?;
                match base.ty().clone() {
                    Ty::Struct(si) => {
                        let (_, ft, off) = self.structs[si].fields.iter().find(|(n, ..)| n == f).cloned().ok_or_else(|| anyhow!("no field '{f}' in {}", self.structs[si].name))?;
                        match &base {
                            Place::Lib { .. } => bail!("field of a library global ({f})"),
                            _ => base.sub(off, ft).ok_or_else(|| anyhow!("field access")),
                        }
                    }
                    Ty::V(s, n) => {
                        let comps = swizzle(f, n)?;
                        let ty = Ty::V(s, comps.len() as u8);
                        let contiguous = comps.windows(2).all(|w| w[1] == w[0] + 1);
                        if contiguous && let Some(p) = base.sub(comps[0], ty.clone()) {
                            return Ok(p);
                        }
                        Ok(Place::Comps { base: Box::new(base), comps, ty })
                    }
                    t => bail!("field '{f}' of a {t:?}"),
                }
            }
            _ => bail!("not an lvalue"),
        }
    }

    /// An operand holding the absolute address of a place (direct or pointer).
    fn address(&mut self, p: &Place) -> Result<u32> {
        match p {
            Place::Direct { op, .. } => {
                let r = self.tmp(1);
                self.emit(Op::ADDR, 0, r, *op, 0, 0);
                Ok(r)
            }
            Place::Ptr { reg, off, .. } => {
                if *off == 0 {
                    return Ok(*reg);
                }
                let k = self.konst(&[*off], Ty::UINT);
                let r = self.tmp(1);
                self.emit(Op::XADD, 1, r, *reg, k, 0);
                Ok(r)
            }
            _ => bail!("cannot take the address of this value"),
        }
    }

    fn load(&mut self, p: &Place) -> Result<Val> {
        match p {
            Place::Direct { op, ty } => Ok(Val { op: *op, ty: ty.clone() }),
            Place::Ptr { reg, off, ty } => {
                let n = self.size(ty);
                let t = self.tmp(n);
                self.emit(Op::LDI, 0, t, *reg, *off, n);
                Ok(Val { op: t, ty: ty.clone() })
            }
            Place::Comps { base, comps, ty } => {
                let b = self.load(base)?;
                let t = self.tmp(comps.len() as u32);
                let sel = comps.iter().enumerate().fold(0u32, |s, (i, c)| s | (c << (2 * i)));
                self.emit(Op::SWZ, comps.len() as u32, t, b.op, sel, 0);
                Ok(Val { op: t, ty: ty.clone() })
            }
            Place::Lib { name, ty } => {
                let id = self.native_id(&format!("__get_{name}"), &[])?;
                let n = self.size(ty);
                let t = self.tmp(n);
                self.emit(Op::NAT, 0, t, id as u32, 0, 0);
                Ok(Val { op: t, ty: ty.clone() })
            }
        }
    }

    fn store(&mut self, p: &Place, v: Val) -> Result<()> {
        let v = self.convert(v, p.ty())?;
        match p {
            Place::Direct { op, ty } => {
                let n = self.size(ty);
                if *op != v.op {
                    self.emit(Op::MOV, n, *op, v.op, 0, 0);
                }
            }
            Place::Ptr { reg, off, ty } => {
                let n = self.size(ty);
                self.emit(Op::STI, 0, *reg, v.op, *off, n);
            }
            Place::Comps { base, comps, .. } => {
                for (i, c) in comps.iter().enumerate() {
                    let sc = base.ty().sc().unwrap();
                    let part = base.sub(*c, Ty::V(sc, 1)).ok_or_else(|| anyhow!("swizzle write"))?;
                    self.store(&part, Val { op: v.op + i as u32, ty: Ty::V(sc, 1) })?;
                }
            }
            Place::Lib { name, ty } => {
                let id = self.native_id(&format!("__set_{name}"), std::slice::from_ref(ty))?;
                let n = self.size(ty);
                let a = self.tmp(n);
                self.emit(Op::MOV, n, a, v.op, 0, 0);
                self.emit(Op::NAT, 0, 0, id as u32, a, 0);
            }
        }
        Ok(())
    }

    // ---- constants and conversions ----
    fn konst(&mut self, words: &[u32], ty: Ty) -> u32 {
        let t = self.tmp(words.len() as u32);
        for (i, ch) in words.chunks(3).enumerate() {
            let w = |k: usize| ch.get(k).copied().unwrap_or(0);
            self.emit(Op::LDK, ch.len() as u32, t + 3 * i as u32, w(0), w(1), w(2));
        }
        let _ = ty;
        t
    }

    fn convert(&mut self, v: Val, to: &Ty) -> Result<Val> {
        if v.ty == *to {
            return Ok(v);
        }
        match (&v.ty, to) {
            (Ty::V(Sc::I, n), Ty::V(Sc::F, m)) if n == m => Ok(self.unop(Op::I2F, &v, to.clone())),
            (Ty::V(Sc::U, n), Ty::V(Sc::F, m)) if n == m => Ok(self.unop(Op::U2F, &v, to.clone())),
            (Ty::V(Sc::I, n), Ty::V(Sc::U, m)) if n == m => Ok(Val { op: v.op, ty: to.clone() }),
            _ => bail!("cannot use a {:?} as a {:?}", v.ty, to),
        }
    }

    /// Explicit conversion (constructor-style): any scalar kind to any scalar kind.
    fn cast(&mut self, v: Val, sc: Sc, n: u32) -> Result<Val> {
        let to = Ty::V(sc, n as u8);
        let from = v.ty.sc().ok_or_else(|| anyhow!("cannot convert a {:?}", v.ty))?;
        Ok(match (from, sc) {
            (a, b) if a == b => Val { op: v.op, ty: to },
            (Sc::F, Sc::I) => self.unop(Op::F2I, &v, to),
            (Sc::F, Sc::U) => self.unop(Op::F2U, &v, to),
            (Sc::I, Sc::F) => self.unop(Op::I2F, &v, to),
            (Sc::U, Sc::F) | (Sc::B, Sc::F) => self.unop(Op::U2F, &v, to),
            (Sc::I, Sc::U) | (Sc::U, Sc::I) | (Sc::B, Sc::I) | (Sc::B, Sc::U) => Val { op: v.op, ty: to },
            (Sc::F, Sc::B) => self.unop(Op::F2B, &v, to),
            (Sc::I | Sc::U, Sc::B) => self.unop(Op::X2B, &v, to),
            _ => bail!("cannot convert {from:?} to {sc:?}"),
        })
    }

    fn unop(&mut self, op: Op, v: &Val, ty: Ty) -> Val {
        let n = ty.n().max(1);
        let t = self.tmp(n);
        self.emit(op, n, t, v.op, 0, 0);
        Val { op: t, ty }
    }

    /// Widen a scalar to n components (for mixed scalar/vector operations).
    fn splat(&mut self, v: Val, n: u32) -> Val {
        if v.ty.n() == n || n <= 1 {
            return v;
        }
        let sc = v.ty.sc().unwrap();
        let t = self.tmp(n);
        self.emit(Op::SPLAT, n, t, v.op, 0, 0);
        Val { op: t, ty: Ty::V(sc, n as u8) }
    }

    /// Bring two operands to a common type (GLSL implicit conversions and scalar widening).
    fn unify(&mut self, a: Val, b: Val) -> Result<(Val, Val, Sc, u32)> {
        let (sa, sb) = (a.ty.sc().ok_or_else(|| anyhow!("operand is a {:?}", a.ty))?, b.ty.sc().ok_or_else(|| anyhow!("operand is a {:?}", b.ty))?);
        let sc = match (sa, sb) {
            (x, y) if x == y => x,
            (Sc::F, Sc::I | Sc::U) | (Sc::I | Sc::U, Sc::F) => Sc::F,
            (Sc::U, Sc::I) | (Sc::I, Sc::U) => Sc::U,
            _ => bail!("mixing {sa:?} and {sb:?}"),
        };
        let n = a.ty.n().max(b.ty.n());
        let a = if sa != sc { self.convert(a.clone(), &Ty::V(sc, a.ty.n() as u8))? } else { a };
        let b = if sb != sc { self.convert(b.clone(), &Ty::V(sc, b.ty.n() as u8))? } else { b };
        let a = self.splat(a, n);
        let b = self.splat(b, n);
        Ok((a, b, sc, n))
    }

    // ---- expressions ----
    fn expr(&mut self, e: &Expr) -> Result<Val> {
        match e {
            Expr::Float(f) => Ok(Val { op: self.konst(&[f.to_bits()], Ty::FLOAT), ty: Ty::FLOAT }),
            Expr::Int(i) => Ok(Val { op: self.konst(&[*i as i32 as u32], Ty::INT), ty: Ty::INT }),
            Expr::Uint(u) => Ok(Val { op: self.konst(&[*u], Ty::UINT), ty: Ty::UINT }),
            Expr::Bool(b) => Ok(Val { op: self.konst(&[*b as u32], Ty::BOOL), ty: Ty::BOOL }),
            Expr::Var(_) | Expr::Index(..) => {
                let p = self.place(e)?;
                self.load(&p)
            }
            Expr::Field(a, f) => {
                if let Expr::Var(v) = &**a
                    && v == "pc"
                {
                    let id = self.native_id(&format!("__pc_{f}"), &[])?;
                    let ret = self.lib.natives[id].ret.clone();
                    let t = self.tmp(1);
                    self.emit(Op::NAT, 0, t, id as u32, 0, 0);
                    return Ok(Val { op: t, ty: ret });
                }
                // a swizzle of an rvalue (e.g. a call result)
                if let Ok(p) = self.place(e) {
                    return self.load(&p);
                }
                let v = self.expr(a)?;
                let Ty::V(s, n) = v.ty else { bail!("field '{f}' of a {:?}", v.ty) };
                let comps = swizzle(f, n)?;
                let t = self.tmp(comps.len() as u32);
                let sel = comps.iter().enumerate().fold(0u32, |acc, (i, c)| acc | (c << (2 * i)));
                self.emit(Op::SWZ, comps.len() as u32, t, v.op, sel, 0);
                Ok(Val { op: t, ty: Ty::V(s, comps.len() as u8) })
            }
            Expr::Unary(op, a) => match *op {
                "++" | "--" => self.incdec(a, op, true),
                "-" => {
                    let v = self.expr(a)?;
                    let o = match v.ty.sc() {
                        Some(Sc::F) => Op::FNEG,
                        Some(Sc::I | Sc::U) => Op::INEG,
                        _ => bail!("negating a {:?}", v.ty),
                    };
                    let ty = v.ty.clone();
                    Ok(self.unop(o, &v, ty))
                }
                "!" => {
                    let v = self.expr(a)?;
                    if v.ty != Ty::BOOL {
                        bail!("! on a {:?}", v.ty);
                    }
                    Ok(self.unop(Op::LNOT, &v, Ty::BOOL))
                }
                "~" => {
                    let v = self.expr(a)?;
                    let ty = v.ty.clone();
                    Ok(self.unop(Op::INOT, &v, ty))
                }
                o => bail!("unary {o}"),
            },
            Expr::Post(op, a) => self.incdec(a, op, false),
            Expr::Bin(op, a, b) => self.binary(op, a, b),
            Expr::Assign(op, l, r) => {
                let p = self.place(l)?;
                let v = if *op == "=" {
                    self.expr(r)?
                } else {
                    let cur = self.load(&p)?;
                    let rv = self.expr(r)?;
                    self.arith(op, cur, rv)?
                };
                let v = self.convert(v, p.ty())?;
                self.store(&p, v.clone())?;
                Ok(v)
            }
            Expr::Ternary(c, a, b) => {
                let cv = self.cond(c)?;
                let jz = self.emit(Op::JZ, 0, cv, 0, 0, 0);
                let va = self.expr(a)?;
                let ty = va.ty.clone();
                let n = self.size(&ty);
                let res = self.tmp(n);
                self.emit(Op::MOV, n, res, va.op, 0, 0);
                let jmp = self.emit(Op::JMP, 0, 0, 0, 0, 0);
                let h = self.here();
                self.patch(jz, h);
                let vb = self.expr(b)?;
                let vb = self.convert(vb, &ty)?;
                self.emit(Op::MOV, n, res, vb.op, 0, 0);
                let h = self.here();
                self.patch(jmp, h);
                Ok(Val { op: res, ty })
            }
            Expr::Call(name, args) => self.call(name, args),
            Expr::ArrayCtor(tname, args) => {
                let et = self.ty(tname)?;
                let es = self.size(&et);
                let t = self.tmp(es * args.len() as u32);
                for (i, a) in args.iter().enumerate() {
                    let v = self.expr(a)?;
                    let v = self.convert(v, &et)?;
                    self.emit(Op::MOV, es, t + i as u32 * es, v.op, 0, 0);
                }
                Ok(Val { op: t, ty: Ty::Array(Box::new(et), args.len() as u32) })
            }
        }
    }

    fn incdec(&mut self, a: &Expr, op: &str, pre: bool) -> Result<Val> {
        let p = self.place(a)?;
        let cur = self.load(&p)?;
        let old = self.tmp(1);
        self.emit(Op::MOV, 1, old, cur.op, 0, 0);
        let one = match cur.ty.sc() {
            Some(Sc::F) => Expr::Float(1.0),
            Some(Sc::I) => Expr::Int(1),
            Some(Sc::U) => Expr::Uint(1),
            _ => bail!("{op} on a {:?}", cur.ty),
        };
        let ov = self.expr(&one)?;
        let new = self.arith(if op == "++" { "+" } else { "-" }, cur.clone(), ov)?;
        self.store(&p, new.clone())?;
        Ok(if pre { new } else { Val { op: old, ty: cur.ty } })
    }

    fn arith(&mut self, op: &str, a: Val, b: Val) -> Result<Val> {
        let (a, b, sc, n) = self.unify(a, b)?;
        let o = match (op, sc) {
            ("+", Sc::F) => Op::FADD,
            ("-", Sc::F) => Op::FSUB,
            ("*", Sc::F) => Op::FMUL,
            ("/", Sc::F) => Op::FDIV,
            ("+", Sc::I | Sc::U) => Op::XADD,
            ("-", Sc::I | Sc::U) => Op::XSUB,
            ("*", Sc::I | Sc::U) => Op::XMUL,
            ("/", Sc::I) => Op::IDIV,
            ("/", Sc::U) => Op::UDIV,
            ("%", Sc::I) => Op::IMOD,
            ("%", Sc::U) => Op::UMOD,
            ("&", Sc::I | Sc::U) => Op::AND,
            ("|", Sc::I | Sc::U) => Op::OR,
            ("^", Sc::I | Sc::U) => Op::XOR,
            ("<<", Sc::I | Sc::U) => Op::SHL,
            (">>", Sc::I) => Op::ISHR,
            (">>", Sc::U) => Op::USHR,
            _ => bail!("operator {op} on {sc:?}"),
        };
        let t = self.tmp(n);
        self.emit(o, n, t, a.op, b.op, 0);
        Ok(Val { op: t, ty: Ty::V(sc, n as u8) })
    }

    fn binary(&mut self, op: &str, a: &Expr, b: &Expr) -> Result<Val> {
        match op {
            "&&" | "||" => {
                // short-circuit: res = a; if (res == (op == "||")) skip b
                let av = self.cond(a)?;
                let res = self.tmp(1);
                self.emit(Op::MOV, 1, res, av, 0, 0);
                let j = self.emit(if op == "&&" { Op::JZ } else { Op::JNZ }, 0, res, 0, 0, 0);
                let bv = self.cond(b)?;
                self.emit(Op::MOV, 1, res, bv, 0, 0);
                let h = self.here();
                self.patch(j, h);
                Ok(Val { op: res, ty: Ty::BOOL })
            }
            "<" | ">" | "<=" | ">=" | "==" | "!=" => {
                let av = self.expr(a)?;
                let bv = self.expr(b)?;
                if av.ty == Ty::BOOL && bv.ty == Ty::BOOL && (op == "==" || op == "!=") {
                    let t = self.tmp(1);
                    self.emit(if op == "==" { Op::XEQ } else { Op::XNE }, 1, t, av.op, bv.op, 0);
                    return Ok(Val { op: t, ty: Ty::BOOL });
                }
                let (av, bv, sc, n) = self.unify(av, bv)?;
                // a > b is b < a
                let (x, y, o) = match (op, sc) {
                    ("<", Sc::F) => (av, bv, Op::FLT),
                    (">", Sc::F) => (bv, av, Op::FLT),
                    ("<=", Sc::F) => (av, bv, Op::FLE),
                    (">=", Sc::F) => (bv, av, Op::FLE),
                    ("==", Sc::F) => (av, bv, Op::FEQ),
                    ("!=", Sc::F) => (av, bv, Op::FNE),
                    ("<", Sc::I) => (av, bv, Op::ILT),
                    (">", Sc::I) => (bv, av, Op::ILT),
                    ("<=", Sc::I) => (av, bv, Op::ILE),
                    (">=", Sc::I) => (bv, av, Op::ILE),
                    ("<", Sc::U) => (av, bv, Op::ULT),
                    (">", Sc::U) => (bv, av, Op::ULT),
                    ("<=", Sc::U) => (av, bv, Op::ULE),
                    (">=", Sc::U) => (bv, av, Op::ULE),
                    ("==", _) => (av, bv, Op::XEQ),
                    ("!=", _) => (av, bv, Op::XNE),
                    _ => bail!("comparison {op} on {sc:?}"),
                };
                if n > 1 && !matches!(op, "==" | "!=") {
                    bail!("{op} on vectors");
                }
                let t = self.tmp(1);
                self.emit(o, n, t, x.op, y.op, 0);
                Ok(Val { op: t, ty: Ty::BOOL })
            }
            _ => {
                let av = self.expr(a)?;
                let bv = self.expr(b)?;
                if op == "*" && (av.ty == Ty::Mat3 || bv.ty == Ty::Mat3) {
                    bail!("matrix products are not supported by the interpreter");
                }
                self.arith(op, av, bv)
            }
        }
    }

    // ---- calls ----
    fn native_id(&self, name: &str, args: &[Ty]) -> Result<usize> {
        let cands = self.natives.get(name).ok_or_else(|| anyhow!("unknown function '{name}'"))?;
        // exact, then with implicit conversions
        for exact in [true, false] {
            for &i in cands {
                let n = &self.lib.natives[i];
                if n.params.len() != args.len() {
                    continue;
                }
                let ok = n.params.iter().zip(args).all(|((_, pt), at)| {
                    pt == at
                        || (!exact
                            && matches!((at, pt), (Ty::V(Sc::I, a), Ty::V(Sc::F | Sc::U, b)) | (Ty::V(Sc::U, a), Ty::V(Sc::F, b)) if a == b))
                });
                if ok {
                    return Ok(i);
                }
            }
        }
        bail!("no '{name}' taking ({args:?})")
    }

    fn call(&mut self, name: &str, args: &[Expr]) -> Result<Val> {
        // constructors and conversions
        if let Some(t) = type_of_name(name, &self.struct_ids) {
            return self.construct(t, args);
        }
        if let Some(v) = self.builtin(name, args)? {
            return Ok(v);
        }
        if let Some(sig) = self.sigs.get(name).cloned() {
            return self.call_vm(name, &sig, args);
        }
        self.call_native(name, args)
    }

    fn call_vm(&mut self, name: &str, sig: &FuncSig, args: &[Expr]) -> Result<Val> {
        if sig.params.len() != args.len() {
            bail!("{name} takes {} arguments", sig.params.len());
        }
        let ret_size = self.size(&sig.ret);
        // the callee's frame: return value, then the parameters
        let psize: u32 = sig.params.iter().map(|(_, t, r)| if *r { 1 } else { self.size(t) }).sum();
        // evaluate the arguments first (they may call functions too), then lay them out
        let mut vals = Vec::new();
        for ((q, t, by_ref), a) in sig.params.iter().zip(args) {
            if *by_ref {
                let p = self.place(a).with_context(|| format!("argument of {name} passed by reference"))?;
                if *p.ty() != *t {
                    bail!("{name}: a {:?} passed for a {:?}", p.ty(), t);
                }
                let _ = q;
                vals.push(self.address(&p)?);
            } else {
                let v = self.expr(a)?;
                let v = self.convert(v, t)?;
                vals.push(v.op);
            }
        }
        let base = self.tmp(ret_size + psize);
        let mut at = base + ret_size;
        for ((_, t, by_ref), v) in sig.params.iter().zip(vals) {
            let n = if *by_ref { 1 } else { self.size(t) };
            self.emit(Op::MOV, n, at, v, 0, 0);
            at += n;
        }
        let insn = self.emit(Op::CALL, 0, 0, 0, base, 0);
        self.call_patches.push((insn, name.to_string()));
        self.fs().calls.push((base, name.to_string()));
        Ok(Val { op: base, ty: sig.ret.clone() })
    }

    fn call_native(&mut self, name: &str, args: &[Expr]) -> Result<Val> {
        // out arguments are places; the rest values
        let cands = self.natives.get(name).cloned().ok_or_else(|| anyhow!("unknown function '{name}'"))?;
        let outs: Vec<bool> = self.lib.natives[cands[0]].params.iter().map(|(q, _)| *q != Qual::In).collect();
        if outs.len() != args.len() {
            bail!("{name} takes {} arguments", outs.len());
        }
        let mut vals = Vec::new();
        let mut places = Vec::new();
        for (a, &out) in args.iter().zip(&outs) {
            if out {
                let p = self.place(a)?;
                vals.push(Val { op: 0, ty: p.ty().clone() });
                places.push(Some(p));
            } else {
                vals.push(self.expr(a)?);
                places.push(None);
            }
        }
        let tys: Vec<Ty> = vals.iter().map(|v| v.ty.clone()).collect();
        let id = self.native_id(name, &tys)?;
        let native: Native = self.lib.natives[id].clone();
        let total: u32 = native.params.iter().map(|(_, t)| self.size(t)).sum();
        let base = self.tmp(total);
        let mut at = base;
        for (((q, pt), v), p) in native.params.iter().zip(&vals).zip(&places) {
            let n = self.size(pt);
            match p {
                Some(p) if *q == Qual::InOut => {
                    let cur = self.load(p)?;
                    self.emit(Op::MOV, n, at, cur.op, 0, 0);
                }
                Some(_) => {}
                None => {
                    let cv = self.convert(v.clone(), pt)?;
                    self.emit(Op::MOV, n, at, cv.op, 0, 0);
                }
            }
            at += n;
        }
        let rs = self.size(&native.ret);
        let dst = self.tmp(rs);
        self.emit(Op::NAT, 0, dst, id as u32, base, 0);
        // copy out arguments back
        let mut at = base;
        for ((_, pt), p) in native.params.iter().zip(&places) {
            let n = self.size(pt);
            if let Some(p) = p {
                self.store(p, Val { op: at, ty: pt.clone() })?;
            }
            at += n;
        }
        Ok(Val { op: dst, ty: native.ret })
    }

    fn construct(&mut self, t: Ty, args: &[Expr]) -> Result<Val> {
        match &t {
            Ty::V(sc, n) => {
                let n = *n as u32;
                let mut vals = Vec::new();
                for a in args {
                    vals.push(self.expr(a)?);
                }
                if n == 1 {
                    let v = vals.into_iter().next().ok_or_else(|| anyhow!("empty conversion"))?;
                    let first = if v.ty.n() > 1 { Val { op: v.op, ty: Ty::V(v.ty.sc().unwrap(), 1) } } else { v };
                    return self.cast(first, *sc, 1);
                }
                // a single scalar fills every component
                if vals.len() == 1 && vals[0].ty.is_scalar() {
                    let v = self.cast(vals.remove(0), *sc, 1)?;
                    return Ok(self.splat(v, n));
                }
                let res = self.tmp(n);
                let mut filled = 0;
                for v in vals {
                    if filled >= n {
                        break;
                    }
                    let Ty::V(_, k) = v.ty else { bail!("constructing a vector from a {:?}", v.ty) };
                    let take = (k as u32).min(n - filled);
                    let part = Val { op: v.op, ty: Ty::V(v.ty.sc().unwrap(), take as u8) };
                    let c = self.cast(part, *sc, take)?;
                    self.emit(Op::MOV, take, res + filled, c.op, 0, 0);
                    filled += take;
                }
                if filled != n {
                    bail!("vector constructor fills {filled} of {n} components");
                }
                Ok(Val { op: res, ty: t })
            }
            Ty::Struct(si) => {
                let fields = self.structs[*si].fields.clone();
                if fields.len() != args.len() {
                    bail!("{} takes {} fields", self.structs[*si].name, fields.len());
                }
                let size = self.structs[*si].size;
                let res = self.tmp(size);
                for ((_, ft, off), a) in fields.iter().zip(args) {
                    let v = self.expr(a)?;
                    let v = self.convert(v, ft)?;
                    let n = self.size(ft);
                    self.emit(Op::MOV, n, res + off, v.op, 0, 0);
                }
                Ok(Val { op: res, ty: t })
            }
            _ => bail!("constructing a {t:?} is not supported by the interpreter"),
        }
    }

    /// GLSL builtins the VM runs as instructions.
    fn builtin(&mut self, name: &str, args: &[Expr]) -> Result<Option<Val>> {
        let f1 = |o: Op| Some((o, 1usize));
        let float1 = match name {
            "floor" => f1(Op::FFLOOR),
            "ceil" => f1(Op::FCEIL),
            "fract" => f1(Op::FFRACT),
            "sqrt" => f1(Op::FSQRT),
            "inversesqrt" => f1(Op::FINVSQRT),
            "sin" => f1(Op::FSIN),
            "cos" => f1(Op::FCOS),
            "tan" => f1(Op::FTAN),
            "asin" => f1(Op::FASIN),
            "acos" => f1(Op::FACOS),
            "exp" => f1(Op::FEXP),
            "log" => f1(Op::FLOG),
            "exp2" => f1(Op::FEXP2),
            "log2" => f1(Op::FLOG2),
            "round" => f1(Op::FROUND),
            "trunc" => f1(Op::FTRUNC),
            "normalize" => f1(Op::FNORM),
            _ => None,
        };
        if let Some((o, _)) = float1 {
            let v = self.expr(&args[0])?;
            let v = self.float_arg(v)?;
            let ty = v.ty.clone();
            return Ok(Some(self.unop(o, &v, ty)));
        }
        Ok(Some(match name {
            "abs" | "sign" => {
                let v = self.expr(&args[0])?;
                let o = match (name, v.ty.sc()) {
                    ("abs", Some(Sc::F)) => Op::FABS,
                    ("abs", Some(Sc::I)) => Op::IABS,
                    ("sign", Some(Sc::F)) => Op::FSIGN,
                    ("sign", Some(Sc::I)) => Op::ISIGN,
                    _ => bail!("{name} of a {:?}", v.ty),
                };
                let ty = v.ty.clone();
                self.unop(o, &v, ty)
            }
            "atan" if args.len() == 1 => {
                let v = self.expr(&args[0])?;
                let ty = v.ty.clone();
                self.unop(Op::FATAN, &v, ty)
            }
            "min" | "max" | "pow" | "mod" | "atan" | "step" => {
                let a = self.expr(&args[0])?;
                let b = self.expr(&args[1])?;
                // step(edge, x): a scalar edge widens to x
                let (a, b, sc, n) = self.unify(a, b)?;
                let o = match (name, sc) {
                    ("min", Sc::F) => Op::FMIN,
                    ("max", Sc::F) => Op::FMAX,
                    ("min", Sc::I) => Op::IMIN,
                    ("max", Sc::I) => Op::IMAX,
                    ("min", Sc::U) => Op::UMIN,
                    ("max", Sc::U) => Op::UMAX,
                    ("pow", Sc::F) => Op::FPOW,
                    ("mod", Sc::F) => Op::FMOD,
                    ("atan", Sc::F) => Op::FATAN2,
                    ("step", Sc::F) => Op::FSTEP,
                    _ => bail!("{name} on {sc:?}"),
                };
                let t = self.tmp(n);
                self.emit(o, n, t, a.op, b.op, 0);
                Val { op: t, ty: Ty::V(sc, n as u8) }
            }
            "mix" | "clamp" | "smoothstep" => {
                let a = self.expr(&args[0])?;
                let b = self.expr(&args[1])?;
                let c = self.expr(&args[2])?;
                // the widest operand decides the width; scalars widen
                let n = a.ty.n().max(b.ty.n()).max(c.ty.n());
                let sc = if [&a, &b, &c].iter().any(|v| v.ty.sc() == Some(Sc::F)) { Sc::F } else { a.ty.sc().unwrap_or(Sc::F) };
                let mut ops = Vec::new();
                for v in [a, b, c] {
                    let v = if v.ty.sc() != Some(sc) { self.convert(v.clone(), &Ty::V(sc, v.ty.n() as u8))? } else { v };
                    ops.push(self.splat(v, n).op);
                }
                let o = match (name, sc) {
                    ("mix", Sc::F) => Op::FMIX,
                    ("clamp", Sc::F) => Op::FCLAMP,
                    ("clamp", Sc::I) => Op::ICLAMP,
                    ("clamp", Sc::U) => Op::UCLAMP,
                    ("smoothstep", Sc::F) => Op::FSMOOTH,
                    _ => bail!("{name} on {sc:?}"),
                };
                let t = self.tmp(n);
                self.emit(o, n, t, ops[0], ops[1], ops[2]);
                Val { op: t, ty: Ty::V(sc, n as u8) }
            }
            "dot" | "distance" | "cross" | "length" => {
                let a = self.expr(&args[0])?;
                let a = self.float_arg(a)?;
                let n = a.ty.n();
                if name == "length" {
                    let t = self.tmp(1);
                    self.emit(Op::FLEN, n, t, a.op, 0, 0);
                    return Ok(Some(Val { op: t, ty: Ty::FLOAT }));
                }
                let b = self.expr(&args[1])?;
                let b = self.float_arg(b)?;
                let (o, out) = match name {
                    "dot" => (Op::FDOT, 1),
                    "distance" => (Op::FDIST, 1),
                    _ => (Op::FCROSS, 3),
                };
                let t = self.tmp(out);
                self.emit(o, n, t, a.op, b.op, 0);
                Val { op: t, ty: Ty::V(Sc::F, out as u8) }
            }
            "floatBitsToUint" | "floatBitsToInt" | "uintBitsToFloat" | "intBitsToFloat" => {
                let v = self.expr(&args[0])?;
                let sc = match name {
                    "floatBitsToUint" => Sc::U,
                    "floatBitsToInt" => Sc::I,
                    _ => Sc::F,
                };
                Val { op: v.op, ty: Ty::V(sc, v.ty.n() as u8) }
            }
            _ => return Ok(None),
        }))
    }

    fn float_arg(&mut self, v: Val) -> Result<Val> {
        match v.ty.sc() {
            Some(Sc::F) => Ok(v),
            Some(Sc::I | Sc::U) => {
                let n = v.ty.n();
                self.convert(v, &Ty::V(Sc::F, n as u8))
            }
            _ => bail!("a {:?} where a float is needed", v.ty),
        }
    }
}

/// Component indices of a swizzle (`xyzw`, `rgba` or `stpq`).
fn swizzle(f: &str, n: u8) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for c in f.chars() {
        let i = match c {
            'x' | 'r' | 's' => 0,
            'y' | 'g' | 't' => 1,
            'z' | 'b' | 'p' => 2,
            'w' | 'a' | 'q' => 3,
            _ => bail!("bad swizzle .{f}"),
        };
        if i >= n as u32 {
            bail!("swizzle .{f} out of range");
        }
        out.push(i);
    }
    if out.is_empty() || out.len() > 4 {
        bail!("bad swizzle .{f}");
    }
    Ok(out)
}

impl Program {
    pub fn entry(&self, api: usize) -> Option<u32> {
        self.words.get(super::H_ENTRIES + api).copied().filter(|&a| a != u32::MAX)
    }
}

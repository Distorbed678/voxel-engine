//! The interpreter's GLSL, generated from the same `Library` the bytecode compiler uses: opcode
//! constants, load/store helpers per type, the native dispatch, and the API glue that implements
//! the generated-code API (`super::API`) by running the VM.

use std::collections::HashMap;
use std::fmt::Write as _;

use anyhow::{Result, bail};

use super::parse::{Item, Parser, Qual};
use super::{ABS, API, H_ENTRIES, H_FRAME, H_INIT, H_SEA, HEADER_DEFINES, Library, NativeKind, Op, Sc, Ty};

/// Words below an API call's frame for its out-parameters (`PropLayer`, `bool carved`, ...).
pub const ENTRY_SCRATCH: u32 = 32;

fn glsl_ty(t: &Ty, lib: &Library) -> String {
    match t {
        Ty::Void => "void".into(),
        Ty::Mat3 => "mat3".into(),
        Ty::V(s, 1) => match s {
            Sc::F => "float",
            Sc::I => "int",
            Sc::U => "uint",
            Sc::B => "bool",
        }
        .into(),
        Ty::V(s, n) => format!(
            "{}vec{n}",
            match s {
                Sc::F => "",
                Sc::I => "i",
                Sc::U => "u",
                Sc::B => "b",
            }
        ),
        Ty::Struct(i) => lib.structs[*i].name.clone(),
        Ty::Array(..) => "void".into(),
    }
}

/// Load / store helpers: `T vm_ld_T(uint a)`, `void vm_st_T(uint a, T v)`.
fn helpers(lib: &Library) -> String {
    let mut s = String::new();
    let scalars = [("float", "uintBitsToFloat(R[a])", "floatBitsToUint(v)"), ("int", "int(R[a])", "uint(v)"), ("uint", "R[a]", "v"), ("bool", "R[a] != 0u", "v ? 1u : 0u")];
    for (t, ld, st) in scalars {
        let _ = writeln!(s, "{t} vm_ld_{t}(uint a) {{ return {ld}; }}\nvoid vm_st_{t}(uint a, {t} v) {{ R[a] = {st}; }}");
    }
    for (p, base) in [("", "float"), ("i", "int"), ("u", "uint"), ("b", "bool")] {
        for n in 2..=4 {
            let t = format!("{p}vec{n}");
            let comps: Vec<String> = (0..n).map(|i| format!("vm_ld_{base}(a + {i}u)")).collect();
            let _ = writeln!(s, "{t} vm_ld_{t}(uint a) {{ return {t}({}); }}", comps.join(", "));
            let sts: Vec<String> = (0..n).map(|i| format!("vm_st_{base}(a + {i}u, v[{i}]);")).collect();
            let _ = writeln!(s, "void vm_st_{t}(uint a, {t} v) {{ {} }}", sts.join(" "));
        }
    }
    let _ = writeln!(s, "mat3 vm_ld_mat3(uint a) {{ return mat3(vm_ld_vec3(a), vm_ld_vec3(a + 3u), vm_ld_vec3(a + 6u)); }}");
    let _ = writeln!(s, "void vm_st_mat3(uint a, mat3 v) {{ vm_st_vec3(a, v[0]); vm_st_vec3(a + 3u, v[1]); vm_st_vec3(a + 6u, v[2]); }}");
    for st in &lib.structs {
        // structs with array fields aren't passed to natives
        if st.fields.iter().any(|(_, t, _)| matches!(t, Ty::Array(..))) {
            continue;
        }
        let mut ld = String::new();
        let mut sto = String::new();
        for (name, t, off) in &st.fields {
            let tn = glsl_ty(t, lib);
            let _ = write!(ld, " v.{name} = vm_ld_{tn}(a + {off}u);");
            let _ = write!(sto, " vm_st_{tn}(a + {off}u, v.{name});");
        }
        let (pre, post) = if st.props_only { ("#ifdef PROPS_PASS\n", "#endif\n") } else { ("", "") };
        let _ = writeln!(s, "{pre}{0} vm_ld_{0}(uint a) {{ {0} v;{ld} return v; }}\nvoid vm_st_{0}(uint a, {0} v) {{{sto} }}\n{post}", st.name);
    }
    s
}

/// `void vm_native(uint id, uint dst, uint arg)`: run native `id` with its arguments at `arg`
/// (absolute), its result to `dst`, out-arguments written back.
fn natives(lib: &Library) -> String {
    let mut s = String::from("void vm_native(uint id, uint dst, uint arg) {\n    switch (id) {\n");
    let mut props = false;
    for (i, n) in lib.natives.iter().enumerate() {
        if n.props_only != props {
            s.push_str(if n.props_only { "#ifdef PROPS_PASS\n" } else { "#endif\n" });
            props = n.props_only;
        }
        let mut pre = String::new();
        let mut args = Vec::new();
        let mut post = String::new();
        let mut at = 0u32;
        for (k, (q, t)) in n.params.iter().enumerate() {
            let tn = glsl_ty(t, lib);
            let size = super::codegen::size_of(t, &lib.structs);
            if *q == Qual::In {
                args.push(format!("vm_ld_{tn}(arg + {at}u)"));
            } else {
                let init = if *q == Qual::InOut { format!(" = vm_ld_{tn}(arg + {at}u)") } else { String::new() };
                let _ = write!(pre, "{tn} o{k}{init}; ");
                args.push(format!("o{k}"));
                let _ = write!(post, " vm_st_{tn}(arg + {at}u, o{k});");
            }
            at += size;
        }
        let call = match &n.kind {
            NativeKind::Func => format!("{}({})", n.name, args.join(", ")),
            NativeKind::Get(g) => g.clone(),
            NativeKind::Set(g) => {
                let _ = writeln!(s, "        case {i}u: {g} = {}; break;", args[0]);
                continue;
            }
            NativeKind::Push(f) => format!("pc.{f}"),
        };
        let body = if n.ret == Ty::Void { format!("{call};") } else { format!("vm_st_{}(dst, {call});", glsl_ty(&n.ret, lib)) };
        let _ = writeln!(s, "        case {i}u: {{ {pre}{body}{post} }} break;");
    }
    if props {
        s.push_str("#endif\n");
    }
    s.push_str("        default: break;\n    }\n}\n");
    s
}

/// The API: each function writes its arguments into an entry frame, runs the VM and reads the
/// results. Columns are opaque here: the VM keeps the column at address 0.
fn api(lib: &Library) -> Result<String> {
    let mut s = String::new();
    let _ = writeln!(s, "struct WgColumn {{ uint vmColumn; }};\nfloat gSeaLevel;");
    for (i, d) in HEADER_DEFINES.iter().enumerate() {
        let _ = writeln!(s, "#define {d} vm_word({}u)", H_ENTRIES + API.len() + i);
    }
    let names: Vec<String> = lib.structs.iter().map(|s| s.name.clone()).chain(["WgColumn".to_string()]).collect();
    let mut ids: HashMap<String, usize> = lib.structs.iter().enumerate().map(|(i, s)| (s.name.clone(), i)).collect();
    // WgColumn is the program's: treat it as an opaque marker
    ids.insert("WgColumn".into(), usize::MAX);
    for (ai, sig) in API.iter().enumerate() {
        let toks = super::lex::tokenize(&format!("{sig};"), &HashMap::new())?;
        let mut p = Parser::new(&toks, names.clone());
        let Some(Item::Func(f)) = p.items()?.into_iter().next() else { bail!("bad API signature {sig}") };
        let ret = if f.ret == "void" { Ty::Void } else { super::codegen::type_of_name(&f.ret, &ids).unwrap() };
        let ret_size = super::codegen::size_of(&ret, &lib.structs);
        let mut body = String::new();
        let _ = writeln!(body, "    uint e = vm_word({}u);", H_ENTRIES + ai);
        let default = match &ret {
            Ty::Void => "return;".to_string(),
            t => format!("return {}(0);", glsl_ty(t, lib)),
        };
        let _ = writeln!(body, "    if (e == 0xFFFFFFFFu) {{ {} {default} }}", f.params.iter().filter(|p| p.qual == Qual::Out && p.ty != "WgColumn").map(|p| format!("{} = {}(0);", p.name, p.ty)).collect::<Vec<_>>().join(" "));
        let _ = writeln!(body, "    uint s = vm_word({H_FRAME}u), fb = s + {ENTRY_SCRATCH}u;");
        let mut at = ret_size;
        let mut scratch = 0u32;
        let mut after = String::new();
        for prm in &f.params {
            if prm.ty == "WgColumn" {
                let _ = writeln!(body, "    R[fb + {at}u] = 0u;  // the column");
                if prm.qual == Qual::Out {
                    let _ = writeln!(after, "    {}.vmColumn = 0u;", prm.name);
                }
                at += 1;
                continue;
            }
            let t = super::codegen::type_of_name(&prm.ty, &ids).unwrap();
            let tn = glsl_ty(&t, lib);
            let size = super::codegen::size_of(&t, &lib.structs);
            if prm.qual == Qual::In && !t.aggregate() {
                let _ = writeln!(body, "    vm_st_{tn}(fb + {at}u, {});", prm.name);
                at += size;
            } else {
                if prm.qual != Qual::Out {
                    let _ = writeln!(body, "    vm_st_{tn}(s + {scratch}u, {});", prm.name);
                }
                let _ = writeln!(body, "    R[fb + {at}u] = s + {scratch}u;");
                if prm.qual != Qual::In {
                    let _ = writeln!(after, "    {} = vm_ld_{tn}(s + {scratch}u);", prm.name);
                }
                scratch += size;
                at += 1;
            }
        }
        if scratch > ENTRY_SCRATCH {
            bail!("{}: out-parameters need {scratch} words of scratch", f.name);
        }
        if f.name == "wg_init" {
            // global initialisers first; afterwards the main templates read gSeaLevel
            let _ = writeln!(body, "    vm_clear();\n    vm_run(vm_word({H_INIT}u), fb);");
        }
        let _ = writeln!(body, "    vm_run(e, fb);");
        body.push_str(&after);
        if f.name == "wg_init" {
            let _ = writeln!(body, "    uint sea = vm_word({H_SEA}u);\n    gSeaLevel = sea == 0xFFFFFFFFu ? 0.0 : uintBitsToFloat(R[sea]);");
        }
        if ret != Ty::Void {
            let _ = writeln!(body, "    return vm_ld_{}(fb);", glsl_ty(&ret, lib));
        }
        let _ = writeln!(s, "{} {{\n{body}}}", sig);
    }
    Ok(s)
}

/// The generated part of the interpreter: everything between the prelude and the main template.
pub fn interpreter_source(lib: &Library) -> Result<String> {
    let mut s = String::from("// ---- world generator interpreter (generated: crates/worldgen/src/vm) ----\n");
    for op in Op::ALL {
        let _ = writeln!(s, "#define VM_{} {}u", op.name(), *op as u32);
    }
    let _ = writeln!(s, "#define VM_ABS {ABS}u\n#define VM_REGS {}u", super::VM_REGS);
    s.push_str("#include \"worldgen/vm_core.glsl\"\n");
    s.push_str(&helpers(lib));
    s.push_str(&natives(lib));
    s.push_str("#include \"worldgen/vm_run.glsl\"\n");
    s.push_str(&api(lib)?);
    Ok(s)
}

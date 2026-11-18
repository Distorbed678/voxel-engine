//! Parser of the GLSL subset the world generator emits: structs, globals (const or not),
//! functions (prototypes and definitions); statements: declarations, expressions, blocks, if,
//! while, for, switch, return, break, continue; expressions with GLSL precedence.

use std::collections::HashSet;

use anyhow::{Result, anyhow, bail};

use super::lex::{Tok, Token};

#[derive(Clone, Debug)]
pub enum Expr {
    Float(f32),
    Int(i64),
    Uint(u32),
    Bool(bool),
    Var(String),
    /// prefix `-`, `!`, `~`, `++`, `--`
    Unary(&'static str, Box<Expr>),
    /// postfix `++`, `--`
    Post(&'static str, Box<Expr>),
    Bin(&'static str, Box<Expr>, Box<Expr>),
    /// `=`, `+=`, ...
    Assign(&'static str, Box<Expr>, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    /// function call, vector/matrix/struct constructor or conversion
    Call(String, Vec<Expr>),
    /// `T[N](...)` / `T[](...)`
    ArrayCtor(String, Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    /// struct field or swizzle
    Field(Box<Expr>, String),
}

#[derive(Clone, Debug)]
pub struct VarDecl {
    pub name: String,
    /// array dimensions after the name (`float h[3][MAXH]`)
    pub dims: Vec<Expr>,
    pub init: Option<Expr>,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Decl { ty: String, ty_dims: Vec<Expr>, vars: Vec<VarDecl>, line: u32 },
    Expr(Expr, u32),
    Block(Vec<Stmt>),
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    While(Expr, Box<Stmt>),
    For(Option<Box<Stmt>>, Option<Expr>, Option<Expr>, Box<Stmt>),
    /// (case value, None = default; statements after the label)
    Switch(Expr, Vec<(Option<Expr>, Vec<Stmt>)>),
    Return(Option<Expr>, u32),
    Break,
    Continue,
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qual {
    In,
    Out,
    InOut,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub qual: Qual,
    pub ty: String,
    pub name: String,
    pub dims: Vec<Expr>,
}

#[derive(Clone, Debug)]
pub struct Func {
    pub ret: String,
    pub name: String,
    pub params: Vec<Param>,
    pub body: Option<Vec<Stmt>>,
    pub line: u32,
}

#[derive(Clone, Debug)]
pub struct StructDef {
    pub name: String,
    /// (type, name, array dims)
    pub fields: Vec<(String, String, Vec<Expr>)>,
}

#[derive(Clone, Debug)]
pub enum Item {
    Struct(StructDef),
    Global { is_const: bool, ty: String, vars: Vec<VarDecl>, line: u32 },
    Func(Func),
}

pub const BUILTIN_TYPES: &[&str] = &[
    "void", "bool", "int", "uint", "float", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "uvec2", "uvec3", "uvec4", "bvec2", "bvec3", "bvec4", "mat3",
];

pub struct Parser<'a> {
    toks: &'a [Token],
    i: usize,
    pub types: HashSet<String>,
}

impl<'a> Parser<'a> {
    pub fn new(toks: &'a [Token], structs: impl IntoIterator<Item = String>) -> Self {
        let mut types: HashSet<String> = BUILTIN_TYPES.iter().map(|s| s.to_string()).collect();
        types.extend(structs);
        Self { toks, i: 0, types }
    }

    fn line(&self) -> u32 {
        self.toks.get(self.i).or(self.toks.last()).map(|t| t.line).unwrap_or(0)
    }
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i).map(|t| &t.t)
    }
    fn peek_at(&self, k: usize) -> Option<&Tok> {
        self.toks.get(self.i + k).map(|t| &t.t)
    }
    fn is_p(&self, p: &str) -> bool {
        matches!(self.peek(), Some(Tok::P(q)) if *q == p)
    }
    fn is_ident(&self, w: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if s == w)
    }
    fn eat_p(&mut self, p: &str) -> bool {
        if self.is_p(p) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect_p(&mut self, p: &str) -> Result<()> {
        if self.eat_p(p) {
            Ok(())
        } else {
            bail!("line {}: expected '{p}', found {:?}", self.line(), self.peek())
        }
    }
    fn ident(&mut self) -> Result<String> {
        match self.peek() {
            Some(Tok::Ident(s)) => {
                let s = s.clone();
                self.i += 1;
                Ok(s)
            }
            t => bail!("line {}: expected a name, found {t:?}", self.line()),
        }
    }
    fn is_type_here(&self) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if self.types.contains(s))
    }

    // ---- top level ----
    pub fn items(&mut self) -> Result<Vec<Item>> {
        let mut out = Vec::new();
        while self.peek().is_some() {
            if self.eat_p(";") {
                continue;
            }
            if self.is_ident("struct") {
                self.i += 1;
                let name = self.ident()?;
                self.types.insert(name.clone());
                self.expect_p("{")?;
                let mut fields = Vec::new();
                while !self.eat_p("}") {
                    let ty = self.ident()?;
                    loop {
                        let n = self.ident()?;
                        let dims = self.dims()?;
                        fields.push((ty.clone(), n, dims));
                        if !self.eat_p(",") {
                            break;
                        }
                    }
                    self.expect_p(";")?;
                }
                self.expect_p(";")?;
                out.push(Item::Struct(StructDef { name, fields }));
                continue;
            }
            let line = self.line();
            let is_const = self.is_ident("const");
            if is_const {
                self.i += 1;
            }
            if !self.is_type_here() {
                bail!("line {line}: expected a declaration, found {:?}", self.peek());
            }
            let ty = self.ident()?;
            let name = self.ident()?;
            if self.eat_p("(") {
                let mut params = Vec::new();
                if !self.eat_p(")") {
                    loop {
                        let qual = if self.is_ident("in") {
                            self.i += 1;
                            Qual::In
                        } else if self.is_ident("out") {
                            self.i += 1;
                            Qual::Out
                        } else if self.is_ident("inout") {
                            self.i += 1;
                            Qual::InOut
                        } else {
                            Qual::In
                        };
                        if self.is_ident("const") {
                            self.i += 1;
                        }
                        let pty = self.ident()?;
                        if pty == "void" && self.is_p(")") {
                            break;
                        }
                        let pname = self.ident()?;
                        let dims = self.dims()?;
                        params.push(Param { qual, ty: pty, name: pname, dims });
                        if !self.eat_p(",") {
                            break;
                        }
                    }
                    self.expect_p(")")?;
                }
                let body = if self.eat_p(";") { None } else { Some(self.block_body()?) };
                out.push(Item::Func(Func { ret: ty, name, params, body, line }));
                continue;
            }
            let vars = self.var_rest(name)?;
            out.push(Item::Global { is_const, ty, vars, line });
        }
        Ok(out)
    }

    fn dims(&mut self) -> Result<Vec<Expr>> {
        let mut d = Vec::new();
        while self.eat_p("[") {
            if self.eat_p("]") {
                d.push(Expr::Int(-1));
                continue;
            }
            d.push(self.expr()?);
            self.expect_p("]")?;
        }
        Ok(d)
    }

    /// After a declaration's type and first name: `dims (= init)?, name dims (= init)? ... ;`
    fn var_rest(&mut self, first: String) -> Result<Vec<VarDecl>> {
        let mut vars = Vec::new();
        let mut name = first;
        loop {
            let dims = self.dims()?;
            let init = if self.eat_p("=") { Some(self.assign()?) } else { None };
            vars.push(VarDecl { name, dims, init });
            if !self.eat_p(",") {
                break;
            }
            name = self.ident()?;
        }
        self.expect_p(";")?;
        Ok(vars)
    }

    fn block_body(&mut self) -> Result<Vec<Stmt>> {
        self.expect_p("{")?;
        let mut v = Vec::new();
        while !self.eat_p("}") {
            if self.peek().is_none() {
                bail!("unterminated block");
            }
            v.push(self.stmt()?);
        }
        Ok(v)
    }

    // ---- statements ----
    fn stmt(&mut self) -> Result<Stmt> {
        let line = self.line();
        if self.is_p("{") {
            return Ok(Stmt::Block(self.block_body()?));
        }
        if self.eat_p(";") {
            return Ok(Stmt::Empty);
        }
        if let Some(Tok::Ident(w)) = self.peek() {
            match w.as_str() {
                "if" => {
                    self.i += 1;
                    self.expect_p("(")?;
                    let c = self.expr()?;
                    self.expect_p(")")?;
                    let t = Box::new(self.stmt()?);
                    let e = if self.is_ident("else") {
                        self.i += 1;
                        Some(Box::new(self.stmt()?))
                    } else {
                        None
                    };
                    return Ok(Stmt::If(c, t, e));
                }
                "while" => {
                    self.i += 1;
                    self.expect_p("(")?;
                    let c = self.expr()?;
                    self.expect_p(")")?;
                    return Ok(Stmt::While(c, Box::new(self.stmt()?)));
                }
                "for" => {
                    self.i += 1;
                    self.expect_p("(")?;
                    let init = if self.eat_p(";") { None } else { Some(Box::new(self.simple_stmt()?)) };
                    let cond = if self.is_p(";") { None } else { Some(self.expr()?) };
                    self.expect_p(";")?;
                    let step = if self.is_p(")") { None } else { Some(self.expr()?) };
                    self.expect_p(")")?;
                    return Ok(Stmt::For(init, cond, step, Box::new(self.stmt()?)));
                }
                "switch" => {
                    self.i += 1;
                    self.expect_p("(")?;
                    let v = self.expr()?;
                    self.expect_p(")")?;
                    self.expect_p("{")?;
                    let mut cases: Vec<(Option<Expr>, Vec<Stmt>)> = Vec::new();
                    while !self.eat_p("}") {
                        if self.is_ident("case") {
                            self.i += 1;
                            let e = self.expr()?;
                            self.expect_p(":")?;
                            cases.push((Some(e), Vec::new()));
                        } else if self.is_ident("default") {
                            self.i += 1;
                            self.expect_p(":")?;
                            cases.push((None, Vec::new()));
                        } else {
                            let s = self.stmt()?;
                            cases.last_mut().ok_or_else(|| anyhow!("line {line}: statement before the first case"))?.1.push(s);
                        }
                    }
                    return Ok(Stmt::Switch(v, cases));
                }
                "return" => {
                    self.i += 1;
                    let e = if self.is_p(";") { None } else { Some(self.expr()?) };
                    self.expect_p(";")?;
                    return Ok(Stmt::Return(e, line));
                }
                "break" => {
                    self.i += 1;
                    self.expect_p(";")?;
                    return Ok(Stmt::Break);
                }
                "continue" => {
                    self.i += 1;
                    self.expect_p(";")?;
                    return Ok(Stmt::Continue);
                }
                _ => {}
            }
        }
        self.simple_stmt()
    }

    /// A declaration or an expression statement (both end with `;`).
    fn simple_stmt(&mut self) -> Result<Stmt> {
        let line = self.line();
        let save = self.i;
        let mut is_const = false;
        if self.is_ident("const") || self.is_ident("precise") {
            self.i += 1;
            is_const = true;
        }
        if self.is_type_here() {
            // `T name` or `T[N] name` is a declaration; `T(` is a constructor expression
            let decl = matches!(self.peek_at(1), Some(Tok::Ident(_))) || matches!(self.peek_at(1), Some(Tok::P("[")));
            if decl {
                let ty = self.ident()?;
                let ty_dims = self.dims()?;
                if matches!(self.peek(), Some(Tok::Ident(_))) {
                    let first = self.ident()?;
                    let vars = self.var_rest(first)?;
                    return Ok(Stmt::Decl { ty, ty_dims, vars, line });
                }
                // `T[N](...)`: an array constructor expression after all
                self.i = save;
            }
        }
        if is_const {
            self.i = save + 1;
        }
        let e = self.expr()?;
        self.expect_p(";")?;
        Ok(Stmt::Expr(e, line))
    }

    // ---- expressions ----
    /// An expression (the comma operator is not part of the subset).
    pub fn expr(&mut self) -> Result<Expr> {
        self.assign()
    }

    fn assign(&mut self) -> Result<Expr> {
        let lhs = self.ternary()?;
        for op in ["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="] {
            if self.is_p(op) {
                self.i += 1;
                let rhs = self.assign()?;
                let op: &'static str = match op {
                    "=" => "=",
                    "+=" => "+",
                    "-=" => "-",
                    "*=" => "*",
                    "/=" => "/",
                    "%=" => "%",
                    "&=" => "&",
                    "|=" => "|",
                    "^=" => "^",
                    "<<=" => "<<",
                    _ => ">>",
                };
                return Ok(Expr::Assign(op, Box::new(lhs), Box::new(rhs)));
            }
        }
        Ok(lhs)
    }

    fn ternary(&mut self) -> Result<Expr> {
        let c = self.binary(0)?;
        if self.eat_p("?") {
            let a = self.assign()?;
            self.expect_p(":")?;
            let b = self.assign()?;
            return Ok(Expr::Ternary(Box::new(c), Box::new(a), Box::new(b)));
        }
        Ok(c)
    }

    fn binary(&mut self, level: usize) -> Result<Expr> {
        const LEVELS: &[&[&str]] = &[&["||"], &["&&"], &["|"], &["^"], &["&"], &["==", "!="], &["<", ">", "<=", ">="], &["<<", ">>"], &["+", "-"], &["*", "/", "%"]];
        if level == LEVELS.len() {
            return self.unary();
        }
        let mut lhs = self.binary(level + 1)?;
        while let Some(Tok::P(p)) = self.peek() {
            let p = *p;
            if !LEVELS[level].contains(&p) {
                break;
            }
            self.i += 1;
            let rhs = self.binary(level + 1)?;
            lhs = Expr::Bin(p, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr> {
        for op in ["-", "!", "~", "++", "--", "+"] {
            if self.is_p(op) {
                self.i += 1;
                let e = self.unary()?;
                let op: &'static str = match op {
                    "-" => "-",
                    "!" => "!",
                    "~" => "~",
                    "++" => "++",
                    "--" => "--",
                    _ => return Ok(e),
                };
                return Ok(Expr::Unary(op, Box::new(e)));
            }
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Expr> {
        let mut e = self.primary()?;
        loop {
            if self.eat_p("[") {
                let i = self.expr()?;
                self.expect_p("]")?;
                e = Expr::Index(Box::new(e), Box::new(i));
            } else if self.eat_p(".") {
                let f = self.ident()?;
                e = Expr::Field(Box::new(e), f);
            } else if self.eat_p("++") {
                e = Expr::Post("++", Box::new(e));
            } else if self.eat_p("--") {
                e = Expr::Post("--", Box::new(e));
            } else {
                break;
            }
        }
        Ok(e)
    }

    fn args(&mut self) -> Result<Vec<Expr>> {
        let mut a = Vec::new();
        if self.eat_p(")") {
            return Ok(a);
        }
        loop {
            a.push(self.assign()?);
            if self.eat_p(")") {
                break;
            }
            self.expect_p(",")?;
        }
        Ok(a)
    }

    fn primary(&mut self) -> Result<Expr> {
        let line = self.line();
        let Some(t) = self.peek().cloned() else { bail!("unexpected end of source") };
        self.i += 1;
        Ok(match t {
            Tok::Float(f) => Expr::Float(f),
            Tok::Int(i) => Expr::Int(i),
            Tok::Uint(u) => Expr::Uint(u),
            Tok::P("(") => {
                let e = self.expr()?;
                self.expect_p(")")?;
                e
            }
            Tok::Ident(w) if w == "true" => Expr::Bool(true),
            Tok::Ident(w) if w == "false" => Expr::Bool(false),
            Tok::Ident(w) => {
                if self.types.contains(&w) && self.is_p("[") {
                    // array constructor T[N](...) / T[](...)
                    self.i += 1;
                    if !self.eat_p("]") {
                        self.expr()?;
                        self.expect_p("]")?;
                    }
                    self.expect_p("(")?;
                    Expr::ArrayCtor(w, self.args()?)
                } else if self.eat_p("(") {
                    Expr::Call(w, self.args()?)
                } else {
                    Expr::Var(w)
                }
            }
            other => bail!("line {line}: unexpected {other:?}"),
        })
    }
}

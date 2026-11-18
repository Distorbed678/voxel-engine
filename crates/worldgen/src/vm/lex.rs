//! Tokens of the GLSL subset, and the preprocessor (object-like `#define`s, `#ifdef` /
//! `#ifndef` / `#else` / `#endif`; other directives are dropped).

use std::collections::HashMap;

use anyhow::{Result, bail};

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ident(String),
    Float(f32),
    Int(i64),
    Uint(u32),
    /// punctuation and operators
    P(&'static str),
}

const PUNCT: &[&str] = &[
    "<<=", ">>=", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "(", ")", "[", "]", "{", "}", ".",
    ",", ";", "?", ":", "+", "-", "*", "/", "%", "<", ">", "=", "!", "~", "&", "|", "^",
];

/// A token and its line (for error messages).
#[derive(Clone, Debug)]
pub struct Token {
    pub t: Tok,
    pub line: u32,
}

/// Strip comments, keeping line breaks.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                if b[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i += 2;
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// Preprocess: collect object-like `#define`s into `defines` (later ones win) and drop the lines
/// of inactive `#ifdef` branches. Macro expansion happens in `tokenize`.
pub fn preprocess(src: &str, defines: &mut HashMap<String, String>) -> String {
    let src = strip_comments(src);
    let mut out = String::with_capacity(src.len());
    // (active, any branch taken)
    let mut stack: Vec<(bool, bool)> = Vec::new();
    for line in src.lines() {
        let t = line.trim_start();
        let active = stack.iter().all(|s| s.0);
        if let Some(d) = t.strip_prefix('#') {
            let d = d.trim_start();
            let mut parts = d.splitn(2, char::is_whitespace);
            let dir = parts.next().unwrap_or("");
            let rest = parts.next().unwrap_or("").trim();
            match dir {
                "ifdef" | "ifndef" => {
                    let has = defines.contains_key(rest.split_whitespace().next().unwrap_or(""));
                    let on = if dir == "ifdef" { has } else { !has };
                    stack.push((on, on));
                }
                "else" => {
                    if let Some(s) = stack.last_mut() {
                        s.0 = !s.1;
                        s.1 = true;
                    }
                }
                "endif" => {
                    stack.pop();
                }
                "define" if active => {
                    let mut p = rest.splitn(2, char::is_whitespace);
                    let name = p.next().unwrap_or("");
                    // function-like macros are library-internal: skip them
                    if !name.contains('(') && !name.is_empty() {
                        defines.insert(name.to_string(), p.next().unwrap_or("").trim().to_string());
                    }
                }
                _ => {}
            }
            out.push('\n');
            continue;
        }
        if active {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Tokenize preprocessed source, expanding object-like macros (recursively).
pub fn tokenize(src: &str, defines: &HashMap<String, String>) -> Result<Vec<Token>> {
    let mut out = Vec::new();
    lex_into(src, defines, &mut out, 1, 0)?;
    Ok(out)
}

fn lex_into(src: &str, defines: &HashMap<String, String>, out: &mut Vec<Token>, line0: u32, depth: u32) -> Result<()> {
    if depth > 16 {
        bail!("macro expansion too deep");
    }
    let b = src.as_bytes();
    let mut i = 0;
    let mut line = line0;
    while i < b.len() {
        let c = b[i];
        if c == b'\n' {
            if depth == 0 {
                line += 1;
            }
            i += 1;
            continue;
        }
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if is_ident_start(c) {
            let s = i;
            while i < b.len() && is_ident(b[i]) {
                i += 1;
            }
            let w = &src[s..i];
            if let Some(v) = defines.get(w) {
                lex_into(v, defines, out, line, depth + 1)?;
            } else {
                out.push(Token { t: Tok::Ident(w.to_string()), line });
            }
            continue;
        }
        if c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            let s = i;
            if c == b'0' && i + 1 < b.len() && (b[i + 1] == b'x' || b[i + 1] == b'X') {
                i += 2;
                while i < b.len() && b[i].is_ascii_hexdigit() {
                    i += 1;
                }
                let v = u64::from_str_radix(&src[s + 2..i], 16)?;
                if i < b.len() && (b[i] == b'u' || b[i] == b'U') {
                    i += 1;
                    out.push(Token { t: Tok::Uint(v as u32), line });
                } else {
                    out.push(Token { t: Tok::Int(v as i64), line });
                }
                continue;
            }
            let mut float = false;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                float |= b[i] == b'.';
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                float = true;
                i += 1;
                if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                    i += 1;
                }
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let text = &src[s..i];
            if i < b.len() && (b[i] == b'f' || b[i] == b'F') && float {
                i += 1;
            }
            if float {
                out.push(Token { t: Tok::Float(text.parse()?), line });
            } else if i < b.len() && (b[i] == b'u' || b[i] == b'U') {
                i += 1;
                out.push(Token { t: Tok::Uint(text.parse::<u64>()? as u32), line });
            } else {
                out.push(Token { t: Tok::Int(text.parse()?), line });
            }
            continue;
        }
        let rest = &src[i..];
        let Some(p) = PUNCT.iter().find(|p| rest.starts_with(**p)) else {
            bail!("line {line}: unexpected character '{}'", c as char);
        };
        out.push(Token { t: Tok::P(p), line });
        i += p.len();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_and_lex() {
        let mut d = HashMap::new();
        let src = "#define N 3u\n#ifdef X\nfloat a;\n#else\nfloat b = 1.5e2 + N; // c\n#endif\nuint c = 0x10u << 2;";
        let p = preprocess(src, &mut d);
        let t: Vec<Tok> = tokenize(&p, &d).unwrap().into_iter().map(|t| t.t).collect();
        assert_eq!(
            t,
            [
                Tok::Ident("float".into()),
                Tok::Ident("b".into()),
                Tok::P("="),
                Tok::Float(150.0),
                Tok::P("+"),
                Tok::Uint(3),
                Tok::P(";"),
                Tok::Ident("uint".into()),
                Tok::Ident("c".into()),
                Tok::P("="),
                Tok::Uint(16),
                Tok::P("<<"),
                Tok::Int(2),
                Tok::P(";"),
            ]
        );
    }
}

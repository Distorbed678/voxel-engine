//! Node graph data model (serialised as RON, one file per graph).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::nodes::{self, PinType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum GraphKind {
    World,
    Zone,
    Biome,
    Function,
}

impl GraphKind {
    pub fn label(self) -> &'static str {
        match self {
            GraphKind::World => "World",
            GraphKind::Zone => "Zone",
            GraphKind::Biome => "Biome",
            GraphKind::Function => "Function",
        }
    }
    pub fn dir(self) -> &'static str {
        match self {
            GraphKind::World => "world",
            GraphKind::Zone => "zones",
            GraphKind::Biome => "biomes",
            GraphKind::Function => "functions",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    F(f32),
    I(i32),
    B(bool),
    /// enum option name
    E(String),
    /// block name
    Block(String),
    /// reference to another graph by name
    Ref(String),
    Color([f32; 3]),
    /// piecewise curve points (x, y), sorted by x
    Curve(Vec<[f32; 2]>),
}

impl Value {
    pub fn as_f32(&self) -> f32 {
        match self {
            Value::F(v) => *v,
            Value::I(v) => *v as f32,
            Value::B(b) => *b as i32 as f32,
            _ => 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: u32,
    pub kind: String,
    pub pos: [f32; 2],
    #[serde(default)]
    pub params: BTreeMap<String, Value>,
    #[serde(default)]
    pub comment: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Link {
    pub from_node: u32,
    pub from_pin: u8,
    pub to_node: u32,
    pub to_pin: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graph {
    pub name: String,
    pub kind: GraphKind,
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    #[serde(default)]
    pub next_id: u32,
}

impl Graph {
    pub fn new(name: &str, kind: GraphKind) -> Self {
        let mut g = Self { name: name.to_string(), kind, nodes: Vec::new(), links: Vec::new(), next_id: 1 };
        let out = match kind {
            GraphKind::World => "world_output",
            GraphKind::Zone => "zone_output",
            GraphKind::Biome => "biome_output",
            GraphKind::Function => "function_output",
        };
        g.add(out, [600.0, 200.0]);
        g
    }

    pub fn node(&self, id: u32) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: u32) -> Option<&mut Node> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// Add a node of `kind` with default parameters. Returns its id.
    pub fn add(&mut self, kind: &str, pos: [f32; 2]) -> u32 {
        let id = self.next_id.max(1);
        self.next_id = id + 1;
        let mut params = BTreeMap::new();
        if let Some(def) = nodes::def(kind) {
            for p in def.params {
                params.insert(p.name.to_string(), p.kind.default_value());
            }
        }
        self.nodes.push(Node { id, kind: kind.to_string(), pos, params, comment: String::new() });
        id
    }

    pub fn set(&mut self, id: u32, param: &str, v: Value) -> &mut Self {
        if let Some(n) = self.node_mut(id) {
            n.params.insert(param.to_string(), v);
        }
        self
    }

    pub fn remove_node(&mut self, id: u32) {
        self.nodes.retain(|n| n.id != id);
        self.links.retain(|l| l.from_node != id && l.to_node != id);
    }

    /// Connect `from.out_pin -> to.in_pin`, replacing any existing link into that input.
    pub fn connect(&mut self, from: u32, out_pin: u8, to: u32, in_pin: u8) -> bool {
        if from == to {
            return false;
        }
        let (Some(a), Some(b)) = (self.node(from), self.node(to)) else { return false };
        let (Some(da), Some(db)) = (nodes::def(&a.kind), nodes::def(&b.kind)) else { return false };
        let (Some(o), Some(i)) = (da.outputs.get(out_pin as usize), db.inputs.get(in_pin as usize)) else { return false };
        if !PinType::compatible(o.1, i.1) {
            return false;
        }
        if self.would_cycle(from, to) {
            return false;
        }
        self.links.retain(|l| !(l.to_node == to && l.to_pin == in_pin));
        self.links.push(Link { from_node: from, from_pin: out_pin, to_node: to, to_pin: in_pin });
        true
    }

    fn would_cycle(&self, from: u32, to: u32) -> bool {
        // adding from->to creates a cycle if `from` is reachable from `to`
        let mut stack = vec![to];
        let mut seen = std::collections::HashSet::new();
        while let Some(n) = stack.pop() {
            if n == from {
                return true;
            }
            if !seen.insert(n) {
                continue;
            }
            for l in &self.links {
                if l.from_node == n {
                    stack.push(l.to_node);
                }
            }
        }
        false
    }

    pub fn input_link(&self, node: u32, pin: u8) -> Option<&Link> {
        self.links.iter().find(|l| l.to_node == node && l.to_pin == pin)
    }

    pub fn output_node(&self) -> Option<&Node> {
        let kind = match self.kind {
            GraphKind::World => "world_output",
            GraphKind::Zone => "zone_output",
            GraphKind::Biome => "biome_output",
            GraphKind::Function => "function_output",
        };
        self.nodes.iter().find(|n| n.kind == kind)
    }
}

/// The complete generator: every graph, by kind.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Project {
    pub graphs: Vec<Graph>,
}

impl Project {
    pub fn get(&self, kind: GraphKind, name: &str) -> Option<&Graph> {
        self.graphs.iter().find(|g| g.kind == kind && g.name == name)
    }

    pub fn get_mut(&mut self, kind: GraphKind, name: &str) -> Option<&mut Graph> {
        self.graphs.iter_mut().find(|g| g.kind == kind && g.name == name)
    }

    pub fn names(&self, kind: GraphKind) -> Vec<String> {
        let mut v: Vec<String> = self.graphs.iter().filter(|g| g.kind == kind).map(|g| g.name.clone()).collect();
        v.sort();
        v
    }

    pub fn save_dir(&self, dir: &Path) -> Result<()> {
        for kind in [GraphKind::World, GraphKind::Zone, GraphKind::Biome, GraphKind::Function] {
            let d = dir.join(kind.dir());
            std::fs::create_dir_all(&d)?;
            // remove stale files
            if let Ok(rd) = std::fs::read_dir(&d) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.extension().map(|e| e == "ron").unwrap_or(false) {
                        let stem = p.file_stem().unwrap().to_string_lossy().to_string();
                        if self.get(kind, &stem).is_none() {
                            let _ = std::fs::remove_file(&p);
                        }
                    }
                }
            }
        }
        let cfg = ron::ser::PrettyConfig::new().depth_limit(4).struct_names(false);
        for g in &self.graphs {
            let path = dir.join(g.kind.dir()).join(format!("{}.ron", g.name));
            let s = ron::ser::to_string_pretty(g, cfg.clone())?;
            std::fs::write(&path, s).with_context(|| format!("writing {}", path.display()))?;
        }
        Ok(())
    }

    pub fn load_dir(dir: &Path) -> Result<Self> {
        let mut p = Project::default();
        for kind in [GraphKind::World, GraphKind::Zone, GraphKind::Biome, GraphKind::Function] {
            let d = dir.join(kind.dir());
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            let mut files: Vec<_> = rd.flatten().map(|e| e.path()).filter(|p| p.extension().map(|e| e == "ron").unwrap_or(false)).collect();
            files.sort();
            for f in files {
                let s = std::fs::read_to_string(&f)?;
                let mut g: Graph = ron::from_str(&s).with_context(|| format!("parsing {}", f.display()))?;
                g.kind = kind;
                g.next_id = g.next_id.max(g.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1);
                p.graphs.push(g);
            }
        }
        if p.graphs.is_empty() {
            anyhow::bail!("no graphs in {}", dir.display());
        }
        Ok(p)
    }
}

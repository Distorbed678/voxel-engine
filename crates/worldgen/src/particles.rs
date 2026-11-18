//! Particle types as data: each Particle graph (a Particle Output node) is one type. The engine
//! uploads them as a table the GPU particle system reads; Particles environment nodes pick which
//! types appear where (see `env`).

use crate::graph::{GraphKind, Project, Value};
use crate::nodes;

pub type Rgb = [f32; 3];

pub const SHAPES: [&str; 4] = ["Soft", "Streak", "Square", "Spark"];
pub const MOTIONS: [&str; 7] = ["Fall", "Drift", "Rise", "Swirl", "Wander", "Flock", "Orbit"];
pub const TIMES: [&str; 3] = ["Any", "Day", "Night"];
pub const NEAR: [&str; 7] = ["Anywhere", "Leaves", "Plants", "Water", "Glowing", "Ground", "Block"];

#[derive(Clone, Debug, PartialEq)]
pub struct ParticleDef {
    pub name: String,
    /// index into SHAPES
    pub shape: u32,
    pub color: Rgb,
    pub color2: Rgb,
    pub alpha: f32,
    /// half-width in blocks
    pub size: f32,
    pub glow: f32,
    pub blink: f32,
    /// index into MOTIONS
    pub motion: u32,
    pub speed: f32,
    pub sway: f32,
    pub wind: f32,
    /// seconds
    pub life: f32,
    /// share of the particle pool in calm weather and at the peak of a storm
    pub density: f32,
    pub storm_density: f32,
    /// index into TIMES
    pub time: u32,
    /// index into NEAR, and the block for "Block"
    pub near: u32,
    pub near_block: String,
    /// dies under a roof (rain, snow)
    pub open_sky: bool,
}

impl ParticleDef {
    /// The pool share it wants at weather intensity `weather` (0..1) and `daylight` (0..1).
    pub fn share(&self, weather: f32, daylight: f32) -> f32 {
        let time = match self.time {
            1 => daylight,
            2 => 1.0 - daylight,
            _ => 1.0,
        };
        (self.density + (self.storm_density - self.density) * weather.clamp(0.0, 1.0)) * time
    }
}

/// Every Particle graph's type, in the project's (sorted) order.
pub fn particle_defs(project: &Project) -> Vec<ParticleDef> {
    project
        .names(GraphKind::Particle)
        .iter()
        .filter_map(|name| {
            let g = project.get(GraphKind::Particle, name)?;
            let n = g.output_node()?;
            let value = |k: &str| {
                n.params
                    .get(k)
                    .cloned()
                    .or_else(|| nodes::def(&n.kind).and_then(|d| d.params.iter().find(|p| p.name == k)).map(|p| p.kind.default_value()))
            };
            let f = |k: &str| value(k).map_or(0.0, |v| v.as_f32());
            let c = |k: &str| match value(k) {
                Some(Value::Color(c)) => c,
                _ => [1.0; 3],
            };
            let e = |k: &str, opts: &[&str]| match value(k) {
                Some(Value::E(s)) => opts.iter().position(|o| *o == s).unwrap_or(0) as u32,
                _ => 0,
            };
            Some(ParticleDef {
                name: name.clone(),
                shape: e("shape", &SHAPES),
                color: c("color"),
                color2: c("color2"),
                alpha: f("alpha"),
                size: f("size"),
                glow: f("glow"),
                blink: f("blink"),
                motion: e("motion", &MOTIONS),
                speed: f("speed"),
                sway: f("sway"),
                wind: f("wind"),
                life: f("life"),
                density: f("density"),
                storm_density: f("storm_density"),
                time: e("time", &TIMES),
                near: e("near", &NEAR),
                near_block: match value("near_block") {
                    Some(Value::Block(b)) => b,
                    _ => String::new(),
                },
                open_sky: matches!(value("open_sky"), Some(Value::B(true))),
            })
        })
        .collect()
}

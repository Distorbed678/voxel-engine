//! Hytale-style world generation graphs: World -> Zones -> Biomes (+ reusable functions),
//! compiled to GLSL for the GPU generator.

pub mod compile;
pub mod connect;
pub mod defaults;
pub mod graph;
pub mod layout;
pub mod nodes;

pub use compile::{BiomeInfo, Compiled, ParamKey, ZoneInfo, compile};
pub use graph::{Graph, GraphKind, Link, Node, Project, Value};

//! The environment as graphs (roadmap item 22): sky, fog, colour grading, clouds and light.
//!
//! Environment nodes (Sky, Fog, Grading, Clouds, Light, and Environment to combine them) feed the
//! Environment inputs of the World Output, of each layer, of Zone Outputs and of Biome Outputs.
//! Each sets the groups it covers; the camera's environment is the built-in defaults overridden
//! group by group by its world, then its layer, zone and biome (the most specific wins). The
//! values are read on the CPU only: editing them never rebuilds or regenerates anything.

use crate::graph::{Graph, Node, Value};
use crate::nodes;

pub type Rgb = [f32; 3];

#[derive(Clone, Debug, PartialEq)]
pub struct SkyEnv {
    /// day, dusk, night
    pub zenith: [Rgb; 3],
    pub horizon: [Rgb; 3],
    /// the lower hemisphere (light from below), day and night
    pub ground: [Rgb; 2],
    pub sun: Rgb,
    pub sun_dusk: Rgb,
    /// disk size (1 = the default)
    pub sun_size: f32,
    pub moon: Rgb,
    pub moon_size: f32,
    pub stars: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FogEnv {
    pub color: Rgb,
    /// how much the fog takes its colour (the rest is the horizon's)
    pub strength: f32,
    pub density: f32,
    pub falloff: f32,
    pub start: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GradeEnv {
    pub tint: Rgb,
    pub saturation: f32,
    pub contrast: f32,
    pub exposure: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CloudEnv {
    pub coverage: f32,
    pub base: f32,
    pub thickness: f32,
    pub density: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LightEnv {
    /// multipliers of sunlight, moonlight, sky ambient and block light
    pub sun: f32,
    pub moon: f32,
    pub sky: f32,
    pub block: f32,
    /// what's left where nothing glows (caves)
    pub cave: f32,
}

/// The particle types (Particle graph names) around the camera, and a density multiplier.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticlesEnv {
    pub types: Vec<String>,
    pub density: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeatherEnv {
    /// seconds per cycle, share of the time it storms (0..1), peak intensity
    pub cycle: f32,
    pub storminess: f32,
    pub peak: f32,
    /// wind speed in calm weather and at the peak of a storm
    pub wind: f32,
    pub storm_wind: f32,
    /// at the peak: fog density multiplier, the storm's fog colour and how much it takes over
    pub fog: f32,
    pub fog_color: Rgb,
    pub fog_tint: f32,
    /// at the peak: cloud coverage added, sky and sunlight darkening, lightning per minute
    pub clouds: f32,
    pub darken: f32,
    pub lightning: f32,
}

impl WeatherEnv {
    /// The automatic weather intensity (0..1) at time `t` seconds: two slow waves, stormy for
    /// `storminess` of the time.
    pub fn intensity(&self, t: f64) -> f32 {
        let f = std::f64::consts::TAU / self.cycle.max(1.0) as f64;
        let w = (t * f).sin() * 0.6 + (t * f * 2.366 + 1.3).sin() * 0.4;
        let threshold = 1.0 - 2.0 * self.storminess;
        (((w as f32) - threshold) * 2.0).clamp(0.0, 1.0) * self.peak
    }
}

/// What one level (world, layer, zone, biome) sets: None leaves a group to the levels above.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EnvOverride {
    pub sky: Option<SkyEnv>,
    pub fog: Option<FogEnv>,
    pub grade: Option<GradeEnv>,
    pub clouds: Option<CloudEnv>,
    pub light: Option<LightEnv>,
    pub particles: Option<ParticlesEnv>,
    pub weather: Option<WeatherEnv>,
}

/// Every group set: the environment the renderer uses.
#[derive(Clone, Debug, PartialEq)]
pub struct Env {
    pub sky: SkyEnv,
    pub fog: FogEnv,
    pub grade: GradeEnv,
    pub clouds: CloudEnv,
    pub light: LightEnv,
    pub particles: ParticlesEnv,
    pub weather: WeatherEnv,
}

impl Default for Env {
    /// The engine's look before environments were graphs.
    fn default() -> Self {
        Env {
            sky: SkyEnv {
                zenith: [[0.22, 0.45, 0.92], [0.25, 0.3, 0.55], [0.006, 0.01, 0.03]],
                horizon: [[0.62, 0.78, 0.98], [1.0, 0.5, 0.25], [0.02, 0.03, 0.06]],
                ground: [[0.32, 0.28, 0.22], [0.02, 0.02, 0.025]],
                sun: [1.0, 0.96, 0.9],
                sun_dusk: [1.0, 0.55, 0.3],
                sun_size: 1.0,
                moon: [0.8, 0.85, 1.0],
                moon_size: 1.0,
                stars: 1.0,
            },
            fog: FogEnv { color: [0.7, 0.8, 0.95], strength: 0.55, density: 0.0012, falloff: 0.012, start: 32.0 },
            grade: GradeEnv { tint: [1.0; 3], saturation: 1.0, contrast: 1.0, exposure: 1.0 },
            clouds: CloudEnv { coverage: 0.45, base: 230.0, thickness: 90.0, density: 0.06 },
            light: LightEnv { sun: 1.0, moon: 1.0, sky: 1.0, block: 1.0, cave: 0.018 },
            particles: ParticlesEnv { types: Vec::new(), density: 1.0 },
            weather: WeatherEnv {
                cycle: 1530.0,
                storminess: 0.375,
                peak: 1.0,
                wind: 0.6,
                storm_wind: 2.2,
                fog: 2.0,
                fog_color: [0.55, 0.6, 0.68],
                fog_tint: 0.0,
                clouds: 0.3,
                darken: 0.0,
                lightning: 0.0,
            },
        }
    }
}

fn mix(a: &mut f32, b: f32, k: f32) {
    *a += (b - *a) * k;
}
fn mix3(a: &mut Rgb, b: &Rgb, k: f32) {
    for i in 0..3 {
        mix(&mut a[i], b[i], k);
    }
}

impl Env {
    /// The defaults overridden by each level in turn (world, layer, zone, biome: the last wins).
    pub fn resolve<'a>(levels: impl IntoIterator<Item = Option<&'a EnvOverride>>) -> Env {
        let mut e = Env::default();
        for o in levels.into_iter().flatten() {
            e.apply(o);
        }
        e
    }

    /// One named value (`group.field`, e.g. `fog.color`, `light.cave`, `sky.zenith_day`), for
    /// scripts and display.
    pub fn field(&self, name: &str) -> Option<Vec<f32>> {
        let (s, f, g, c, l) = (&self.sky, &self.fog, &self.grade, &self.clouds, &self.light);
        let v = |x: f32| Some(vec![x]);
        let col = |x: &Rgb| Some(x.to_vec());
        match name {
            "sky.zenith_day" => col(&s.zenith[0]),
            "sky.zenith_dusk" => col(&s.zenith[1]),
            "sky.zenith_night" => col(&s.zenith[2]),
            "sky.horizon_day" => col(&s.horizon[0]),
            "sky.horizon_dusk" => col(&s.horizon[1]),
            "sky.horizon_night" => col(&s.horizon[2]),
            "sky.ground_day" => col(&s.ground[0]),
            "sky.ground_night" => col(&s.ground[1]),
            "sky.sun" => col(&s.sun),
            "sky.sun_dusk" => col(&s.sun_dusk),
            "sky.sun_size" => v(s.sun_size),
            "sky.moon" => col(&s.moon),
            "sky.moon_size" => v(s.moon_size),
            "sky.stars" => v(s.stars),
            "fog.color" => col(&f.color),
            "fog.strength" => v(f.strength),
            "fog.density" => v(f.density),
            "fog.height_falloff" => v(f.falloff),
            "fog.start" => v(f.start),
            "grade.tint" => col(&g.tint),
            "grade.saturation" => v(g.saturation),
            "grade.contrast" => v(g.contrast),
            "grade.exposure" => v(g.exposure),
            "clouds.coverage" => v(c.coverage),
            "clouds.base" => v(c.base),
            "clouds.thickness" => v(c.thickness),
            "clouds.density" => v(c.density),
            "light.sun" => v(l.sun),
            "light.moon" => v(l.moon),
            "light.sky" => v(l.sky),
            "light.block" => v(l.block),
            "light.cave" => v(l.cave),
            "particles.density" => v(self.particles.density),
            "weather.wind" => v(self.weather.wind),
            "weather.storm_wind" => v(self.weather.storm_wind),
            "weather.fog" => v(self.weather.fog),
            "weather.fog_color" => col(&self.weather.fog_color),
            "weather.lightning" => v(self.weather.lightning),
            _ => None,
        }
    }

    /// Move a share `k` (0..1) of the way to `to`.
    pub fn lerp(&mut self, to: &Env, k: f32) {
        let (s, t) = (&mut self.sky, &to.sky);
        for i in 0..3 {
            mix3(&mut s.zenith[i], &t.zenith[i], k);
            mix3(&mut s.horizon[i], &t.horizon[i], k);
        }
        for i in 0..2 {
            mix3(&mut s.ground[i], &t.ground[i], k);
        }
        mix3(&mut s.sun, &t.sun, k);
        mix3(&mut s.sun_dusk, &t.sun_dusk, k);
        mix3(&mut s.moon, &t.moon, k);
        mix(&mut s.sun_size, t.sun_size, k);
        mix(&mut s.moon_size, t.moon_size, k);
        mix(&mut s.stars, t.stars, k);
        let (f, t) = (&mut self.fog, &to.fog);
        mix3(&mut f.color, &t.color, k);
        mix(&mut f.strength, t.strength, k);
        mix(&mut f.density, t.density, k);
        mix(&mut f.falloff, t.falloff, k);
        mix(&mut f.start, t.start, k);
        let (g, t) = (&mut self.grade, &to.grade);
        mix3(&mut g.tint, &t.tint, k);
        mix(&mut g.saturation, t.saturation, k);
        mix(&mut g.contrast, t.contrast, k);
        mix(&mut g.exposure, t.exposure, k);
        let (c, t) = (&mut self.clouds, &to.clouds);
        mix(&mut c.coverage, t.coverage, k);
        mix(&mut c.base, t.base, k);
        mix(&mut c.thickness, t.thickness, k);
        mix(&mut c.density, t.density, k);
        let (l, t) = (&mut self.light, &to.light);
        mix(&mut l.sun, t.sun, k);
        mix(&mut l.moon, t.moon, k);
        mix(&mut l.sky, t.sky, k);
        mix(&mut l.block, t.block, k);
        mix(&mut l.cave, t.cave, k);
        // the set of particle types switches at once (the engine fades each type's share)
        self.particles.types.clone_from(&to.particles.types);
        mix(&mut self.particles.density, to.particles.density, k);
        let (w, t) = (&mut self.weather, &to.weather);
        mix(&mut w.cycle, t.cycle, k);
        mix(&mut w.storminess, t.storminess, k);
        mix(&mut w.peak, t.peak, k);
        mix(&mut w.wind, t.wind, k);
        mix(&mut w.storm_wind, t.storm_wind, k);
        mix(&mut w.fog, t.fog, k);
        mix3(&mut w.fog_color, &t.fog_color, k);
        mix(&mut w.fog_tint, t.fog_tint, k);
        mix(&mut w.clouds, t.clouds, k);
        mix(&mut w.darken, t.darken, k);
        mix(&mut w.lightning, t.lightning, k);
    }

    /// Override the groups `o` sets.
    pub fn apply(&mut self, o: &EnvOverride) {
        if let Some(s) = &o.sky {
            self.sky = s.clone();
        }
        if let Some(f) = &o.fog {
            self.fog = f.clone();
        }
        if let Some(g) = &o.grade {
            self.grade = g.clone();
        }
        if let Some(c) = &o.clouds {
            self.clouds = c.clone();
        }
        if let Some(l) = &o.light {
            self.light = l.clone();
        }
        if let Some(p) = &o.particles {
            self.particles = p.clone();
        }
        if let Some(w) = &o.weather {
            self.weather = w.clone();
        }
    }
}

impl EnvOverride {
    /// Later groups win.
    fn merge(&mut self, o: EnvOverride) {
        if o.sky.is_some() {
            self.sky = o.sky;
        }
        if o.fog.is_some() {
            self.fog = o.fog;
        }
        if o.grade.is_some() {
            self.grade = o.grade;
        }
        if o.clouds.is_some() {
            self.clouds = o.clouds;
        }
        if o.light.is_some() {
            self.light = o.light;
        }
        if o.particles.is_some() {
            self.particles = o.particles;
        }
        if o.weather.is_some() {
            self.weather = o.weather;
        }
    }
    pub fn is_empty(&self) -> bool {
        *self == EnvOverride::default()
    }
}

/// Wire environment part `part` into input `pin` of node `out`: directly when the input is
/// free, else into the Environment node already there (or a new one combining both). Later
/// inputs win, so `part` overrides what was there.
pub fn attach_part(g: &mut Graph, out: u32, pin: u8, part: u32) {
    let Some(l) = g.input_link(out, pin).copied() else {
        g.connect(part, 0, out, pin);
        return;
    };
    let combine = match g.node(l.from_node) {
        Some(n) if n.kind == "environment" => l.from_node,
        _ => {
            let env = g.add("environment", [0.0, 0.0]);
            g.connect(l.from_node, l.from_pin, env, 0);
            g.connect(env, 0, out, pin);
            env
        }
    };
    if let Some(free) = (0..6u8).find(|&p| g.input_link(combine, p).is_none()) {
        g.connect(part, 0, combine, free);
    }
}

/// Biome Outputs once had their own fog, grading and particle settings (fog_color, fog_density
/// as a multiple of the default, grade_tint, saturation, contrast; ambient, precipitation,
/// particle_density): move them into Fog, Grading and Particles nodes on the Environment input.
/// Returns whether the graph changed.
pub fn migrate_biome_output(g: &mut Graph) -> bool {
    if g.kind != crate::graph::GraphKind::Biome {
        return false;
    }
    let Some(o) = g.output_node().map(|n| n.id) else { return false };
    let params = &mut g.node_mut(o).unwrap().params;
    let mut take = |k: &str| params.remove(k);
    let (color, density) = (take("fog_color"), take("fog_density"));
    let (tint, saturation, contrast) = (take("grade_tint"), take("saturation"), take("contrast"));
    let (ambient, precipitation, particle_density) = (take("ambient"), take("precipitation"), take("particle_density"));
    let mut parts = Vec::new();
    if color.is_some() || density.is_some() {
        let f = g.add("env_fog", [0.0, 0.0]);
        if let Some(c) = color {
            g.set(f, "color", c);
        }
        g.set(f, "density", Value::F(0.0012 * density.map_or(1.0, |d| d.as_f32())));
        parts.push(f);
    }
    if tint.is_some() || saturation.is_some() || contrast.is_some() {
        let e = g.add("env_grading", [0.0, 0.0]);
        for (k, v) in [("tint", tint), ("saturation", saturation), ("contrast", contrast)] {
            if let Some(v) = v {
                g.set(e, k, v);
            }
        }
        parts.push(e);
    }
    // the old fixed particle kinds are default Particle graphs now
    let name = |v: &Option<Value>| match v {
        Some(Value::E(s)) => s.clone(),
        _ => String::new(),
    };
    let mut types: Vec<&str> = match name(&ambient).as_str() {
        "Ash" => vec!["ash"],
        "Embers" => vec!["embers", "ash"],
        "Fireflies" => vec!["fireflies"],
        "Pollen" => vec!["pollen"],
        _ => vec![],
    };
    match name(&precipitation).as_str() {
        "Rain" => types.push("rain"),
        "Snow" => types.push("snow"),
        _ => {}
    }
    let changed = !parts.is_empty() || ambient.is_some() || precipitation.is_some() || particle_density.is_some();
    if !types.is_empty() {
        let p = g.add("env_particles", [0.0, 0.0]);
        for (i, t) in types.iter().enumerate() {
            g.set(p, &format!("type{}", i + 1), Value::Ref(t.to_string()));
        }
        if let Some(d) = particle_density {
            g.set(p, "density", d);
        }
        parts.push(p);
    }
    for p in parts {
        attach_part(g, o, BIOME_ENV_PIN, p);
    }
    changed
}

/// The Environment inputs of the Biome, Zone and World Outputs
pub const BIOME_ENV_PIN: u8 = 13;
pub const ZONE_ENV_PIN: u8 = 3;
pub const WORLD_ENV_PIN: u8 = 4;

fn value(n: &Node, name: &str) -> Value {
    n.params
        .get(name)
        .cloned()
        .or_else(|| nodes::def(&n.kind).and_then(|d| d.params.iter().find(|p| p.name == name)).map(|p| p.kind.default_value()))
        .unwrap_or(Value::F(0.0))
}
fn f(n: &Node, name: &str) -> f32 {
    value(n, name).as_f32()
}
fn c(n: &Node, name: &str) -> Rgb {
    match value(n, name) {
        Value::Color(c) => c,
        _ => [1.0; 3],
    }
}

/// The environment wired into input `pin` of node `node` in graph `g`.
pub fn resolve(g: &Graph, node: u32, pin: u8) -> EnvOverride {
    match g.input_link(node, pin) {
        Some(l) => resolve_node(g, l.from_node, 0),
        None => EnvOverride::default(),
    }
}

fn resolve_node(g: &Graph, id: u32, depth: u32) -> EnvOverride {
    let mut o = EnvOverride::default();
    let Some(n) = g.node(id) else { return o };
    if depth > 32 {
        return o;
    }
    match n.kind.as_str() {
        "environment" => {
            for pin in 0..6u8 {
                if let Some(l) = g.input_link(id, pin) {
                    o.merge(resolve_node(g, l.from_node, depth + 1));
                }
            }
        }
        "env_sky" => {
            o.sky = Some(SkyEnv {
                zenith: [c(n, "zenith_day"), c(n, "zenith_dusk"), c(n, "zenith_night")],
                horizon: [c(n, "horizon_day"), c(n, "horizon_dusk"), c(n, "horizon_night")],
                ground: [c(n, "ground_day"), c(n, "ground_night")],
                sun: c(n, "sun"),
                sun_dusk: c(n, "sun_dusk"),
                sun_size: f(n, "sun_size"),
                moon: c(n, "moon"),
                moon_size: f(n, "moon_size"),
                stars: f(n, "stars"),
            })
        }
        "env_fog" => o.fog = Some(FogEnv { color: c(n, "color"), strength: f(n, "strength"), density: f(n, "density"), falloff: f(n, "height_falloff"), start: f(n, "start") }),
        "env_grading" => o.grade = Some(GradeEnv { tint: c(n, "tint"), saturation: f(n, "saturation"), contrast: f(n, "contrast"), exposure: f(n, "exposure") }),
        "env_clouds" => o.clouds = Some(CloudEnv { coverage: f(n, "coverage"), base: f(n, "base"), thickness: f(n, "thickness"), density: f(n, "density") }),
        "env_light" => o.light = Some(LightEnv { sun: f(n, "sun"), moon: f(n, "moon"), sky: f(n, "sky"), block: f(n, "block"), cave: f(n, "cave") }),
        "env_particles" => {
            let types = (1..=6)
                .filter_map(|i| match n.params.get(&format!("type{i}")) {
                    Some(Value::Ref(r)) if !r.is_empty() => Some(r.clone()),
                    _ => None,
                })
                .collect();
            o.particles = Some(ParticlesEnv { types, density: f(n, "density") })
        }
        "env_weather" => {
            o.weather = Some(WeatherEnv {
                cycle: f(n, "cycle"),
                storminess: f(n, "storminess"),
                peak: f(n, "peak"),
                wind: f(n, "wind"),
                storm_wind: f(n, "storm_wind"),
                fog: f(n, "fog"),
                fog_color: c(n, "fog_color"),
                fog_tint: f(n, "fog_tint"),
                clouds: f(n, "clouds"),
                darken: f(n, "darken"),
                lightning: f(n, "lightning"),
            })
        }
        _ => {}
    }
    o
}

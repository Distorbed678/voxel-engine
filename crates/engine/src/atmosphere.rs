//! Time-of-day and the environment: sun direction, sky/fog colours and light levels fed to the
//! shaders. The environment (`worldgen::env`) comes from the world's graphs; `env` blends toward
//! the camera's (world, layer, zone, biome) over a second or two.

use glam::Vec3;
use worldgen::env::Env;

#[derive(Clone, Debug)]
pub struct Atmosphere {
    /// 0 = midnight, 0.25 = sunrise, 0.5 = noon, 0.75 = sunset
    pub time_of_day: f32,
    /// seconds per day (the World Output's Day Length)
    pub day_length_s: f32,
    pub paused: bool,
    /// the camera's environment, blended
    pub env: Env,
    /// fog density after the fog slider and the weather
    pub fog_density: f32,
    /// the height the fog thins from (the camera layer's sea level, or its floor)
    pub fog_base: f32,
    /// the exposure slider (times the environment's)
    pub exposure: f32,
    /// Weather intensity 0..1 (rain, heavier snow, more clouds and fog).
    pub weather: f32,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self {
            time_of_day: 0.36,
            day_length_s: 600.0,
            paused: true,
            env: Env::default(),
            fog_density: 0.0012,
            fog_base: 62.0,
            exposure: 0.75,
            weather: 0.0,
        }
    }
}

pub struct SkyColors {
    pub sun_dir: Vec3,
    pub sun_intensity: f32,
    pub sun_color: Vec3,
    pub zenith: Vec3,
    pub horizon: Vec3,
    pub ground: Vec3,
    pub fog: Vec3,
}

fn lerp3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// The default moonlight, relative to the default moon colour
const MOONLIGHT: Vec3 = Vec3::new(0.45 / 0.8, 0.55 / 0.85, 0.85);

impl Atmosphere {
    pub fn update(&mut self, dt: f32) {
        if !self.paused {
            self.time_of_day = (self.time_of_day + dt / self.day_length_s.max(1.0)).fract();
        }
    }

    pub fn colors(&self) -> SkyColors {
        let (sky, light) = (&self.env.sky, &self.env.light);
        let angle = (self.time_of_day - 0.25) * std::f32::consts::TAU;
        let sun_dir = Vec3::new(angle.cos(), angle.sin(), 0.35).normalize();
        let e = sun_dir.y; // elevation proxy
        let day = ((e + 0.05) / 0.35).clamp(0.0, 1.0);
        let dusk = (1.0 - (e.abs() / 0.25)).clamp(0.0, 1.0);

        let [zen_day, zen_dusk, zen_night] = sky.zenith.map(Vec3::from);
        let [hor_day, hor_dusk, hor_night] = sky.horizon.map(Vec3::from);
        let mut zenith = lerp3(zen_night, zen_day, day);
        let mut horizon = lerp3(hor_night, hor_day, day);
        zenith = lerp3(zenith, zen_dusk, dusk * 0.6);
        horizon = lerp3(horizon, hor_dusk, dusk * 0.75);

        let (sun_color, sun_intensity) = if e > -0.05 {
            let k = (e / 0.3).clamp(0.0, 1.0);
            (lerp3(Vec3::from(sky.sun_dusk), Vec3::from(sky.sun), k), 1.9 * ((e + 0.05) / 0.15).clamp(0.0, 1.0) * light.sun)
        } else {
            // moonlight (lighting shader flips to -sun_dir when the sun is down)
            (Vec3::from(sky.moon) * MOONLIGHT, 0.35 * ((-e - 0.05) / 0.15).clamp(0.0, 1.0) * light.moon)
        };
        let ground = lerp3(Vec3::from(sky.ground[1]), Vec3::from(sky.ground[0]), day);
        // the fog colour dims at night, where the sun lights it
        let dim = 1.0 + (0.2 + 0.8 * day - 1.0) * light.sun.min(1.0);
        let fog = lerp3(horizon, Vec3::from(self.env.fog.color) * dim, self.env.fog.strength);
        SkyColors { sun_dir, sun_intensity, sun_color, zenith, horizon, ground, fog }
    }
}

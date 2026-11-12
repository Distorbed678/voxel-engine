//! Time-of-day: sun direction and sky/fog colours fed to the shaders.

use glam::Vec3;

#[derive(Clone, Debug)]
pub struct Atmosphere {
    /// 0 = midnight, 0.25 = sunrise, 0.5 = noon, 0.75 = sunset
    pub time_of_day: f32,
    pub day_length_s: f32,
    pub paused: bool,
    pub fog_density: f32,
    pub fog_height_falloff: f32,
    pub fog_start: f32,
    pub exposure: f32,
    pub sea_level: f32,
    /// biome-driven fog tint (blended in by the environment system)
    pub fog_tint: Vec3,
    pub fog_tint_strength: f32,
    /// Colour grading (blended towards the camera's biome): tint, saturation, contrast.
    pub grade_tint: Vec3,
    /// Weather intensity 0..1 (rain, heavier snow, more clouds and fog).
    pub weather: f32,
    pub saturation: f32,
    pub contrast: f32,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self {
            time_of_day: 0.36,
            day_length_s: 600.0,
            paused: true,
            fog_density: 0.0012,
            fog_height_falloff: 0.012,
            fog_start: 32.0,
            exposure: 0.75,
            sea_level: 62.0,
            fog_tint: Vec3::new(0.75, 0.82, 0.95),
            fog_tint_strength: 0.0,
            grade_tint: Vec3::ONE,
            weather: 0.0,
            saturation: 1.0,
            contrast: 1.0,
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

impl Atmosphere {
    pub fn update(&mut self, dt: f32) {
        if !self.paused {
            self.time_of_day = (self.time_of_day + dt / self.day_length_s).fract();
        }
    }

    pub fn colors(&self) -> SkyColors {
        let angle = (self.time_of_day - 0.25) * std::f32::consts::TAU;
        let sun_dir = Vec3::new(angle.cos(), angle.sin(), 0.35).normalize();
        let e = sun_dir.y; // elevation proxy
        let day = ((e + 0.05) / 0.35).clamp(0.0, 1.0);
        let dusk = (1.0 - (e.abs() / 0.25)).clamp(0.0, 1.0);

        let zen_day = Vec3::new(0.22, 0.45, 0.92);
        let hor_day = Vec3::new(0.62, 0.78, 0.98);
        let zen_night = Vec3::new(0.006, 0.01, 0.03);
        let hor_night = Vec3::new(0.02, 0.03, 0.06);
        let hor_dusk = Vec3::new(1.0, 0.5, 0.25);
        let zen_dusk = Vec3::new(0.25, 0.3, 0.55);

        let mut zenith = lerp3(zen_night, zen_day, day);
        let mut horizon = lerp3(hor_night, hor_day, day);
        zenith = lerp3(zenith, zen_dusk, dusk * 0.6);
        horizon = lerp3(horizon, hor_dusk, dusk * 0.75);

        let (sun_color, sun_intensity) = if e > -0.05 {
            let warm = Vec3::new(1.0, 0.55, 0.3);
            let white = Vec3::new(1.0, 0.96, 0.9);
            (lerp3(warm, white, (e / 0.3).clamp(0.0, 1.0)), 1.9 * ((e + 0.05) / 0.15).clamp(0.0, 1.0))
        } else {
            // moonlight (lighting shader flips to -sun_dir when the sun is down)
            (Vec3::new(0.45, 0.55, 0.85), 0.35 * ((-e - 0.05) / 0.15).clamp(0.0, 1.0))
        };
        let ground = lerp3(Vec3::new(0.02, 0.02, 0.025), Vec3::new(0.32, 0.28, 0.22), day);
        let mut fog = horizon;
        fog = lerp3(fog, self.fog_tint * (0.2 + 0.8 * day), self.fog_tint_strength);
        SkyColors { sun_dir, sun_intensity, sun_color, zenith, horizon, ground, fog }
    }
}

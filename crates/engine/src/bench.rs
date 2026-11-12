//! Benchmark mode: a scripted camera flight that measures frame times, GPU pass timings and
//! chunk generation throughput while the world streams in at flying speed.

use std::time::Instant;

use glam::DVec3;

use crate::camera::Camera;

pub struct Benchmark {
    start: Instant,
    pub duration: f32,
    origin: DVec3,
    frame_ms: Vec<f32>,
    gpu: std::collections::BTreeMap<&'static str, (f64, u32)>,
    jobs_start: u64,
    pub report: Option<String>,
}

impl Benchmark {
    pub fn new(origin: DVec3, jobs_start: u64) -> Self {
        Self {
            start: Instant::now(),
            duration: 24.0,
            origin,
            frame_ms: Vec::new(),
            gpu: Default::default(),
            jobs_start,
            report: None,
        }
    }

    /// Advance the scripted camera. Returns false when finished.
    pub fn update(&mut self, cam: &mut Camera, dt: f32, gpu: &[(&'static str, f32)]) -> bool {
        let t = self.start.elapsed().as_secs_f32();
        if t > self.duration {
            return false;
        }
        self.frame_ms.push(dt * 1000.0);
        for (n, ms) in gpu {
            let e = self.gpu.entry(n).or_insert((0.0, 0));
            e.0 += *ms as f64;
            e.1 += 1;
        }
        // phase 1: fast straight flight (streaming stress), phase 2: wide orbit, phase 3: climb
        let speed = 90.0;
        let p = if t < 10.0 {
            self.origin + DVec3::new(0.0, 60.0, -(t as f64) * speed)
        } else if t < 18.0 {
            let a = (t - 10.0) as f64 * 0.6;
            let c = self.origin + DVec3::new(0.0, 60.0, -900.0);
            c + DVec3::new(a.sin() * 300.0, 0.0, a.cos() * 300.0)
        } else {
            self.origin + DVec3::new(0.0, 60.0 + (t as f64 - 18.0) * 80.0, -900.0)
        };
        cam.pos = p;
        cam.yaw = if t < 10.0 { 0.0 } else { (t - 10.0) * 0.6 + std::f32::consts::FRAC_PI_2 };
        cam.pitch = if t < 18.0 { -0.25 } else { -0.6 };
        true
    }

    pub fn finish(&mut self, jobs_total: u64, quads: u64, device: &str) -> String {
        let mut v = self.frame_ms.clone();
        v.sort_by(|a, b| a.total_cmp(b));
        let n = v.len().max(1);
        let avg = v.iter().sum::<f32>() / n as f32;
        let p99 = v[((n as f32 * 0.99) as usize).min(n - 1)];
        let p1_low_fps = 1000.0 / p99.max(0.001);
        let jobs = jobs_total - self.jobs_start;
        let mut r = String::new();
        r.push_str(&format!("Voxel engine benchmark on {device}\n"));
        r.push_str(&format!("frames {n}, avg {:.2} ms ({:.0} fps), 1% low {:.0} fps, worst {:.2} ms\n", avg, 1000.0 / avg, p1_low_fps, v[n - 1]));
        r.push_str(&format!("chunks generated+meshed: {jobs} ({:.0} / s), quads resident {:.1}M\n", jobs as f32 / self.duration, quads as f64 / 1e6));
        r.push_str("GPU passes (avg ms per frame where active):\n");
        for (k, (s, c)) in &self.gpu {
            r.push_str(&format!("  {k:<12} {:.3} ms\n", s / *c as f64));
        }
        self.report = Some(r.clone());
        r
    }
}

//! GPU biome probes: evaluate the generator's zone/biome layout at the camera (environment
//! blending) and on a grid around it (minimap). Results arrive one frame-in-flight later.

use anyhow::Result;
use glam::DVec3;
use vkcore::{Buffer, VkContext};

pub const MAP_N: usize = 64;
const MAX_PROBES: usize = 1 + MAP_N * MAP_N;

#[derive(Clone, Copy, Debug, Default)]
pub struct ProbeResult {
    pub biome: u32,
    pub zone: u32,
    pub height: f32,
}

/// (origin x, origin z, spacing) of a MAP_N x MAP_N probe grid, and whether it is a survey.
type Grid = ((f64, f64, f64), bool);

pub struct Probes {
    pub input: Vec<Buffer>,
    pub output: Vec<Buffer>,
    counts: Vec<u32>,
    grids: Vec<Option<Grid>>,
    pub camera: Option<ProbeResult>,
    /// latest minimap: (origin x, origin z, spacing) + cells (row-major, z rows)
    pub map: Option<((f64, f64, f64), Vec<ProbeResult>)>,
    pub spacing: f64,
    /// Survey request (center x, center z, spacing): one wide grid evaluated on the next frame
    /// instead of the minimap (used by scripts to find biomes).
    pub survey_request: Option<(f64, f64, f64)>,
    /// Latest survey result, same layout as `map`.
    pub survey: Option<((f64, f64, f64), Vec<ProbeResult>)>,
}

impl Probes {
    pub fn new(ctx: &VkContext, frames: usize) -> Result<Self> {
        let mut input = Vec::new();
        let mut output = Vec::new();
        for i in 0..frames {
            input.push(Buffer::upload(ctx, &format!("probe_in_{i}"), (MAX_PROBES * 8) as u64)?);
            output.push(Buffer::readback(ctx, &format!("probe_out_{i}"), (MAX_PROBES * 16) as u64)?);
        }
        Ok(Self { input, output, counts: vec![0; frames], grids: vec![None; frames], camera: None, map: None, spacing: 24.0, survey_request: None, survey: None })
    }

    /// Fill this slot's probe list. Returns the number of probes to dispatch.
    pub fn prepare(&mut self, slot: usize, cam: DVec3, frame: u64) -> u32 {
        let mut pts: Vec<[f32; 2]> = Vec::with_capacity(MAX_PROBES);
        pts.push([cam.x as f32, cam.z as f32]);
        self.grids[slot] = None;
        let grid = if let Some((cx, cz, s)) = self.survey_request.take() {
            let half = (MAP_N as f64) * 0.5 * s;
            Some(((cx - half, cz - half, s), true))
        } else if frame.is_multiple_of(20) {
            let s = self.spacing;
            let half = (MAP_N as f64) * 0.5 * s;
            Some(((((cam.x - half) / s).floor() * s, ((cam.z - half) / s).floor() * s, s), false))
        } else {
            None
        };
        if let Some(((ox, oz, s), _)) = grid {
            for j in 0..MAP_N {
                for i in 0..MAP_N {
                    pts.push([(ox + (i as f64 + 0.5) * s) as f32, (oz + (j as f64 + 0.5) * s) as f32]);
                }
            }
            self.grids[slot] = grid;
        }
        self.input[slot].write(0, &pts);
        self.counts[slot] = pts.len() as u32;
        self.counts[slot]
    }

    pub fn read(&mut self, slot: usize) {
        let n = self.counts[slot] as usize;
        if n == 0 {
            return;
        }
        let raw: Vec<[u32; 4]> = self.output[slot].read(0, n);
        let conv = |r: [u32; 4]| ProbeResult {
            biome: r[0] & 0xFFFF,
            zone: r[0] >> 16,
            height: f32::from_bits(r[3]),
        };
        self.camera = Some(conv(raw[0]));
        if let Some((g, survey)) = self.grids[slot]
            && n > 1 {
                let cells = Some((g, raw[1..].iter().map(|r| conv(*r)).collect()));
                if survey {
                    self.survey = cells;
                } else {
                    self.map = cells;
                }
            }
    }

    pub fn height_at(&self, x: f64, z: f64) -> Option<f32> {
        let ((ox, oz, s), cells) = self.map.as_ref()?;
        let i = ((x - ox) / s).floor() as i64;
        let j = ((z - oz) / s).floor() as i64;
        if i < 0 || j < 0 || i >= MAP_N as i64 || j >= MAP_N as i64 {
            return None;
        }
        let h = cells[(j as usize) * MAP_N + i as usize].height;
        (h > -1000.0).then_some(h)
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        for b in self.input.iter_mut().chain(self.output.iter_mut()) {
            b.destroy(ctx);
        }
    }
}

//! Prop editor preview: one instance of the open Prop graph, generated on the GPU by the real
//! props pass into empty space (preview jobs, `GENJOB_PREVIEW`), read back, and drawn in an ImGui
//! window as textured faces (the block atlas) with an orbit camera, the prop's search box and the
//! ground plane. Re-roll picks another instance; "Place in world" puts it in front of the camera
//! as an edit.

use dear_imgui_rs::{Condition, MouseButton, Ui};
use glam::{IVec3, Mat3, Vec3};

use crate::blockgpu::AtlasRef;
use crate::blocks::{BlockKind, BlockRegistry};
use crate::world::{RegionRead, World};

struct Face {
    corners: [Vec3; 4],
    centre: Vec3,
    normal: Vec3,
    layer: u16,
    shade: f32,
}

pub struct PropPreview {
    pub open: bool,
    /// the prop graph shown (name) and its index in the compiled generator
    pub graph: Option<(String, u32)>,
    pub seed: u32,
    /// the search box of the prop (radius, height, below)
    pub bounds: [f32; 3],
    pending: Option<u64>,
    /// a new instance is needed (graph, seed or parameters changed)
    pub stale: bool,
    /// the last instance: min corner, size, blocks
    pub result: Option<(IVec3, IVec3, Vec<u16>)>,
    faces: Vec<Face>,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    /// place the instance in front of the camera (handled by the engine)
    pub place: bool,
    cells: usize,
}

impl Default for PropPreview {
    fn default() -> Self {
        Self::new()
    }
}

const DIRS: [IVec3; 6] = [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z];
const SHADE: [f32; 6] = [0.8, 0.8, 1.0, 0.5, 0.65, 0.65];

impl PropPreview {
    pub fn new() -> Self {
        Self {
            open: false,
            graph: None,
            seed: 1,
            bounds: [6.0, 12.0, 2.0],
            pending: None,
            stale: true,
            result: None,
            faces: Vec::new(),
            yaw: 0.7,
            pitch: 0.45,
            zoom: 1.0,
            place: false,
            cells: 0,
        }
    }

    /// Show `graph` (index `idx`); a different graph means a new instance.
    pub fn show(&mut self, name: &str, idx: u32, bounds: [f32; 3]) {
        if self.graph.as_ref().map(|g| (g.0.as_str(), g.1)) != Some((name, idx)) || self.bounds != bounds {
            self.graph = Some((name.to_string(), idx));
            self.bounds = bounds;
            self.stale = true;
        }
    }

    /// The box read back around the origin.
    fn read_box(&self) -> (IVec3, IVec3) {
        let [r, h, below] = self.bounds.map(|v| v.clamp(1.0, 96.0).ceil() as i32);
        (IVec3::new(-r - 1, -below - 1, -r - 1), IVec3::new(r + 1, h + 1, r + 1))
    }

    /// Request a new instance if needed (one at a time).
    pub fn update(&mut self, world: &mut World) {
        if !self.open || !self.stale || self.pending.is_some() {
            return;
        }
        let Some((_, idx)) = self.graph else { return };
        let (a, b) = self.read_box();
        self.pending = Some(world.request_prop_preview(idx, self.seed, a, b));
        self.stale = false;
    }

    /// A finished read: ours?
    pub fn take(&mut self, r: RegionRead, blocks: &BlockRegistry) -> Option<RegionRead> {
        if self.pending != Some(r.id) {
            return Some(r);
        }
        self.pending = None;
        self.mesh(&r, blocks);
        self.result = Some((r.min, r.size, r.blocks));
        None
    }

    fn mesh(&mut self, r: &RegionRead, blocks: &BlockRegistry) {
        self.faces.clear();
        self.cells = 0;
        let solid = |v: u16| v != 0 && matches!(blocks.get(v).kind, BlockKind::Opaque | BlockKind::Transparent);
        for y in 0..r.size.y {
            for z in 0..r.size.z {
                for x in 0..r.size.x {
                    let p = r.min + IVec3::new(x, y, z);
                    let v = r.get(p);
                    if v == 0 {
                        continue;
                    }
                    self.cells += 1;
                    let def = blocks.get(v);
                    for (d, dir) in DIRS.iter().enumerate() {
                        if solid(r.get(p + *dir)) && def.kind == BlockKind::Opaque {
                            continue;
                        }
                        if r.get(p + *dir) == v {
                            continue;
                        }
                        let Some(&slot) = blocks.texture_index.get(&def.faces[d]) else { continue };
                        let layer = blocks.textures[slot].base;
                        // the face's square, corners ordered top-left, top-right, bottom-right, bottom-left
                        let n = dir.as_vec3();
                        let c = p.as_vec3() + Vec3::splat(0.5) + n * 0.5;
                        let (u, w) = if d < 2 {
                            (Vec3::Z * n.x, Vec3::Y)
                        } else if d < 4 {
                            (Vec3::X, Vec3::Z * n.y)
                        } else {
                            (Vec3::X * -n.z, Vec3::Y)
                        };
                        let corners = [c - u * 0.5 + w * 0.5, c + u * 0.5 + w * 0.5, c + u * 0.5 - w * 0.5, c - u * 0.5 - w * 0.5];
                        self.faces.push(Face { corners, centre: c, normal: n, layer, shade: SHADE[d] });
                        if self.faces.len() > 60_000 {
                            return;
                        }
                    }
                }
            }
        }
    }

    /// The last instance as a clipboard, cropped to its blocks; pasting puts its anchor (the first
    /// air cell above the ground it grew on) where the paste goes.
    pub fn to_clipboard(&self) -> Option<crate::prefab::Prefab> {
        let (min, size, blocks) = self.result.as_ref()?;
        let at = |p: IVec3| blocks[((p.y * size.z + p.z) * size.x + p.x) as usize];
        let (mut lo, mut hi) = (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN));
        for y in 0..size.y {
            for z in 0..size.z {
                for x in 0..size.x {
                    let p = IVec3::new(x, y, z);
                    if at(p) != 0 {
                        lo = lo.min(p);
                        hi = hi.max(p);
                    }
                }
            }
        }
        if lo.x > hi.x {
            return None;
        }
        let ns = hi - lo + IVec3::ONE;
        let mut out = Vec::with_capacity((ns.x * ns.y * ns.z) as usize);
        for y in 0..ns.y {
            for z in 0..ns.z {
                for x in 0..ns.x {
                    out.push(at(lo + IVec3::new(x, y, z)));
                }
            }
        }
        let name = self.graph.as_ref().map(|g| g.0.clone()).unwrap_or_default();
        Some(crate::prefab::Prefab { name, size: ns, blocks: out, origin: -(*min + lo) })
    }

    pub fn draw(&mut self, ui: &Ui, atlas: Option<AtlasRef>) {
        if !self.open {
            return;
        }
        let ds = ui.io().display_size();
        let mut open = true;
        ui.window("Prop Preview")
            .size([420.0, 480.0], Condition::FirstUseEver)
            .position([ds[0] - 430.0, (ds[1] - 520.0).max(30.0)], Condition::FirstUseEver)
            .opened(&mut open)
            .build(|| {
                let name = self.graph.as_ref().map(|g| g.0.clone()).unwrap_or_default();
                ui.text(format!("{name}   seed {}", self.seed));
                ui.same_line();
                if ui.small_button("Re-roll") {
                    self.seed = self.seed.wrapping_mul(747796405).wrapping_add(2891336453) >> 1;
                    self.stale = true;
                }
                ui.same_line();
                if ui.small_button("Place in world") {
                    self.place = true;
                }
                if self.pending.is_some() {
                    ui.same_line();
                    ui.text_disabled("building...");
                }
                ui.text_disabled(format!("{} blocks, box r {:.0} h {:.0} below {:.0}   drag: orbit, wheel: zoom", self.cells, self.bounds[0], self.bounds[1], self.bounds[2]));
                let avail = ui.content_region_avail();
                let size = [avail[0].max(64.0), avail[1].max(64.0)];
                let p0 = ui.cursor_screen_pos();
                ui.invisible_button("##prop_view", size);
                if ui.is_item_active() && ui.is_mouse_dragging(MouseButton::Left) {
                    let d = ui.io().mouse_delta();
                    self.yaw -= d[0] * 0.01;
                    self.pitch = (self.pitch + d[1] * 0.01).clamp(-1.4, 1.4);
                }
                if ui.is_item_hovered() {
                    let w = ui.io().mouse_wheel();
                    if w != 0.0 {
                        self.zoom = (self.zoom * (1.0 + w * 0.1)).clamp(0.2, 8.0);
                    }
                }
                self.view(ui, atlas, p0, size);
            });
        self.open = open;
    }

    /// Orbit view: perspective projection of the faces, far ones first (painter's algorithm).
    fn view(&self, ui: &Ui, atlas: Option<AtlasRef>, p0: [f32; 2], size: [f32; 2]) {
        let dl = ui.get_window_draw_list();
        dl.add_rect(p0, [p0[0] + size[0], p0[1] + size[1]], [0.16, 0.2, 0.26, 1.0]).filled(true).build();
        let [r, h, below] = self.bounds;
        let target = Vec3::new(0.5, (h - below) * 0.4, 0.5);
        let extent = r.max(h * 0.6).max(2.0);
        let dist = extent * 2.6 / self.zoom;
        let rot = Mat3::from_rotation_y(self.yaw) * Mat3::from_rotation_x(-self.pitch);
        let eye = target + rot * Vec3::new(0.0, 0.0, dist);
        let fwd = (target - eye).normalize();
        let right = fwd.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(fwd);
        let f = size[1].min(size[0]) * 0.9;
        let centre = [p0[0] + size[0] * 0.5, p0[1] + size[1] * 0.5];
        let project = |p: Vec3| -> Option<[f32; 2]> {
            let v = p - eye;
            let z = v.dot(fwd);
            if z < 0.1 {
                return None;
            }
            Some([centre[0] + v.dot(right) / z * f, centre[1] - v.dot(up) / z * f])
        };
        // ground grid and the prop's box
        let gr = r.ceil() + 1.0;
        for i in -(gr as i32)..=(gr as i32) {
            let t = i as f32 + 0.5;
            for (a, b) in [(Vec3::new(t, 0.0, -gr + 0.5), Vec3::new(t, 0.0, gr + 0.5)), (Vec3::new(-gr + 0.5, 0.0, t), Vec3::new(gr + 0.5, 0.0, t))] {
                if let (Some(pa), Some(pb)) = (project(a), project(b)) {
                    dl.add_line(pa, pb, [0.5, 0.55, 0.6, 0.35]).build();
                }
            }
        }
        let c = |x: f32, y: f32, z: f32| Vec3::new(x + 0.5, y, z + 0.5);
        let corners = [c(-r, -below, -r), c(r, -below, -r), c(r, -below, r), c(-r, -below, r), c(-r, h, -r), c(r, h, -r), c(r, h, r), c(-r, h, r)];
        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)] {
            if let (Some(pa), Some(pb)) = (project(corners[a]), project(corners[b])) {
                dl.add_line(pa, pb, [1.0, 0.8, 0.3, 0.5]).build();
            }
        }
        // the faces turned towards the eye, far to near
        let mut order: Vec<(f32, usize)> = self
            .faces
            .iter()
            .enumerate()
            .filter(|(_, fc)| fc.normal.dot(eye - fc.centre) > 0.0)
            .map(|(i, fc)| ((fc.centre - eye).length_squared(), i))
            .collect();
        order.sort_by(|a, b| b.0.total_cmp(&a.0));
        dl.set_sampler_nearest();
        for (_, i) in order {
            let fc = &self.faces[i];
            let pts: Vec<[f32; 2]> = fc.corners.iter().filter_map(|&p| project(p)).collect();
            if pts.len() != 4 {
                continue;
            }
            let k = fc.shade;
            match atlas {
                Some(a) => {
                    let (uv0, uv1) = a.uv(fc.layer);
                    dl.add_image_quad(a.id, pts[0], pts[1], pts[2], pts[3], uv0, [uv1[0], uv0[1]], uv1, [uv0[0], uv1[1]], [k, k, k, 1.0]);
                }
                None => dl.add_quad_filled(pts[0], pts[1], pts[2], pts[3], [0.5 * k, 0.6 * k, 0.4 * k, 1.0]),
            }
        }
        dl.set_sampler_linear();
    }
}

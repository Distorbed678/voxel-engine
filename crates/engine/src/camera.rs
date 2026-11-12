use glam::{DVec3, IVec3, Mat4, Vec3, Vec4};

/// Free-flying camera. Position is f64 so the world can be huge; rendering is camera-relative.
#[derive(Clone, Debug)]
pub struct Camera {
    pub pos: DVec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_deg: f32,
    pub speed: f32,
    pub near: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self { pos: DVec3::new(0.5, 120.0, 0.5), yaw: 0.6, pitch: -0.25, fov_y_deg: 75.0, speed: 30.0, near: 0.05 }
    }
}

pub struct CameraMatrices {
    pub view: Mat4,
    pub proj: Mat4,
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub frustum: [Vec4; 6],
    pub cam_block: IVec3,
    pub cam_frac: Vec3,
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(-sy * cp, sp, -cy * cp)
    }

    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy, 0.0, -sy)
    }

    pub fn matrices(&self, aspect: f32) -> CameraMatrices {
        let fwd = self.forward();
        let view = look_to_rh(fwd, Vec3::Y);
        let proj = perspective_infinite_reverse_rh(self.fov_y_deg.to_radians(), aspect, self.near);
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();
        let cam_block = self.pos.floor().as_ivec3();
        let cam_frac = (self.pos - self.pos.floor()).as_vec3();
        CameraMatrices { view, proj, view_proj, inv_view_proj, frustum: frustum_planes(&view_proj), cam_block, cam_frac }
    }
}

/// Right-handed view matrix at the origin looking along `dir`.
pub fn look_to_rh(dir: Vec3, up: Vec3) -> Mat4 {
    let f = dir.normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    Mat4::from_cols(
        Vec4::new(s.x, u.x, -f.x, 0.0),
        Vec4::new(s.y, u.y, -f.y, 0.0),
        Vec4::new(s.z, u.z, -f.z, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    )
}

/// Reversed-Z infinite perspective with Vulkan's y-down clip space (depth 1 at near, 0 at infinity).
pub fn perspective_infinite_reverse_rh(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (0.5 * fov_y).tan();
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, -f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, -1.0),
        Vec4::new(0.0, 0.0, near, 0.0),
    )
}

/// Gribb-Hartmann plane extraction (inward-facing, normalised). Clip z in [0, w].
fn frustum_planes(m: &Mat4) -> [Vec4; 6] {
    let r0 = m.row(0);
    let r1 = m.row(1);
    let r2 = m.row(2);
    let r3 = m.row(3);
    let mut planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2, r2];
    for p in planes.iter_mut() {
        let l = p.truncate().length().max(1e-6);
        *p /= l;
    }
    // r2 is the reversed-Z "far" plane at infinity: degenerate normal; disable it.
    if planes[5].truncate().length() < 0.5 {
        planes[5] = Vec4::new(0.0, 0.0, 0.0, 1.0);
    }
    planes
}

/// Fit `cascades` orthographic shadow cascades to the view frustum slices. Projections are
/// texel-snapped in world space (f64) so shadows do not shimmer when the camera moves.
pub struct Cascades {
    pub view_proj: [Mat4; 4],
    pub planes: [[Vec4; 6]; 4],
    pub splits: [f32; 4],
}

pub fn fit_cascades(cam: &Camera, aspect: f32, light_dir: Vec3, max_dist: f32, res: u32) -> Cascades {
    let splits = [max_dist * 0.06, max_dist * 0.16, max_dist * 0.42, max_dist];
    let fwd = cam.forward();
    let right = fwd.cross(Vec3::Y).normalize();
    let up = right.cross(fwd);
    let th = (cam.fov_y_deg.to_radians() * 0.5).tan();
    let l = light_dir.normalize();
    let lup = if l.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let lr = l.cross(lup).normalize();
    let lu = lr.cross(l);
    let mut out = Cascades { view_proj: [Mat4::IDENTITY; 4], planes: [[Vec4::ZERO; 6]; 4], splits };
    let mut near = cam.near;
    for c in 0..4 {
        let far = splits[c];
        let mut corners = Vec::with_capacity(8);
        for z in [near, far] {
            let h = z * th;
            let w = h * aspect;
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                corners.push(fwd * z + right * (w * sx) + up * (h * sy));
            }
        }
        let center = corners.iter().fold(Vec3::ZERO, |a, b| a + *b) / 8.0;
        let mut radius = corners.iter().map(|p| (*p - center).length()).fold(0.0f32, f32::max);
        radius = (radius * 16.0).ceil() / 16.0;
        // snap the centre to whole shadow texels along the light's right/up axes (world space)
        let texel = (2.0 * radius) as f64 / res as f64;
        let wc = cam.pos + center.as_dvec3();
        let (lrd, lud) = (lr.as_dvec3(), lu.as_dvec3());
        let a = (wc.dot(lrd) / texel).floor() * texel;
        let b = (wc.dot(lud) / texel).floor() * texel;
        let snapped = wc + lrd * (a - wc.dot(lrd)) + lud * (b - wc.dot(lud));
        let center_rel = (snapped - cam.pos).as_vec3();
        let depth_range = radius * 2.0 + 600.0;
        let eye = center_rel + l * (radius + 500.0);
        // light camera looks along -l: view-space z = l.(p - eye) (negative in front)
        let view = Mat4::from_cols(
            Vec4::new(lr.x, lu.x, l.x, 0.0),
            Vec4::new(lr.y, lu.y, l.y, 0.0),
            Vec4::new(lr.z, lu.z, l.z, 0.0),
            Vec4::new(-lr.dot(eye), -lu.dot(eye), -l.dot(eye), 1.0),
        );
        // reversed-Z orthographic: view z in [-depth_range, 0] -> depth [0, 1]
        let proj = Mat4::from_cols(
            Vec4::new(1.0 / radius, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -1.0 / radius, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0 / depth_range, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 1.0),
        );
        let vp = proj * view;
        out.view_proj[c] = vp;
        let r0 = vp.row(0);
        let r1 = vp.row(1);
        let r2 = vp.row(2);
        let r3 = vp.row(3);
        let mut planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2];
        for p in planes.iter_mut() {
            let len = p.truncate().length().max(1e-6);
            *p /= len;
        }
        out.planes[c] = planes;
        near = far;
    }
    out
}

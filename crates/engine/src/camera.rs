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

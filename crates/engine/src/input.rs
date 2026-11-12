use std::collections::HashSet;

use winit::keyboard::KeyCode;

#[derive(Default)]
pub struct Input {
    pub held: HashSet<KeyCode>,
    pub pressed: HashSet<KeyCode>,
    pub mouse_held: [bool; 3],
    pub mouse_pressed: [bool; 3],
    pub mouse_delta: (f64, f64),
    pub scroll: f32,
    pub cursor: (f64, f64),
}

impl Input {
    pub fn key(&self, k: KeyCode) -> bool {
        self.held.contains(&k)
    }
    pub fn pressed(&self, k: KeyCode) -> bool {
        self.pressed.contains(&k)
    }
    pub fn ctrl(&self) -> bool {
        self.key(KeyCode::ControlLeft) || self.key(KeyCode::ControlRight)
    }
    pub fn shift(&self) -> bool {
        self.key(KeyCode::ShiftLeft) || self.key(KeyCode::ShiftRight)
    }
    pub fn alt(&self) -> bool {
        self.key(KeyCode::AltLeft) || self.key(KeyCode::AltRight)
    }
    /// Clear per-frame edge state.
    pub fn end_frame(&mut self) {
        self.pressed.clear();
        self.mouse_pressed = [false; 3];
        self.mouse_delta = (0.0, 0.0);
        self.scroll = 0.0;
    }
}

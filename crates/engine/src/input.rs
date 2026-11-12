use std::collections::HashSet;

use winit::event::{ElementState, WindowEvent};
use winit::keyboard::KeyCode;

/// Whether a window event is passed on to ImGui.
///
/// While the mouse drives the camera (`mouse_captured`: fly mode or RMB look-drag), clicks,
/// motion and the wheel stay with the engine, and in fly mode so do key presses. Releases
/// always go through: ImGui must see every button and key it saw pressed come back up. A
/// button it still believes held since a click outside its windows turns off all window
/// hovering and capture, so every later click on a panel fell through to the world (look-drag
/// used to swallow the RMB release this way).
pub fn forward_to_imgui(event: &WindowEvent, mouse_captured: bool, fly_mode: bool) -> bool {
    match event {
        WindowEvent::MouseInput { state: ElementState::Released, .. } => true,
        WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Released => true,
        WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => !mouse_captured,
        WindowEvent::KeyboardInput { .. } => !fly_mode,
        _ => true,
    }
}

/// Deliver a mouse event straight to ImGui, the way the winit backend would. Used without a
/// window (scripted input in headless runs), where pixels are logical coordinates.
pub fn feed_imgui(ctx: &mut dear_imgui_rs::Context, event: &WindowEvent) {
    use winit::event::{MouseButton, MouseScrollDelta};
    let io = ctx.io_mut();
    match event {
        WindowEvent::CursorMoved { position, .. } => io.add_mouse_pos_event([position.x as f32, position.y as f32]),
        WindowEvent::MouseInput { state, button, .. } => {
            let b = match button {
                MouseButton::Left => dear_imgui_rs::MouseButton::Left,
                MouseButton::Right => dear_imgui_rs::MouseButton::Right,
                MouseButton::Middle => dear_imgui_rs::MouseButton::Middle,
                _ => return,
            };
            io.add_mouse_button_event(b, *state == ElementState::Pressed);
        }
        WindowEvent::MouseWheel { delta, .. } => {
            let (x, y) = match delta {
                MouseScrollDelta::LineDelta(x, y) => (*x, *y),
                MouseScrollDelta::PixelDelta(p) => (p.x as f32 / 40.0, p.y as f32 / 40.0),
            };
            io.add_mouse_wheel_event([x, y]);
        }
        _ => {}
    }
}

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

#[cfg(test)]
mod tests {
    use super::{feed_imgui, forward_to_imgui};
    use dear_imgui_rs as imgui;
    use winit::dpi::PhysicalPosition;
    use winit::event::{DeviceId, ElementState, MouseButton, WindowEvent};

    struct Sim {
        ctx: imgui::Context,
    }

    impl Sim {
        fn new() -> Self {
            let mut ctx = imgui::Context::create();
            let _ = ctx.set_ini_filename::<std::path::PathBuf>(None);
            {
                let io = ctx.io_mut();
                io.set_display_size([800.0, 600.0]);
                io.set_delta_time(1.0 / 60.0);
            }
            ctx.font_atlas().try_claim_legacy_renderer().expect("font atlas").build();
            Self { ctx }
        }

        /// Forward an event the way the engine does, then deliver it like the winit backend.
        fn send(&mut self, ev: WindowEvent, mouse_captured: bool) {
            if forward_to_imgui(&ev, mouse_captured, false) {
                feed_imgui(&mut self.ctx, &ev);
            }
        }

        /// One UI frame with a 200×200 panel in the top-left; returns ImGui's mouse capture.
        fn frame(&mut self) -> bool {
            let ui = self.ctx.frame();
            ui.window("panel").position([0.0, 0.0], imgui::Condition::Always).size([200.0, 200.0], imgui::Condition::Always).build(|| ui.text("panel"));
            self.ctx.end_frame();
            self.ctx.io().want_capture_mouse()
        }
    }

    fn cursor(x: f64, y: f64) -> WindowEvent {
        WindowEvent::CursorMoved { device_id: DeviceId::dummy(), position: PhysicalPosition::new(x, y) }
    }

    fn button(button: MouseButton, state: ElementState) -> WindowEvent {
        WindowEvent::MouseInput { device_id: DeviceId::dummy(), state, button }
    }

    /// RMB look-drag over the world, then hovering a panel: the panel must own the mouse again,
    /// otherwise a click on it falls through and edits the block behind it.
    #[test]
    fn panel_captures_mouse_after_look_drag() {
        let mut s = Sim::new();
        s.send(cursor(500.0, 400.0), false);
        s.frame();
        assert!(!s.frame(), "cursor over the world must not be captured");
        s.send(button(MouseButton::Right, ElementState::Pressed), false);
        s.frame();
        // the engine is now in look-drag: the cursor is grabbed until RMB comes up
        s.send(button(MouseButton::Right, ElementState::Released), true);
        s.frame();
        s.send(cursor(100.0, 100.0), false);
        s.frame();
        assert!(s.frame(), "panel under the cursor must capture the mouse after a look-drag");
    }
}

//! Action-mapped input system.
//!
//! Platform-agnostic input manager that maps physical key scancodes
//! to logical game actions. The windowing layer converts platform
//! events into `key_down`/`key_up`/`mouse_move` calls.

use std::collections::{HashMap, HashSet};

/// Logical game actions (hold-style).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    MoveForward,
    MoveBackward,
    StrafeLeft,
    StrafeRight,
    FlyUp,
    FlyDown,
    Sprint,
}

/// Maps physical key scancodes to logical actions and tracks input state.
pub struct InputManager {
    bindings: HashMap<u32, Action>,
    active: HashSet<Action>,
    mouse_delta: (f64, f64),
    /// Whether the cursor is grabbed for mouse look.
    /// Set by the app layer; `mouse_move` ignores deltas when false.
    pub cursor_grabbed: bool,
}

impl InputManager {
    /// Create an empty input manager with no bindings.
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            active: HashSet::new(),
            mouse_delta: (0.0, 0.0),
            cursor_grabbed: false,
        }
    }

    /// Bind a physical key scancode to a logical action.
    pub fn bind(&mut self, key: u32, action: Action) {
        self.bindings.insert(key, action);
    }

    /// Activate the action bound to this key scancode.
    pub fn key_down(&mut self, key: u32) {
        if let Some(&action) = self.bindings.get(&key) {
            self.active.insert(action);
        }
    }

    /// Deactivate the action bound to this key scancode.
    pub fn key_up(&mut self, key: u32) {
        if let Some(&action) = self.bindings.get(&key) {
            self.active.remove(&action);
        }
    }

    /// Accumulate mouse movement delta. Ignored when cursor is not grabbed.
    pub fn mouse_move(&mut self, dx: f64, dy: f64) {
        if self.cursor_grabbed {
            self.mouse_delta.0 += dx;
            self.mouse_delta.1 += dy;
        }
    }

    /// Returns true if the given action is currently held.
    pub fn is_active(&self, action: Action) -> bool {
        self.active.contains(&action)
    }

    /// Returns -1.0, 0.0, or 1.0 based on the negative/positive action pair.
    pub fn get_axis(&self, negative: Action, positive: Action) -> f32 {
        let neg = self.active.contains(&negative) as i32;
        let pos = self.active.contains(&positive) as i32;
        (pos - neg) as f32
    }

    /// Returns the accumulated mouse delta without consuming it.
    pub fn peek_mouse_delta(&self) -> (f64, f64) {
        self.mouse_delta
    }

    /// Returns the accumulated mouse delta since last call and resets it.
    pub fn take_mouse_delta(&mut self) -> (f64, f64) {
        let delta = self.mouse_delta;
        self.mouse_delta = (0.0, 0.0);
        delta
    }

    /// Clear all active actions and mouse delta.
    pub fn reset(&mut self) {
        self.active.clear();
        self.mouse_delta = (0.0, 0.0);
    }
}

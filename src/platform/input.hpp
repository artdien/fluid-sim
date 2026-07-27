#pragma once

#include <optional>
#include <string>
#include <utility>

#include "platform/types.hpp"

namespace fluidsim::platform {

struct MousePosition {
  f32 x;
  f32 y;
};

struct MousePositionDelta {
  f32 dx;
  f32 dy;
};

struct MouseInput {
  MousePosition position {0.0f, 0.0f};
  MousePositionDelta position_delta {0.0f, 0.0f};
  bool pressed {false};
};

struct KeyboardInput {
  std::string key {""};
};

/// Stores a mouse input event in a queue for later retrieval.
///
/// @param mouse_input Event to be stored.
auto add_mouse_input_event(MouseInput mouse_input) -> void;

/// Stores a keyboard input event in a queue for later retrieval.
///
/// @param mouse_input Event to be stored.
auto add_keyboard_input_event(KeyboardInput keyboard_input) -> void;

/// Retrieves a stored mouse input event.
///
/// Since events are stored in a queue, they are retrieved in FIFO order.
///
/// @return Optional containing mouse input event if queue is non-empty, otherwise std::nullopt.
auto get_mouse_input_event() -> std::optional<MouseInput>;

/// Retrieves a stored keyboard input event.
///
/// Since events are stored in a queue, they are retrieved in FIFO order.
///
/// @return Optional containing keyboard input event if queue is non-empty, otherwise std::nullopt.
auto get_keyboard_input_event() -> std::optional<KeyboardInput>;

} // namespace fluidsim::platform

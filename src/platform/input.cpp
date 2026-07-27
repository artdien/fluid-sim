#include "platform/input.hpp"

#include <queue>

namespace fluidsim::platform {

namespace {

std::queue<MouseInput> mouse_input_queue {};
std::queue<KeyboardInput> keyboard_input_queue {};

} // namespace

auto add_mouse_input_event(MouseInput mouse_input) -> void {
  mouse_input_queue.push(mouse_input);
}

auto add_keyboard_input_event(KeyboardInput keyboard_input) -> void {
  keyboard_input_queue.push(keyboard_input);
}

auto get_mouse_input_event() -> std::optional<MouseInput> {
  auto mouse_input {std::optional<MouseInput> {}};
  if (!mouse_input_queue.empty()) {
    mouse_input.emplace(mouse_input_queue.front());
    mouse_input_queue.pop();
  }
  return mouse_input;
}

auto get_keyboard_input_event() -> std::optional<KeyboardInput> {
  auto keyboard_input {std::optional<KeyboardInput> {}};
  if (!keyboard_input_queue.empty()) {
    keyboard_input.emplace(keyboard_input_queue.front());
    keyboard_input_queue.pop();
  }
  return keyboard_input;
}

} // namespace fluidsim::platform

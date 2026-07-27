#include <format>
#include <string_view>

#include "buffer/framebuffer.hpp"
#include "platform/window.hpp"
#include "rendering/renderer.hpp"
#include "simulation/solver.hpp"

using namespace fluidsim::platform;
using namespace fluidsim::simulation;
using namespace fluidsim::rendering;
using namespace fluidsim::buffer;

namespace {

constexpr auto WINDOW_TITLE {std::string_view {"Fluid Simulation"}};

auto process_input(Window* window, Solver* solver, const MouseInput& mouse, const KeyboardInput& keyboard) -> void {
  if (keyboard.key == "esc") {
    window->close();
  }
  if (mouse.pressed) {
    solver->add_external_force(mouse.position.x, mouse.position.y, mouse.position_delta.dx, mouse.position_delta.dy);
  }
}

} // namespace

auto main() -> int {
  constexpr auto width {1920u};
  constexpr auto height {1080u};

  auto window {Window {width, height}};
  auto solver {Solver {width, height}};
  auto renderer {Renderer {width, height}};
  auto framebuffer {Framebuffer {width, height}};

  window.open([&](MouseInput mouse [[maybe_unused]], KeyboardInput keyboard, f64 elapsed_time) {
    window.set_title(std::format("{} ({:.2f}ms)", WINDOW_TITLE, elapsed_time));
    process_input(&window, &solver, mouse, keyboard);

    solver.step();
    framebuffer.update(solver.grid());
    renderer.render(framebuffer.texture_id());
  });

  return 0;
}

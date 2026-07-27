#include "buffer/framebuffer.hpp"
#include "platform/window.hpp"
#include "rendering/renderer.hpp"
#include "simulation/solver.hpp"

using namespace fluidsim::platform;
using namespace fluidsim::simulation;
using namespace fluidsim::rendering;
using namespace fluidsim::buffer;

auto main() -> int {
  constexpr auto width {1920u};
  constexpr auto height {1080u};

  auto window {Window {width, height, "Fluid Simulation"}};
  auto solver {Solver {width, height}};
  auto renderer {Renderer {width, height}};
  auto framebuffer {Framebuffer {width, height}};

  window.open([&](MouseInput mouse [[maybe_unused]], KeyboardInput keyboard [[maybe_unused]], f64 elapsed_time [[maybe_unused]]) {
    solver.step();
    framebuffer.update(solver.grid());
    renderer.render(framebuffer.texture_id());
  });

  return 0;
}

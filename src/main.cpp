#include <format>
#include <random>
#include <string_view>

#include "buffer/framebuffer.hpp"
#include "platform/window.hpp"
#include "rendering/renderer.hpp"
#include "simulation/solver.hpp"
#include "ui/menu.hpp"
#include "utils/cli.hpp"

using namespace fluidsim::platform;
using namespace fluidsim::simulation;
using namespace fluidsim::rendering;
using namespace fluidsim::buffer;
using namespace fluidsim::ui;
using namespace fluidsim::utils;

namespace {

constexpr auto WINDOW_TITLE {std::string_view {"Fluid Simulation"}};
constexpr auto UPDATE_TIME_MS {1000.0 / 60.0};
constexpr auto MAX_LAG_MS {100.0};

auto random_device {std::random_device {}};
auto seed {std::seed_seq {static_cast<u32>(random_device())}};
auto generator {std::mt19937(seed)};
auto uniform {std::uniform_real_distribution<f32> {0.0f, 1.0f}};

auto parameters {SolverParameters {}};
auto configuration {ExternalConfiguration {}};
auto initial_state {InitialState::EMPTY};
auto randomize_external_dye {true};

auto process_input(Window* window, Menu* menu, Solver* solver, const MouseInput& mouse, const KeyboardInput& keyboard) -> void {
  if (keyboard.key == "esc") {
    window->close();
  }
  if (keyboard.key == "m") {
    menu->toggle();
  }

  if (mouse.pressed) {
    const auto r {randomize_external_dye ? uniform(generator) : 1.0f};
    const auto g {randomize_external_dye ? uniform(generator) : 1.0f};
    const auto b {randomize_external_dye ? uniform(generator) : 1.0f};

    solver->add_external_force(mouse.position.x, mouse.position.y, mouse.position_delta.dx, mouse.position_delta.dy);
    solver->add_external_dye(mouse.position.x, mouse.position.y, r, g, b);
  }
}

} // namespace

auto main(i32 argc, c8* argv[]) -> int {
  const auto width {parse_cli_argument(argc, argv, "-width").value_or(1920u)};
  const auto height {parse_cli_argument(argc, argv, "-height").value_or(1080u)};

  auto window {Window {width, height}};
  auto solver {Solver {width, height, parameters, configuration, initial_state}};
  auto renderer {Renderer {width, height}};
  auto framebuffer {Framebuffer {width, height}};
  auto menu {Menu {&solver, &parameters, &configuration, &initial_state, &randomize_external_dye}};

  auto lag {0.0};

  window.open([&](MouseInput mouse [[maybe_unused]], KeyboardInput keyboard, f64 elapsed_time) {
    window.set_title(std::format("{} ({:.2f}ms)", WINDOW_TITLE, elapsed_time));
    process_input(&window, &menu, &solver, mouse, keyboard);

    lag = std::min(lag + elapsed_time, MAX_LAG_MS);
    while (lag >= UPDATE_TIME_MS) {
      solver.step();
      lag -= UPDATE_TIME_MS;
    }

    framebuffer.update(solver.grid());
    renderer.render(framebuffer.texture_id());
    menu.display();
  });

  return 0;
}

#include "platform/window.hpp"

using namespace fluidsim::platform;

auto main() -> int {
  constexpr auto width {1920u};
  constexpr auto height {1080u};

  Window window {width, height, "Fluid Simulation"};
  window.open([](auto _) {});

  return 0;
}

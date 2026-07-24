#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto advect(GridView dye_next, GridView dye, GridView u, GridView v, f32 dt) -> void;

} // namespace fluidsim::simulation::kernels

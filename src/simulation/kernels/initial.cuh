#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto initialize_horizontal_split(GridView<f32> dye) -> void;

} // namespace fluidsim::simulation::kernels

#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto update_boundary(GridView dye) -> void;

} // namespace fluidsim::simulation::kernels

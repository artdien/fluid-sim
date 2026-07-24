#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto diffuse(GridView dye_next, GridView dye, f32 dt, f32 viscosity) -> void;

} // namespace fluidsim::simulation::kernels

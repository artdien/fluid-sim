#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto add_external_force(GridView<float2> velocity, f32 position_x, f32 position_y, f32 force_x, f32 force_y, f32 radius) -> void;

} // namespace fluidsim::simulation::kernels

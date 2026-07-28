#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto advect_velocity(GridView<float2> velocity_next, GridView<float2> velocity) -> void;

auto advect_dye(GridView<f32> dye_next, GridView<f32> dye, GridView<float2> velocity) -> void;

} // namespace fluidsim::simulation::kernels

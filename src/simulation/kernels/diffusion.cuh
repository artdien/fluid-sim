#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto diffuse_velocity(GridView<float2> velocity_next, GridView<float2> velocity) -> void;

auto diffuse_dye(GridView<f32> dye_next, GridView<f32> dye) -> void;

} // namespace fluidsim::simulation::kernels

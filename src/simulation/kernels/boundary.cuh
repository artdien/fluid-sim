#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto update_velocity_boundary(GridView<float2> velocity) -> void;

auto update_pressure_boundary(GridView<f32> pressure) -> void;

auto update_dye_boundary(GridView<f32> dye) -> void;

} // namespace fluidsim::simulation::kernels

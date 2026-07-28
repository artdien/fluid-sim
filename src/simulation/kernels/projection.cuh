#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto calculate_divergence(GridView<f32> divergence, GridView<float2> velocity) -> void;

auto solve_pressure(GridView<f32> pressure_next, GridView<f32> pressure, GridView<f32> divergence) -> void;

auto project(GridView<float2> velocity, GridView<f32> p) -> void;

} // namespace fluidsim::simulation::kernels

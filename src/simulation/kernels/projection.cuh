#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto calculate_divergence(GridView divergence, GridView u, GridView v, f32 dt, f32 density) -> void;

auto solve_pressure(GridView pressure_next, GridView pressure, GridView divergence) -> void;

auto project(GridView u, GridView v, GridView p, f32 dt, f32 density) -> void;

} // namespace fluidsim::simulation::kernels

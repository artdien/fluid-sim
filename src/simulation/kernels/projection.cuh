#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto calculate_divergence(GridView divergence, GridView u, GridView v) -> void;

auto solve_pressure(GridView pressure_next, GridView pressure, GridView divergence) -> void;

auto project(GridView u, GridView v, GridView p) -> void;

} // namespace fluidsim::simulation::kernels

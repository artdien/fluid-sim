#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto update_u_boundary(GridView u) -> void;

auto update_v_boundary(GridView u) -> void;

auto update_pressure_boundary(GridView pressure) -> void;

auto update_dye_boundary(GridView dye) -> void;

} // namespace fluidsim::simulation::kernels

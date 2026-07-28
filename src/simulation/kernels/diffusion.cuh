#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto diffuse_u(GridView u_next, GridView u) -> void;

auto diffuse_v(GridView v_next, GridView v) -> void;

auto diffuse_dye(GridView dye_next, GridView dye) -> void;

} // namespace fluidsim::simulation::kernels

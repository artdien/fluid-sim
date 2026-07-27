#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto advect_u(GridView u_next, GridView u, GridView v, f32 dt) -> void;

auto advect_v(GridView v_next, GridView u, GridView v, f32 dt) -> void;

auto advect_dye(GridView dye_next, GridView dye, GridView u, GridView v, f32 dt) -> void;

} // namespace fluidsim::simulation::kernels

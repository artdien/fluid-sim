#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

auto diffuse_u(GridView u_next, GridView u, f32 dt, f32 viscosity) -> void;

auto diffuse_v(GridView v_next, GridView v, f32 dt, f32 viscosity) -> void;

auto diffuse_dye(GridView dye_next, GridView dye, f32 dt, f32 viscosity) -> void;

} // namespace fluidsim::simulation::kernels

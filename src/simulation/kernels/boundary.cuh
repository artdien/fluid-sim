#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

/// @brief Enforces free-slip boundary conditions for the velocity field.
///
/// This method performs a three-pass update on the ghost cells surrounding the grid:
/// columns, rows, and finally corners.
///
/// @param velocity Velocity grid for which to enforce boundary conditions (updated in-place).
///
/// @note This function launches multiple kernels sequentially to update all boundary values.
auto update_velocity_boundary(GridView<float2> velocity) -> void;

/// @brief Enforces free-slip boundary conditions for the pressure field.
///
/// This method performs a three-pass update on the ghost cells surrounding the grid:
/// columns, rows, and finally corners.
///
/// @param pressure Pressure grid for which to enforce boundary conditions (updated in-place).
///
/// @note This function launches multiple kernels sequentially to update all boundary values.
auto update_pressure_boundary(GridView<f32> pressure) -> void;

/// @brief Enforces free-slip boundary conditions for the dye field.
///
/// This method performs a three-pass update on the ghost cells surrounding the grid:
/// columns, rows, and finally corners.
///
/// @param dye Dye grid for which to enforce boundary conditions (updated in-place).
///
/// @note This function launches multiple kernels sequentially to update all boundary values.
auto update_dye_boundary(GridView<f32> dye) -> void;

} // namespace fluidsim::simulation::kernels

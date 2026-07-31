#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

struct BoundaryStreams {
  cudaStream_t corner;
  cudaStream_t column;
  cudaStream_t row;
};

/// @brief Enforces free-slip boundary conditions for the velocity field.
///
/// This method performs a three-pass update on the ghost cells surrounding the grid:
/// columns, rows, and finally corners.
///
/// @param velocity   Velocity grid for which to enforce boundary conditions (updated in-place).
/// @param streams    Streams on which to update the boundary values.
/// @param block_size Length of thread block.
///
/// @note This function launches multiple kernels sequentially to update all boundary values.
///       The corner boundary values are updated on the default stream.
auto update_velocity_boundary(GridView<float2> velocity, BoundaryStreams streams, u32 block_size) -> void;

/// @brief Enforces free-slip boundary conditions for the pressure field.
///
/// This method performs a three-pass update on the ghost cells surrounding the grid:
/// columns, rows, and finally corners.
///
/// @param pressure   Pressure grid for which to enforce boundary conditions (updated in-place).
/// @param streams    Streams on which to update the boundary values.
/// @param block_size Length of thread block.
///
/// @note This function launches multiple kernels sequentially to update all boundary values.
auto update_pressure_boundary(GridView<f32> pressure, BoundaryStreams streams, u32 block_size) -> void;

/// @brief Enforces free-slip boundary conditions for the dye field.
///
/// This method performs a three-pass update on the ghost cells surrounding the grid:
/// columns, rows, and finally corners.
///
/// @param dye        Dye grid for which to enforce boundary conditions (updated in-place).
/// @param streams    Streams on which to update the boundary values.
/// @param block_size Length of thread block.
///
/// @note This function launches multiple kernels sequentially to update all boundary values.
auto update_dye_boundary(GridView<float4> dye, BoundaryStreams streams, u32 block_size) -> void;

} // namespace fluidsim::simulation::kernels

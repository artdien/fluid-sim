#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

/// @brief Computes the divergence of the current velocity field.
///
/// Calculates how much fluid is entering or leaving each cell.
/// This divergence acts as the source term for the pressure Poisson equation.
///
/// @param divergence Destination grid to store calculated divergence values.
/// @param velocity   Current velocity grid.
/// @param block_size Side length of a square thread block (e.g. 16 for a 16x16 block).
auto calculate_divergence(GridView<f32> divergence, GridView<float2> velocity, u32 block_size) -> void;

/// @brief Iteratively solves the Poisson equation for pressure.
///
/// Uses a weighted Jacobi iterative solver to find a pressure field
/// that can counteract the current divergence of the flow.
///
/// @param pressure_next Destination grid for this iteration's pressure values.
/// @param pressure      Source pressure grid from the previous iteration.
/// @param divergence    Divergence grid acting as the constraint.
/// @param block_size    Side length of a square thread block (e.g. 16 for a 16x16 block).
auto solve_pressure(GridView<f32> pressure_next, GridView<f32> pressure, GridView<f32> divergence, u32 block_size) -> void;

/// @brief Projects the velocity field onto a divergence-free field.
///
/// Subtracts the gradient of the solved pressure field from the velocity field,
/// effectively 'correcting' the flow to be incompressible.
///
/// @param velocity   Velocity grid to be corrected (updated in-place).
/// @param pressure   Solved pressure grid used for the correction.
/// @param block_size Side length of a square thread block (e.g. 16 for a 16x16 block).
auto project(GridView<float2> velocity, GridView<f32> pressure, u32 block_size) -> void;

} // namespace fluidsim::simulation::kernels

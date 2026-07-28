#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

/// @brief Simulates viscous diffusion of the velocity field.
///
/// Applies a Laplacian operator across the grid to smooth out velocity
/// gradients, simulating fluid viscosity.
///
/// @param velocity_next Destination grid for the diffused velocity.
/// @param velocity      Current source velocity grid.
auto diffuse_velocity(GridView<float2> velocity_next, GridView<float2> velocity) -> void;

/// @brief Simulates the viscous diffusion of a dye field.
///
/// Spreads the concentration of the dye across neighboring cells based
/// on the simulation's dye viscosity parameter.
///
/// @param dye_next Destination grid for the diffused dye.
/// @param dye      Current source dye grid.
auto diffuse_dye(GridView<f32> dye_next, GridView<f32> dye) -> void;

} // namespace fluidsim::simulation::kernels

#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

/// @brief Calculates the vorticity of the current velocity field.
///
/// Computes the vorticity via the curl of the velocity field.
///
/// @param vorticity  Destination grid to store calculated vorticity values.
/// @param velocity   Current velocity grid.
/// @param block_size Side length of a square thread block (e.g. 16 for a 16x16 block).
auto calculate_vorticity(GridView<f32> vorticity, GridView<float2> velocity, u32 block_size) -> void;

/// @brief Applies the vorticity confinement force to the velocity field.
///
/// Vorticity confinement identifies small-scale vortices and adds a restorative force
/// to keep them from disappearing due to numerical dissipation.
///
/// @param velocity   Velocity grid to be updated (updated in-place).
/// @param vorticity  Vorticity grid containing pre-calculated vorticity magnitudes.
/// @param block_size Side length of a square thread block (e.g. 16 for a 16x16 block).
auto apply_vorticity_confinement(GridView<float2> velocity, GridView<f32> vorticity, u32 block_size) -> void;

} // namespace fluidsim::simulation::kernels

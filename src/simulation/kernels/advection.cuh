#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

/// @brief Transports the velocity field through itself (self-advection).
///
/// Uses a semi-Lagrangian scheme to trace the velocity at each grid cell
/// backward in time and interpolates the value from the previous state.
///
/// @param velocity_next Destination grid for the advected velocity field.
/// @param velocity      Current source velocity grid.
/// @param block_size    Side length of a square thread block (e.g. 16 for a 16x16 block).
auto advect_velocity(GridView<float2> velocity_next, GridView<float2> velocity, u32 block_size) -> void;

/// @brief Transports a dye field along the current velocity flow.
///
/// This method treats the dye as a passive quantity, moving it across the
/// grid based on the provided velocity field.
///
/// @param dye_next   Destination grid for the advected dye concentrations.
/// @param dye        Current source dye grid.
/// @param velocity   Velocity grid used to determine transport paths.
/// @param block_size Side length of a square thread block (e.g. 16 for a 16x16 block).
auto advect_dye(GridView<float4> dye_next, GridView<float4> dye, GridView<float2> velocity, u32 block_size) -> void;

} // namespace fluidsim::simulation::kernels

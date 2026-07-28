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
auto advect_velocity(GridView<float2> velocity_next, GridView<float2> velocity) -> void;

/// @brief Transports a dye field along the current velocity flow.
///
/// This method treats the dye as a passive quantity, moving it across the
/// grid based on the provided velocity field.
///
/// @param dye_next  Destination grid for the advected dye concentrations.
/// @param dye       Current source dye grid.
/// @param velocity  Velocity grid used to determine transport paths.
auto advect_dye(GridView<f32> dye_next, GridView<f32> dye, GridView<float2> velocity) -> void;

} // namespace fluidsim::simulation::kernels

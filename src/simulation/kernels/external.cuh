#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

/// @brief Injects an external force into the velocity field within a circular area of influence.
///
/// This method applies a force to the fluid centered at the specified coordinates.
/// The impact of the force is modulated by a linear falloff, where the maximum
/// magnitude is applied at the center and decreases linearly to zero at the radius boundary.
///
/// @param velocity   Velocity field GridView to be modified.
/// @param position_x X-coordinate of the center of the force application.
/// @param position_y Y-coordinate of the center of the force application.
/// @param force_x    Magnitude and direction of the force along the x-axis.
/// @param force_y    Magnitude and direction of the force along the y-axis.
/// @param radius     Radius of the area affected by the force.
auto add_external_force(GridView<float2> velocity, f32 position_x, f32 position_y, f32 force_x, f32 force_y, f32 radius) -> void;

} // namespace fluidsim::simulation::kernels

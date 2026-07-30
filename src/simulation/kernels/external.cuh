#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

/// @brief Injects an external force into the velocity field within a circular area of influence.
///
/// This method applies a force to the fluid centered at the specified coordinates.
/// The impact of the force is modulated by a linear falloff, where the maximum
/// magnitude is applied at the center and decreases linearly to zero at the radius boundary.
///
/// @param velocity   Velocity field to be modified.
/// @param x          X-coordinate of the center of the force application.
/// @param y          Y-coordinate of the center of the force application.
/// @param f_x        Magnitude and direction of the force along the x-axis.
/// @param f_y        Magnitude and direction of the force along the y-axis.
/// @param block_size Side length of a square thread block (e.g. 16 for a 16x16 block).
auto add_external_force(GridView<float2> velocity, f32 x, f32 y, f32 f_x, f32 f_y, u32 block_size) -> void;

/// @brief Injects an external dye source into the dye field within a circular area of influence.
///
/// This method adds dye to the fluid centered at the specified coordinates.
/// The impact of the dye is modulated by a linear falloff, where the maximum
/// magnitude is applied at the center and decreases linearly to zero at the radius boundary.
///
/// @param dye        Dye field to be modified.
/// @param x          X-coordinate of the center of the added dye source.
/// @param y          Y-coordinate of the center of the added dye source.
/// @param r          Red channel of the dye source.
/// @param g          Green channel of the dye source.
/// @param b          Blue channel of the dye source.
/// @param block_size Side length of a square thread block (e.g. 16 for a 16x16 block).
auto add_external_dye(GridView<float4> dye, f32 x, f32 y, f32 r, f32 g, f32 b, u32 block_size) -> void;

} // namespace fluidsim::simulation::kernels

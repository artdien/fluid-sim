#pragma once

#include "simulation/grid.hpp"

namespace fluidsim::simulation::kernels {

/// @brief Initializes the dye field with a horizontally split concentration.
///
/// This method sets the left half of the simulation domain
/// to a maximum concentration (1.0) and leaves the right half empty (0.0).
///
/// @param dye Dye grid to be initialized (updated in-place).
auto initialize_horizontal_split(GridView<f32> dye) -> void;

/// @brief Initializes the dye field with a vertically split concentration.
///
/// This method sets the upper half of the simulation domain
/// to a maximum concentration (1.0) and leaves the bottom half empty (0.0).
///
/// @param dye Dye grid to be initialized (updated in-place).
auto initialize_vertical_split(GridView<f32> dye) -> void;

/// @brief Initializes the dye field with an empty concentration.
///
/// This method sets the entire simulation domain to a minimum concentration (0.0).
///
/// @param dye Dye grid to be initialized (updated in-place).
auto initialize_empty(GridView<f32> dye) -> void;

} // namespace fluidsim::simulation::kernels

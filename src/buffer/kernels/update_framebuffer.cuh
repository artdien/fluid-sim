#pragma once

#include <glad/glad.h> // must be included before cuda_gl_interop.h

#include <cuda_gl_interop.h>

#include "simulation/grid.hpp"

namespace fluidsim::buffer::kernels {

/// @brief Renders a simulation grid directly into a CUDA surface framebuffer for visualization.
///
/// This method maps the scalar values from a simulation grid to a color representation
/// and writes them as float4 pixels directly into the GPU's surface memory.
///
/// @param surface    CUDA surface object representing the target framebuffer.
/// @param grid       Grid (e.g. dye concentration) to be visualized.
///
/// @note Values are clamped between 0.0 and 1.0,
///       ensuring that the resulting colors remain within a valid range for display.
auto update_framebuffer(cudaSurfaceObject_t surface, simulation::GridView<float4> grid) -> void;

} // namespace fluidsim::buffer::kernels

#pragma once

#include <cuda_gl_interop.h>

#include "simulation/grid.hpp"

namespace fluidsim::buffer::kernels {

auto update_framebuffer(cudaSurfaceObject_t surface, simulation::GridView grid) -> void;

} // namespace fluidsim::buffer::kernels

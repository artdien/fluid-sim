#include "simulation/solver.hpp"

#include <cuda_runtime.h>

#include "simulation/kernels/dummy.cuh"
#include "utils/cuda.hpp"

namespace fluidsim::simulation {

Solver::Solver(u32 width, u32 height) : grid_ {width, height} {}

auto Solver::step() -> void {
  kernels::dummy(grid_.view());
  utils::check_async_cuda_error();
}

auto Solver::grid() const -> const GridView {
  return grid_.view();
}

} // namespace fluidsim::simulation

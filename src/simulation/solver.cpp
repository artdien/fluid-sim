#include "simulation/solver.hpp"

#include <cuda_runtime.h>

#include "simulation/kernels/advection.cuh"
#include "simulation/kernels/boundary.cuh"
#include "simulation/kernels/diffusion.cuh"
#include "simulation/kernels/initial.cuh"

namespace fluidsim::simulation {

namespace {

constexpr auto dt {1.0f};
constexpr auto viscosity {0.2f};

} // namespace

Solver::Solver(u32 width, u32 height)
    : u_ {width, height},  //
      v_ {width, height},  //
      dye_ {width, height} //
{
  kernels::initialize_horizontal_split(dye_.current());
}

auto Solver::step() -> void {
  kernels::advect(dye_.next(), dye_.current(), u_.current(), v_.current(), dt);
  kernels::update_boundary(dye_.next());
  dye_.swap();

  kernels::diffuse(dye_.next(), dye_.current(), dt, viscosity);
  kernels::update_boundary(dye_.next());
  dye_.swap();
}

auto Solver::grid() const -> const GridView {
  return dye_.current();
}

} // namespace fluidsim::simulation

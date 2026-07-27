#include "simulation/solver.hpp"

#include <cuda_runtime.h>

#include "simulation/kernels/advection.cuh"
#include "simulation/kernels/boundary.cuh"
#include "simulation/kernels/diffusion.cuh"
#include "simulation/kernels/initial.cuh"
#include "simulation/kernels/projection.cuh"

namespace fluidsim::simulation {

namespace {

constexpr auto dt {1.0f};
constexpr auto density {1.0f};
constexpr auto viscosity {0.2f};
constexpr auto viscosity_dye {0.2f};
constexpr auto jacobi_iterations {40u};

} // namespace

Solver::Solver(u32 width, u32 height)
    : u_ {width, height},         //
      v_ {width, height},         //
      dye_ {width, height},       //
      pressure_ {width, height},  //
      divergence_ {width, height} //
{
  kernels::initialize_horizontal_split(dye_.current());
}

auto Solver::step() -> void {
  kernels::advect_u(u_.next(), u_.current(), v_.current(), dt);
  kernels::update_u_boundary(u_.next());
  kernels::advect_v(v_.next(), u_.current(), v_.current(), dt);
  kernels::update_v_boundary(v_.next());
  u_.swap();
  v_.swap();

  if (viscosity > 0.0f) {
    kernels::diffuse_u(u_.next(), u_.current(), dt, viscosity);
    kernels::update_u_boundary(u_.next());
    kernels::diffuse_v(v_.next(), v_.current(), dt, viscosity);
    kernels::update_v_boundary(v_.next());
    u_.swap();
    v_.swap();
  }

  kernels::calculate_divergence(divergence_.view(), u_.current(), v_.current(), dt, density);

  for (auto i {0u}; i < jacobi_iterations; ++i) {
    kernels::solve_pressure(pressure_.next(), pressure_.current(), divergence_.view());
    kernels::update_pressure_boundary(pressure_.next());
    pressure_.swap();
  }

  kernels::project(u_.current(), v_.current(), pressure_.current(), dt, density);
  kernels::update_u_boundary(u_.current());
  kernels::update_v_boundary(v_.current());

  kernels::advect_dye(dye_.next(), dye_.current(), u_.current(), v_.current(), dt);
  kernels::update_dye_boundary(dye_.next());
  dye_.swap();

  if (viscosity_dye > 0.0f) {
    kernels::diffuse_dye(dye_.next(), dye_.current(), dt, viscosity_dye);
    kernels::update_dye_boundary(dye_.next());
    dye_.swap();
  }
}

auto Solver::grid() const -> const GridView {
  return dye_.current();
}

} // namespace fluidsim::simulation

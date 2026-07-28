#include "simulation/solver.hpp"

#include <cuda_runtime.h>

#include "simulation/kernels/advection.cuh"
#include "simulation/kernels/boundary.cuh"
#include "simulation/kernels/diffusion.cuh"
#include "simulation/kernels/external.cuh"
#include "simulation/kernels/initial.cuh"
#include "simulation/kernels/parameters.cuh"
#include "simulation/kernels/projection.cuh"
#include "utils/cuda.hpp"

namespace fluidsim::simulation {

namespace {

auto upload_parameters(const SolverParameters& parameters) -> void {
  utils::check_cuda_error(cudaMemcpyToSymbol(&dt, &parameters.dt, sizeof(f32)));
  utils::check_cuda_error(cudaMemcpyToSymbol(&density, &parameters.density, sizeof(f32)));
  utils::check_cuda_error(cudaMemcpyToSymbol(&viscosity, &parameters.viscosity, sizeof(f32)));
  utils::check_cuda_error(cudaMemcpyToSymbol(&viscosity_dye, &parameters.viscosity_dye, sizeof(f32)));
  utils::check_cuda_error(cudaMemcpyToSymbol(&jacobi_weight, &parameters.jacobi_weight, sizeof(f32)));

  const auto dt_over_density_ {parameters.dt / parameters.density};
  const auto density_over_dt_ {parameters.density / parameters.dt};
  const auto density_inverse_ {1.0f / parameters.density};
  const auto viscosity_times_dt_ {parameters.viscosity * parameters.dt};
  const auto viscosity_dye_times_dt_ {parameters.viscosity_dye * parameters.dt};

  utils::check_cuda_error(cudaMemcpyToSymbol(&dt_over_density, &dt_over_density_, sizeof(f32)));
  utils::check_cuda_error(cudaMemcpyToSymbol(&density_over_dt, &density_over_dt_, sizeof(f32)));
  utils::check_cuda_error(cudaMemcpyToSymbol(&density_inverse, &density_inverse_, sizeof(f32)));
  utils::check_cuda_error(cudaMemcpyToSymbol(&viscosity_times_dt, &viscosity_times_dt_, sizeof(f32)));
  utils::check_cuda_error(cudaMemcpyToSymbol(&viscosity_dye_times_dt, &viscosity_dye_times_dt_, sizeof(f32)));
}

} // namespace

struct Solver::Impl {
  DoubleGrid<float2> velocity;
  DoubleGrid<f32> dye;
  DoubleGrid<f32> pressure;
  Grid<f32> divergence;

  Impl(u32 width, u32 height) : velocity {width, height}, dye {width, height}, pressure {width, height}, divergence {width, height} {}
};

Solver::Solver(const SolverParameters& parameters, u32 width, u32 height) : parameters_ {parameters}, pimpl_ {std::make_unique<Impl>(width, height)} {
  upload_parameters(parameters_);

  kernels::initialize_horizontal_split(pimpl_->dye.current());
}

Solver::~Solver() {
  // Destructor implementation necessary since full definition of struct Impl is only visible in source file.
  // Otherwise unique pointer can't instantiate its own destructor.
}

auto Solver::step() -> void {
  const auto lock {std::lock_guard {mutex_}};

  kernels::advect_velocity(pimpl_->velocity.next(), pimpl_->velocity.current());
  kernels::update_velocity_boundary(pimpl_->velocity.next());
  pimpl_->velocity.swap();

  if (parameters_.viscosity > 0.0f) {
    kernels::diffuse_velocity(pimpl_->velocity.next(), pimpl_->velocity.current());
    kernels::update_velocity_boundary(pimpl_->velocity.next());
    pimpl_->velocity.swap();
  }

  kernels::calculate_divergence(pimpl_->divergence.view(), pimpl_->velocity.current());

  for (auto i {0u}; i < parameters_.jacobi_iterations; ++i) {
    kernels::solve_pressure(pimpl_->pressure.next(), pimpl_->pressure.current(), pimpl_->divergence.view());
    kernels::update_pressure_boundary(pimpl_->pressure.next());
    pimpl_->pressure.swap();
  }

  kernels::project(pimpl_->velocity.current(), pimpl_->pressure.current());
  kernels::update_velocity_boundary(pimpl_->velocity.current());

  kernels::advect_dye(pimpl_->dye.next(), pimpl_->dye.current(), pimpl_->velocity.current());
  kernels::update_dye_boundary(pimpl_->dye.next());
  pimpl_->dye.swap();

  if (parameters_.viscosity_dye > 0.0f) {
    kernels::diffuse_dye(pimpl_->dye.next(), pimpl_->dye.current());
    kernels::update_dye_boundary(pimpl_->dye.next());
    pimpl_->dye.swap();
  }
}

auto Solver::add_external_force(f32 position_x, f32 position_y, f32 force_x, f32 force_y) -> void {
  const auto lock {std::lock_guard {mutex_}};

  kernels::add_external_force(pimpl_->velocity.current(), position_x, position_y, force_x, force_y, 15.0f);
  kernels::update_velocity_boundary(pimpl_->velocity.current());
}

auto Solver::update_parameters(const SolverParameters parameters) -> void {
  const auto lock {std::lock_guard {mutex_}};

  upload_parameters(parameters);
}

auto Solver::grid() const -> const RawGridView {
  return pimpl_->dye.current().raw();
}

} // namespace fluidsim::simulation

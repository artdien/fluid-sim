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

  // Pre-calculate some derived parameters for efficiency
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

auto upload_configuration(const ExternalConfiguration& configuration) {
  utils::check_cuda_error(cudaMemcpyToSymbol(&external_force_radius, &configuration.external_force_radius, sizeof(f32)));
  utils::check_cuda_error(cudaMemcpyToSymbol(&external_dye_radius, &configuration.external_dye_radius, sizeof(f32)));
}

} // namespace

struct Solver::Impl {
  DoubleGrid<float2> velocity;
  DoubleGrid<float4> dye;
  DoubleGrid<f32> pressure;
  Grid<f32> divergence;

  // Streams for updating boundary values concurrently
  cudaStream_t column_stream;
  cudaStream_t row_stream;
  cudaStream_t corner_stream;

  Impl(u32 width, u32 height) : velocity {width, height}, dye {width, height}, pressure {width, height}, divergence {width, height} {
    cudaStreamCreate(&column_stream);
    cudaStreamCreate(&row_stream);
    cudaStreamCreate(&corner_stream);
  }
};

Solver::Solver(u32 width, u32 height, const SolverParameters& parameters, const ExternalConfiguration& configuration, InitialState initial_state)
    : parameters_ {parameters}, configuration_ {configuration}, pimpl_ {std::make_unique<Impl>(width, height)} {
  update_parameters(parameters_);
  update_configuration(configuration_);
  reset(initial_state);
}

Solver::~Solver() {
  cudaStreamDestroy(pimpl_->column_stream);
  cudaStreamDestroy(pimpl_->row_stream);
  cudaStreamDestroy(pimpl_->corner_stream);
}

auto Solver::step() -> void {
  const auto lock {std::lock_guard {mutex_}};

  kernels::advect_velocity(pimpl_->velocity.next(), pimpl_->velocity.current(), parameters_.block_size);
  kernels::update_velocity_boundary(pimpl_->velocity.next(), pimpl_->column_stream, pimpl_->row_stream, parameters_.block_size);
  pimpl_->velocity.swap();

  if (parameters_.viscosity > 0.0f) {
    kernels::diffuse_velocity(pimpl_->velocity.next(), pimpl_->velocity.current(), parameters_.block_size);
    kernels::update_velocity_boundary(pimpl_->velocity.next(), pimpl_->column_stream, pimpl_->row_stream, parameters_.block_size);
    pimpl_->velocity.swap();
  }

  kernels::calculate_divergence(pimpl_->divergence.view(), pimpl_->velocity.current(), parameters_.block_size);

  for (auto i {0u}; i < parameters_.jacobi_iterations; ++i) {
    kernels::solve_pressure(pimpl_->pressure.next(), pimpl_->pressure.current(), pimpl_->divergence.view(), parameters_.block_size);
    kernels::update_pressure_boundary(pimpl_->pressure.next(), pimpl_->column_stream, pimpl_->row_stream, pimpl_->corner_stream, parameters_.block_size);
    pimpl_->pressure.swap();
  }

  kernels::project(pimpl_->velocity.current(), pimpl_->pressure.current(), parameters_.block_size);
  kernels::update_velocity_boundary(pimpl_->velocity.current(), pimpl_->column_stream, pimpl_->row_stream, parameters_.block_size);

  kernels::advect_dye(pimpl_->dye.next(), pimpl_->dye.current(), pimpl_->velocity.current(), parameters_.block_size);
  kernels::update_dye_boundary(pimpl_->dye.next(), pimpl_->column_stream, pimpl_->row_stream, pimpl_->corner_stream, parameters_.block_size);
  pimpl_->dye.swap();

  if (parameters_.viscosity_dye > 0.0f) {
    kernels::diffuse_dye(pimpl_->dye.next(), pimpl_->dye.current(), parameters_.block_size);
    kernels::update_dye_boundary(pimpl_->dye.next(), pimpl_->column_stream, pimpl_->row_stream, pimpl_->corner_stream, parameters_.block_size);
    pimpl_->dye.swap();
  }
}

auto Solver::add_external_force(f32 x, f32 y, f32 f_x, f32 f_y) -> void {
  if (!configuration_.allow_adding_external_force) {
    return;
  }

  const auto lock {std::lock_guard {mutex_}};

  kernels::add_external_force(pimpl_->velocity.current(), x, y, f_x, f_y, parameters_.block_size);
  kernels::update_velocity_boundary(pimpl_->velocity.current(), pimpl_->column_stream, pimpl_->row_stream, parameters_.block_size);
}

auto Solver::add_external_dye(f32 x, f32 y, f32 r, f32 g, f32 b) -> void {
  if (!configuration_.allow_adding_external_dye) {
    return;
  }

  const auto lock {std::lock_guard {mutex_}};

  kernels::add_external_dye(pimpl_->dye.current(), x, y, r, g, b, parameters_.block_size);
  kernels::update_dye_boundary(pimpl_->dye.current(), pimpl_->column_stream, pimpl_->row_stream, pimpl_->corner_stream, parameters_.block_size);
}

auto Solver::update_parameters(const SolverParameters& parameters) -> void {
  const auto lock {std::lock_guard {mutex_}};

  parameters_ = parameters;
  upload_parameters(parameters_);
}

auto Solver::update_configuration(const ExternalConfiguration& configuration) -> void {
  const auto lock {std::lock_guard {mutex_}};

  configuration_ = configuration;
  upload_configuration(configuration_);
}

auto Solver::reset(InitialState initial_state) -> void {
  const auto lock {std::lock_guard {mutex_}};

  pimpl_->velocity.reset();
  pimpl_->dye.reset();
  pimpl_->pressure.reset();
  pimpl_->divergence.reset();

  switch (initial_state) {
    case InitialState::EMPTY:
      kernels::initialize_empty(pimpl_->dye.current(), parameters_.block_size);
      break;

    case InitialState::HORIZONTAL_SPLIT:
      kernels::initialize_horizontal_split(pimpl_->dye.current(), parameters_.block_size);
      break;

    case InitialState::VERTICAL_SPLIT:
      kernels::initialize_vertical_split(pimpl_->dye.current(), parameters_.block_size);
      break;
  }
}

auto Solver::grid() const -> const RawGridView {
  return pimpl_->dye.current().raw();
}

} // namespace fluidsim::simulation

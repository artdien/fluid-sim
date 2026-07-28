#include "simulation/kernels/projection.cuh"

#include "simulation/kernels/parameters.cuh"
#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto calculate_divergence_kernel(GridView<f32> divergence, GridView<float2> velocity) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= divergence.width && j <= divergence.height) {
    const auto vel {velocity.at(i, j)};
    divergence.at(i, j) = 0.25f * density_over_dt * (vel.x - velocity.at(i - 1, j).x + vel.y - velocity.at(i, j - 1).y);
  }
}

__global__ auto solve_pressure_kernel(GridView<f32> pressure_next, GridView<f32> pressure, GridView<f32> divergence) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= pressure.width && j <= pressure.height) {
    pressure_next.at(i, j) =
        (1.0f - jacobi_weight) * pressure.at(i, j) +
        jacobi_weight * (0.25f * (pressure.at(i + 1, j) + pressure.at(i - 1, j) + pressure.at(i, j + 1) + pressure.at(i, j - 1)) - divergence.at(i, j));
  }
}

__global__ auto project_kernel(GridView<float2> velocity, GridView<f32> pressure) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= pressure.width && j <= pressure.height) {
    const auto p {pressure.at(i, j)};
    const auto vel {velocity.at(i, j)};

    // This kernel 'incorrectly' sets some boundary values for the velocity.
    // However, they get corrected when updating the boundary values.
    velocity.at(i, j) = make_float2(vel.x - dt_over_density * (pressure.at(i + 1, j) - p), //
                                    vel.y - dt_over_density * (pressure.at(i, j + 1) - p));
  }
}

} // namespace

auto calculate_divergence(GridView<f32> divergence, GridView<float2> velocity) -> void {
  const auto [blocks, threads] {utils::execution_configuration(divergence.width, divergence.height, 16)};
  calculate_divergence_kernel<<<blocks, threads>>>(divergence, velocity);
  utils::check_async_cuda_error();
}

auto solve_pressure(GridView<f32> pressure_next, GridView<f32> pressure, GridView<f32> divergence) -> void {
  const auto [blocks, threads] {utils::execution_configuration(pressure.width, pressure.height, 16)};
  solve_pressure_kernel<<<blocks, threads>>>(pressure_next, pressure, divergence);
  utils::check_async_cuda_error();
}

auto project(GridView<float2> velocity, GridView<f32> pressure) -> void {
  const auto [blocks, threads] {utils::execution_configuration(pressure.width, pressure.height, 16)};
  project_kernel<<<blocks, threads>>>(velocity, pressure);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

#include "simulation/kernels/projection.cuh"

#include "simulation/kernels/parameters.cuh"
#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto calculate_divergence_kernel(GridView divergence, GridView u, GridView v) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= divergence.width && j <= divergence.height) {
    divergence.at(i, j) = 0.25f * (density / dt) * (u.at(i, j) - u.at(i - 1, j) + v.at(i, j) - v.at(i, j - 1));
  }
}

__global__ auto solve_pressure_kernel(GridView pressure_next, GridView pressure, GridView divergence) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= pressure.width && j <= pressure.height) {
    pressure_next.at(i, j) =
        (1.0f - jacobi_weight) * pressure.at(i, j) +
        jacobi_weight * (0.25f * (pressure.at(i + 1, j) + pressure.at(i - 1, j) + pressure.at(i, j + 1) + pressure.at(i, j - 1)) - divergence.at(i, j));
  }
}

__global__ auto project_kernel(GridView u, GridView v, GridView pressure) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= pressure.width && j <= pressure.height) {
    const auto factor {dt / density};
    const auto p {pressure.at(i, j)};

    // This 'incorrectly' sets some boundary values for u and v.
    // However, they get immediately corrected after this kernel.
    u.at(i, j) -= factor * (pressure.at(i + 1, j) - p);
    v.at(i, j) -= factor * (pressure.at(i, j + 1) - p);
  }
}

} // namespace

auto calculate_divergence(GridView divergence, GridView u, GridView v) -> void {
  const auto [blocks, threads] {utils::execution_configuration(divergence.width, divergence.height, 16)};
  calculate_divergence_kernel<<<blocks, threads>>>(divergence, u, v);
  utils::check_async_cuda_error();
}

auto solve_pressure(GridView pressure_next, GridView pressure, GridView divergence) -> void {
  const auto [blocks, threads] {utils::execution_configuration(pressure.width, pressure.height, 16)};
  solve_pressure_kernel<<<blocks, threads>>>(pressure_next, pressure, divergence);
  utils::check_async_cuda_error();
}

auto project(GridView u, GridView v, GridView pressure) -> void {
  const auto [blocks, threads] {utils::execution_configuration(pressure.width, pressure.height, 16)};
  project_kernel<<<blocks, threads>>>(u, v, pressure);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

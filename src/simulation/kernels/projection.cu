#include "simulation/kernels/projection.cuh"

#include "simulation/kernels/parameters.cuh"
#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto calculate_divergence_kernel(GridView<f32> divergence, GridView<float2> velocity) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= divergence.width && j <= divergence.height) {
    const auto vel {velocity.ro(i, j)};
    divergence.at(i, j) = 0.25f * density_over_dt * (vel.x - velocity.ro(i - 1, j).x + vel.y - velocity.ro(i, j - 1).y);
  }
}

__global__ auto solve_pressure_kernel(GridView<f32> pressure_next, GridView<f32> pressure, GridView<f32> divergence, u32 cache_size) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  const auto l_i {threadIdx.x + 1};
  const auto l_j {threadIdx.y + 1};

  extern __shared__ f32 cache[];

  // Plus three instead of plus two to avoid shared memory bank conflicts
  const auto idx {[&](u32 i, u32 j) { return i + j * (cache_size + 3); }};

  if (i <= pressure.width && j <= pressure.height) {
    cache[idx(l_i, l_j)] = pressure.ro(i, j);

    if (l_i == 1) {
      cache[idx(l_i - 1, l_j)] = pressure.ro(i - 1, j);
    }
    if (l_i == cache_size || i == pressure.width) {
      cache[idx(l_i + 1, l_j)] = pressure.ro(i + 1, j);
    }
    if (l_j == 1) {
      cache[idx(l_i, l_j - 1)] = pressure.ro(i, j - 1);
    }
    if (l_j == cache_size || j == pressure.height) {
      cache[idx(l_i, l_j + 1)] = pressure.ro(i, j + 1);
    }

    __syncthreads();

    const auto center {cache[idx(l_i, l_j)]};
    const auto right {cache[idx(l_i + 1, l_j)]};
    const auto left {cache[idx(l_i - 1, l_j)]};
    const auto up {cache[idx(l_i, l_j + 1)]};
    const auto down {cache[idx(l_i, l_j - 1)]};

    const auto sum {right + left + up + down};

    pressure_next.at(i, j) = (1.0f - jacobi_weight) * center + jacobi_weight * (0.25f * sum - divergence.ro(i, j));
  }
}

__global__ auto project_kernel(GridView<float2> velocity, GridView<f32> pressure) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= pressure.width && j <= pressure.height) {
    const auto p {pressure.ro(i, j)};
    const auto vel {velocity.ro(i, j)};

    // This kernel 'incorrectly' sets some boundary values for the velocity.
    // However, they get corrected when updating the boundary values.
    velocity.at(i, j) = make_float2(vel.x - dt_over_density * (pressure.ro(i + 1, j) - p), //
                                    vel.y - dt_over_density * (pressure.ro(i, j + 1) - p));
  }
}

} // namespace

auto calculate_divergence(GridView<f32> divergence, GridView<float2> velocity) -> void {
  const auto [blocks, threads] {utils::execution_configuration(divergence.width, divergence.height, 16)};
  calculate_divergence_kernel<<<blocks, threads>>>(divergence, velocity);
  utils::check_async_cuda_error();
}

auto solve_pressure(GridView<f32> pressure_next, GridView<f32> pressure, GridView<f32> divergence) -> void {
  const auto cache_size {16u};
  const auto [blocks, threads] {utils::execution_configuration(pressure.width, pressure.height, 16)};
  solve_pressure_kernel<<<blocks, threads, (cache_size + 3) * (cache_size + 2) * sizeof(f32)>>>(pressure_next, pressure, divergence, cache_size);
  utils::check_async_cuda_error();
}

auto project(GridView<float2> velocity, GridView<f32> pressure) -> void {
  const auto [blocks, threads] {utils::execution_configuration(pressure.width, pressure.height, 16)};
  project_kernel<<<blocks, threads>>>(velocity, pressure);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

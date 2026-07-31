#include "simulation/kernels/vorticity.cuh"

#include "simulation/kernels/parameters.cuh"
#include "utils/cuda.hpp"
#include <cmath>

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto calculate_vorticity_kernel(GridView<f32> vorticity, GridView<float2> velocity) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= vorticity.width && j <= vorticity.height) {
    vorticity.at(i, j) = 0.5f * ((velocity.ro(i + 1, j).y - velocity.ro(i - 1, j).y) - (velocity.ro(i, j + 1).x - velocity.ro(i, j - 1).x));
  }
}

__global__ auto apply_vorticity_confinement_kernel(GridView<float2> velocity, GridView<f32> vorticity) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  // This kernel 'incorrectly' sets some boundary values for the velocity.
  // However, they get corrected when updating the boundary values.
  if (i <= velocity.width && j <= velocity.height) {
    const auto vel {velocity.at(i, j)};
    const auto vort {vorticity.ro(i, j)};
    const auto conf {make_float2(fabsf(vorticity.ro(i + 1, j)) - fabsf(vort), //
                                 fabsf(vorticity.ro(i, j + 1)) - fabsf(vort))};
    const auto denom {1.0f / (hypotf(conf.x, conf.y) + 0.01f)};

    velocity.at(i, j) = make_float2(vel.x + dt_over_density * confinement * denom * conf.y * vort, //
                                    vel.y - dt_over_density * confinement * denom * conf.x * vort);
  }
}

} // namespace

auto calculate_vorticity(GridView<f32> vorticity, GridView<float2> velocity, u32 block_size) -> void {
  const auto [blocks, threads] {utils::execution_configuration(vorticity.width, vorticity.height, block_size)};
  calculate_vorticity_kernel<<<blocks, threads>>>(vorticity, velocity);
  utils::check_async_cuda_error();
}

auto apply_vorticity_confinement(GridView<float2> velocity, GridView<f32> vorticity, u32 block_size) -> void {
  const auto [blocks, threads] {utils::execution_configuration(velocity.width, velocity.height, block_size)};
  apply_vorticity_confinement_kernel<<<blocks, threads>>>(velocity, vorticity);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

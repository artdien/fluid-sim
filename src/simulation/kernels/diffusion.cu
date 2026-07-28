#include "simulation/kernels/diffusion.cuh"

#include "simulation/kernels/parameters.cuh"
#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto diffuse_velocity_kernel(GridView<float2> velocity_next, GridView<float2> velocity) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  // This kernel 'incorrectly' sets some boundary values for the velocity.
  // However, they get corrected when updating the boundary values.
  if (i <= velocity.width && j <= velocity.height) {
    const auto center {velocity.ro(i, j)};
    const auto right {velocity.ro(i + 1, j)};
    const auto left {velocity.ro(i - 1, j)};
    const auto up {velocity.ro(i, j + 1)};
    const auto down {velocity.ro(i, j - 1)};

    const auto stencil {make_float2(right.x + left.x + up.x + down.x - 4.0f * center.x, //
                                    right.y + left.y + up.y + down.y - 4.0f * center.y)};

    velocity_next.at(i, j) = make_float2(center.x + dt * viscosity * stencil.x, center.y + dt * viscosity * stencil.y);
  }
}

__global__ auto diffuse_dye_kernel(GridView<f32> dye_next, GridView<f32> dye) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= dye.width && j <= dye.height) {
    const auto center {dye.ro(i, j)};
    const auto stencil {dye.ro(i + 1, j) + dye.ro(i - 1, j) + dye.ro(i, j + 1) + dye.ro(i, j - 1) - 4.0f * center};

    dye_next.at(i, j) = center + dt * viscosity_dye * stencil;
  }
}

} // namespace

auto diffuse_velocity(GridView<float2> velocity_next, GridView<float2> velocity) -> void {
  const auto [blocks, threads] {utils::execution_configuration(velocity.width, velocity.height, 16)};
  diffuse_velocity_kernel<<<blocks, threads>>>(velocity_next, velocity);
  utils::check_async_cuda_error();
}

auto diffuse_dye(GridView<f32> dye_next, GridView<f32> dye) -> void {
  const auto [blocks, threads] {utils::execution_configuration(dye.width, dye.height, 16)};
  diffuse_dye_kernel<<<blocks, threads>>>(dye_next, dye);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

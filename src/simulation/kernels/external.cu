#include "simulation/kernels/external.cuh"

#include "simulation/kernels/parameters.cuh"
#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto add_external_force_kernel(GridView<float2> velocity, f32 position_x, f32 position_y, f32 force_x, f32 force_y) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  // This kernel 'incorrectly' sets some boundary values for the velocity.
  // However, they get corrected when updating the boundary values.
  if (i <= velocity.width && j <= velocity.height) {
    const auto dx {i - position_x};
    const auto dy {j - position_y};
    const auto r2 {dx * dx + dy * dy};
    const auto radius {external_force_radius};

    if (r2 < radius * radius) {
      const auto linear_falloff {1.0f - (sqrtf(r2) / radius)};
      const auto vel {velocity.at(i, j)};

      velocity.at(i, j) = make_float2(vel.x + dt * density_inverse * linear_falloff * force_x, //
                                      vel.y + dt * density_inverse * linear_falloff * force_y);
    }
  }
}

__global__ auto add_external_dye_kernel(GridView<f32> dye, f32 position_x, f32 position_y, f32 value) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= dye.width && j <= dye.height) {
    const auto dx {i - position_x};
    const auto dy {j - position_y};
    const auto r2 {dx * dx + dy * dy};
    const auto radius {external_dye_radius};

    if (r2 < radius * radius) {
      const auto linear_falloff {1.0f - (sqrtf(r2) / radius)};

      dye.at(i, j) = fminf(1.0f, dye.at(i, j) + dt * linear_falloff * value);
    }
  }
}

} // namespace

auto add_external_force(GridView<float2> velocity, f32 position_x, f32 position_y, f32 force_x, f32 force_y, u32 block_size) -> void {
  const auto [blocks, threads] {utils::execution_configuration(velocity.width, velocity.height, block_size)};
  add_external_force_kernel<<<blocks, threads>>>(velocity, position_x, position_y, force_x, force_y);
  utils::check_async_cuda_error();
}

auto add_external_dye(GridView<f32> dye, f32 position_x, f32 position_y, f32 value, u32 block_size) -> void {
  const auto [blocks, threads] {utils::execution_configuration(dye.width, dye.height, block_size)};
  add_external_dye_kernel<<<blocks, threads>>>(dye, position_x, position_y, value);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

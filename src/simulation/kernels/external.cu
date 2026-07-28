#include "simulation/kernels/external.cuh"

#include "simulation/kernels/parameters.cuh"
#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto add_external_force_kernel(GridView<float2> velocity, f32 position_x, f32 position_y, f32 force_x, f32 force_y, f32 radius) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  // This kernel 'incorrectly' sets some boundary values for the velocity.
  // However, they get corrected when updating the boundary values.
  if (i <= velocity.width && j <= velocity.height) {
    const auto dx {i - position_x};
    const auto dy {j - position_y};
    const auto r2 {dx * dx + dy * dy};

    if (r2 < radius * radius) {
      const auto density_inverse {1.0f / density};
      const auto linear_falloff {1.0f - (sqrtf(r2) / radius)};
      const auto vel {velocity.at(i, j)};

      velocity.at(i, j) = make_float2(vel.x + dt * density_inverse * linear_falloff * force_x, //
                                      vel.y + dt * density_inverse * linear_falloff * force_y);
    }
  }
}

} // namespace

auto add_external_force(GridView<float2> velocity, f32 position_x, f32 position_y, f32 force_x, f32 force_y, f32 radius) -> void {
  const auto [blocks, threads] {utils::execution_configuration(velocity.width, velocity.height, 16)};
  add_external_force_kernel<<<blocks, threads>>>(velocity, position_x, position_y, force_x, force_y, radius);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

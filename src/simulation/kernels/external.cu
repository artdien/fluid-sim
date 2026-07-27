#include "simulation/kernels/external.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto add_external_force_kernel(GridView u, GridView v, f32 position_x, f32 position_y, f32 force_x, f32 force_y, f32 radius, f32 dt, f32 density)
    -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= u.width && j <= u.height) {
    const auto dx {i - position_x};
    const auto dy {j - position_y};
    const auto r2 {dx * dx + dy * dy};

    if (r2 < radius * radius) {
      const f32 density_inverse {1.0f / density};
      const f32 linear_falloff {1.0f - (sqrtf(r2) / radius)};

      if (i <= u.width - 1) {
        u.at(i, j) += dt * density_inverse * linear_falloff * force_x;
      }
      if (j <= v.height - 1) {
        v.at(i, j) += dt * density_inverse * linear_falloff * force_y;
      }
    }
  }
}

} // namespace

auto add_external_force(GridView u, GridView v, f32 position_x, f32 position_y, f32 force_x, f32 force_y, f32 radius, f32 dt, f32 density) -> void {
  const auto [blocks, threads] {utils::execution_configuration(u.width, u.height, 16)};
  add_external_force_kernel<<<blocks, threads>>>(u, v, position_x, position_y, force_x, force_y, radius, dt, density);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

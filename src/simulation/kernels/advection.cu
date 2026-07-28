#include "simulation/kernels/advection.cuh"

#include "simulation/kernels/parameters.cuh"
#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__device__ __forceinline__ f32 bilerp(f32 g_00, f32 g_01, f32 g_10, f32 g_11, f32 t_x, f32 t_y) {
  return (1.0f - t_x) * ((1.0f - t_y) * g_00 + t_y * g_01) + t_x * ((1.0f - t_y) * g_10 + t_y * g_11);
}

__global__ auto advect_velocity_kernel(GridView<float2> velocity_next, GridView<float2> velocity) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  // This kernel 'incorrectly' sets some boundary values for the velocity.
  // However, they get corrected when updating the boundary values.
  if (i <= velocity.width && j <= velocity.height) {
    const auto vel {velocity.at(i, j)};
    const auto dt_u {dt * vel.x};
    const auto dt_v {dt * vel.y};

    const auto x_u {fminf(fmaxf(i - dt_u, 0.001f), static_cast<f32>(velocity.width))};
    const auto y_u {fminf(fmaxf(j - 0.5f - dt_v, 0.0f), static_cast<f32>(velocity.height))};
    const auto i_u {ceilf(x_u)}; // since x >= 0.001, i_ is at least 1.
    const auto j_u {ceilf(y_u + 0.5f)};

    const auto velocity_u {bilerp(velocity.at(i_u - 1, j_u - 1).x, velocity.at(i_u - 1, j_u).x, velocity.at(i_u, j_u - 1).x, velocity.at(i_u, j_u).x,
                                  x_u - (i_u - 1.0f), y_u - (j_u - 1.5f))};

    const auto x_v = fminf(fmaxf(i - 0.5f - dt_u, 0.0f), static_cast<f32>(velocity.width));
    const auto y_v = fminf(fmaxf(j - dt_v, 0.001f), static_cast<f32>(velocity.height));
    const auto i_v = ceilf(x_v + 0.5f);
    const auto j_v = ceilf(y_v); // since y >= 0.001, j_ is at least 1.

    const auto velocity_v {bilerp(velocity.at(i_v - 1, j_v - 1).y, velocity.at(i_v - 1, j_v).y, velocity.at(i_v, j_v - 1).y, velocity.at(i_v, j_v).y,
                                  x_v - (i_v - 1.5f), y_v - (j_v - 1.0f))};

    velocity_next.at(i, j) = make_float2(velocity_u, velocity_v);
  }
}

__global__ auto advect_dye_kernel(GridView<f32> dye_next, GridView<f32> dye, GridView<float2> velocity) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= dye.width && j <= dye.height) {
    const auto vel {velocity.at(i, j)};

    const auto x_d {fminf(fmaxf(i - 0.5f - dt * vel.x, 0.0f), static_cast<f32>(dye.width))};
    const auto y_d {fminf(fmaxf(j - 0.5f - dt * vel.y, 0.0f), static_cast<f32>(dye.height))};
    const auto i_d {ceilf(x_d + 0.5f)};
    const auto j_d {ceilf(y_d + 0.5f)};

    dye_next.at(i, j) = bilerp(dye.at(i_d - 1, j_d - 1), dye.at(i_d - 1, j_d), dye.at(i_d, j_d - 1), dye.at(i_d, j_d), x_d - (i_d - 1.5f), y_d - (j_d - 1.5f));
  }
}

} // namespace

auto advect_velocity(GridView<float2> velocity_next, GridView<float2> velocity) -> void {
  const auto [blocks, threads] {utils::execution_configuration(velocity.width, velocity.height, 16)};
  advect_velocity_kernel<<<blocks, threads>>>(velocity_next, velocity);
  utils::check_async_cuda_error();
}

auto advect_dye(GridView<f32> dye_next, GridView<f32> dye, GridView<float2> velocity) -> void {
  const auto [blocks, threads] {utils::execution_configuration(dye.width, dye.height, 16)};
  advect_dye_kernel<<<blocks, threads>>>(dye_next, dye, velocity);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

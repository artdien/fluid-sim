#include "simulation/kernels/advection.cuh"

#include "simulation/kernels/parameters.cuh"
#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__device__ __forceinline__ auto bilerp(f32 g_00, f32 g_01, f32 g_10, f32 g_11, //
                                       f32 x, f32 x_l, f32 x_r,                //
                                       f32 y, f32 y_b, f32 y_t) -> f32 {       //
  return (x_r - x) * (y_t - y) * g_00 +                                        //
         (x_r - x) * (y - y_b) * g_01 +                                        //
         (x - x_l) * (y_t - y) * g_10 +                                        //
         (x - x_l) * (y - y_b) * g_11;                                         //
}

__global__ auto advect_u_kernel(GridView u_next, GridView u, GridView v) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= u.width && j <= u.height) {
    const auto x {fminf(fmaxf(i - dt * u.at(i, j), 0.001f), static_cast<f32>(u.width))};
    const auto y {fminf(fmaxf(j - 0.5f - dt * v.at(i, j), 0.0f), static_cast<f32>(u.height))};
    const auto i_ {ceilf(x)}; // since x >= 0.001, i_ is at least 1.
    const auto j_ {ceilf(y + 0.5f)};

    u_next.at(i, j) = bilerp(u.at(i_ - 1, j_ - 1), u.at(i_ - 1, j_), u.at(i_, j_ - 1), u.at(i_, j_), //
                             x, i_ - 1.0f, i_,                                                       //
                             y, j_ - 1.5f, j_ - 0.5f);                                               //
  }
}

__global__ auto advect_v_kernel(GridView v_next, GridView u, GridView v) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= v.width && j <= v.height) {
    const auto x {fminf(fmaxf(i - 0.5f - dt * u.at(i, j), 0.0f), static_cast<f32>(v.width))};
    const auto y {fminf(fmaxf(j - dt * v.at(i, j), 0.001f), static_cast<f32>(v.height))};
    const auto i_ {ceilf(x + 0.5f)};
    const auto j_ {ceilf(y)}; // since y >= 0.001, j_ is at least 1.

    v_next.at(i, j) = bilerp(v.at(i_ - 1, j_ - 1), v.at(i_ - 1, j_), v.at(i_, j_ - 1), v.at(i_, j_), //
                             x, i_ - 1.5f, i_ - 0.5f,                                                //
                             y, j_ - 1.0f, j_);                                                      //
  }
}

__global__ auto advect_dye_kernel(GridView dye_next, GridView dye, GridView u, GridView v) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= dye.width && j <= dye.height) {
    const auto x {fminf(fmaxf(i - 0.5f - dt * u.at(i, j), 0.0f), static_cast<f32>(dye.width))};
    const auto y {fminf(fmaxf(j - 0.5f - dt * v.at(i, j), 0.0f), static_cast<f32>(dye.height))};
    const auto i_ {ceilf(x + 0.5f)};
    const auto j_ {ceilf(y + 0.5f)};

    dye_next.at(i, j) = bilerp(dye.at(i_ - 1, j_ - 1), dye.at(i_ - 1, j_), dye.at(i_, j_ - 1), dye.at(i_, j_), //
                               x, i_ - 1.5f, i_ - 0.5f,                                                        //
                               y, j_ - 1.5f, j_ - 0.5f);                                                       //
  }
}

} // namespace

auto advect_u(GridView u_next, GridView u, GridView v) -> void {
  const auto [blocks, threads] {utils::execution_configuration(u.width, u.height, 16)};
  advect_u_kernel<<<blocks, threads>>>(u_next, u, v);
  utils::check_async_cuda_error();
}

auto advect_v(GridView v_next, GridView u, GridView v) -> void {
  const auto [blocks, threads] {utils::execution_configuration(v.width, v.height, 16)};
  advect_v_kernel<<<blocks, threads>>>(v_next, u, v);
  utils::check_async_cuda_error();
}

auto advect_dye(GridView dye_next, GridView dye, GridView u, GridView v) -> void {
  const auto [blocks, threads] {utils::execution_configuration(dye.width, dye.height, 16)};
  advect_dye_kernel<<<blocks, threads>>>(dye_next, dye, u, v);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

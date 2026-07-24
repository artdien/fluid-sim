#include "simulation/kernels/advection.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__device__ __forceinline__ auto bilerp(f32 g_00, f32 g_01, f32 g_10, f32 g_11, //
                                       f32 x_i, f32 x_l, f32 x_r,              //
                                       f32 y_j, f32 y_b, f32 y_t) -> f32 {     //
  return (x_r - x_i) * (y_t - y_j) * g_00 +                                    //
         (x_r - x_i) * (y_j - y_b) * g_01 +                                    //
         (x_i - x_l) * (y_t - y_j) * g_10 +                                    //
         (x_i - x_l) * (y_j - y_b) * g_11;                                     //
}

__global__ auto advect_kernel(GridView dye_next, GridView dye, GridView u, GridView v, f32 dt) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= dye.width && j <= dye.height) {
    const auto x {fminf(fmaxf(i - 0.5f - dt * u.at(i, j), 0.0f), static_cast<f32>(dye.width))};
    const auto y {fminf(fmaxf(j - 0.5f - dt * v.at(i, j), 0.0f), static_cast<f32>(dye.height))};
    const auto i_ {ceilf(x)};
    const auto j_ {ceilf(y)};

    dye_next.at(i, j) =
        bilerp(dye.at(i_ - 1, j_ - 1), dye.at(i_, j_ - 1), dye.at(i_ - 1, j_), dye.at(i_, j_), x, i_ - 1.5f, i_ - 0.5f, y, j_ - 1.5f, j_ - 0.5f);
  }
}

} // namespace

auto advect(GridView dye_next, GridView dye, GridView u, GridView v, f32 dt) -> void {
  const auto block_size {16u};
  const auto blocks {dim3(static_cast<u32>(std::ceil((dye_next.width) / static_cast<f32>(block_size))),
                          static_cast<u32>(std::ceil((dye_next.height) / static_cast<f32>(block_size))))};
  const auto threads {dim3(block_size, block_size)};

  advect_kernel<<<blocks, threads>>>(dye_next, dye, u, v, dt);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

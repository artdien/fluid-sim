#include "simulation/kernels/diffusion.cuh"

#include <cmath>

#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto diffuse_kernel(GridView dye_next, GridView dye, f32 dt, f32 viscosity) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= dye.width && j <= dye.height) {
    const auto center {dye.at(i, j)};
    const auto stencil {dye.at(i + 1, j) + dye.at(i - 1, j) + dye.at(i, j + 1) + dye.at(i, j - 1) - 4.0f * center};

    dye_next.at(i, j) = center + dt * viscosity * stencil;
  }
}

} // namespace

auto diffuse(GridView dye_next, GridView dye, f32 dt, f32 viscosity) -> void {
  const auto block_size {16u};
  const auto blocks {dim3(static_cast<u32>(std::ceil((dye_next.width) / static_cast<f32>(block_size))),
                          static_cast<u32>(std::ceil((dye_next.height) / static_cast<f32>(block_size))))};
  const auto threads {dim3(block_size, block_size)};

  diffuse_kernel<<<blocks, threads>>>(dye_next, dye, dt, viscosity);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

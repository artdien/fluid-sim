#include "buffer/kernels/update_framebuffer.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::buffer::kernels {

namespace {

__global__ auto update_framebuffer_kernel(cudaSurfaceObject_t surface, simulation::GridView grid) -> void {
  const u32 i {threadIdx.x + blockIdx.x * blockDim.x};
  const u32 j {threadIdx.y + blockIdx.y * blockDim.y};

  if (i < grid.width && j < grid.height) {
    auto color {make_float4(grid.at(i, j), grid.at(i, j), grid.at(i, j), 1.0f)};
    surf2Dwrite(color, surface, i * sizeof(float4), j);
  }
}

} // namespace

auto update_framebuffer(cudaSurfaceObject_t surface, simulation::GridView grid) -> void {
  update_framebuffer_kernel<<<dim3(grid.width / 16u + 1u, grid.height / 16u + 1u), dim3(16u, 16u)>>>(surface, grid);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::buffer::kernels

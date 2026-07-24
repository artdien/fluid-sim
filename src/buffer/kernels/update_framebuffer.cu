#include "buffer/kernels/update_framebuffer.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::buffer::kernels {

namespace {

__global__ auto update_framebuffer_kernel(cudaSurfaceObject_t surface, simulation::GridView grid) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= grid.width && j <= grid.height) {
    auto color {make_float4(grid.at(i, j), grid.at(i, j), grid.at(i, j), 1.0f)};
    surf2Dwrite(color, surface, (i - 1) * sizeof(float4), (j - 1));
  }
}

} // namespace

auto update_framebuffer(cudaSurfaceObject_t surface, simulation::GridView grid) -> void {
  constexpr auto block_size {16u};
  const auto blocks {dim3(static_cast<u32>(std::ceil((grid.width) / static_cast<f32>(block_size))),
                          static_cast<u32>(std::ceil((grid.height) / static_cast<f32>(block_size))))};
  const auto threads {dim3(block_size, block_size)};

  update_framebuffer_kernel<<<blocks, threads>>>(surface, grid);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::buffer::kernels

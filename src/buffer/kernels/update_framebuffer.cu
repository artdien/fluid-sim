#include "buffer/kernels/update_framebuffer.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::buffer::kernels {

namespace {

__global__ auto update_framebuffer_kernel(cudaSurfaceObject_t surface, simulation::GridView<float4> grid) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= grid.width && j <= grid.height) {
    const auto value {grid.ro(i, j)};
    const auto color {make_float4(__saturatef(value.x), __saturatef(value.y), __saturatef(value.z), 1.0f)};
    surf2Dwrite(color, surface, (i - 1) * sizeof(float4), (j - 1));
  }
}

} // namespace

auto update_framebuffer(cudaSurfaceObject_t surface, simulation::GridView<float4> grid) -> void {
  static const auto block_size {[] {
    i32 _, block_size_1D;
    utils::check_cuda_error(cudaOccupancyMaxPotentialBlockSize(&_, &block_size_1D, update_framebuffer_kernel));
    return block_size_1D >= 256 ? 16u : 8u;
  }()};

  const auto [blocks, threads] {utils::execution_configuration(grid.width, grid.height, block_size)};
  update_framebuffer_kernel<<<blocks, threads>>>(surface, grid);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::buffer::kernels

#include "buffer/kernels/update_framebuffer.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::buffer::kernels {

namespace {

__global__ auto update_framebuffer_kernel(cudaSurfaceObject_t surface, simulation::GridView<f32> grid) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= grid.width && j <= grid.height) {
    auto color {make_float4(__saturatef(grid.ro(i, j)), __saturatef(grid.ro(i, j)), __saturatef(grid.ro(i, j)), 1.0f)};
    surf2Dwrite(color, surface, (i - 1) * sizeof(float4), (j - 1));
  }
}

} // namespace

auto update_framebuffer(cudaSurfaceObject_t surface, simulation::GridView<f32> grid) -> void {
  const auto [blocks, threads] {utils::execution_configuration(grid.width, grid.height, 16)};
  update_framebuffer_kernel<<<blocks, threads>>>(surface, grid);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::buffer::kernels

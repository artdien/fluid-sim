#include "simulation/kernels/dummy.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto dummy_kernel(GridView grid) -> void {
  const u32 i {threadIdx.x + blockIdx.x * blockDim.x};
  const u32 j {threadIdx.y + blockIdx.y * blockDim.y};

  if (i < grid.width && j < grid.height) {
    grid.at(i, j) = 1.0f;
  }
}

} // namespace

auto dummy(GridView grid) -> void {
  dummy_kernel<<<dim3(grid.width / 16u + 1u, grid.height / 16u + 1u), dim3(16, 16)>>>(grid);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

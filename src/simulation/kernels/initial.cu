#include "simulation/kernels/initial.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto initialize_horizontal_split_kernel(GridView dye) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= dye.width && j <= dye.height) {
    if (const auto split {dye.width / 2u}; i < split) {
      dye.at(i, j) = 1.0f;
    }
  }
}

} // namespace

auto initialize_horizontal_split(GridView dye) -> void {
  const auto block_size {16u};
  const auto blocks {dim3(static_cast<u32>(std::ceil((dye.width) / static_cast<f32>(block_size))), //
                          static_cast<u32>(std::ceil((dye.height) / static_cast<f32>(block_size))))};
  const auto threads {dim3(block_size, block_size)};

  initialize_horizontal_split_kernel<<<blocks, threads>>>(dye);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

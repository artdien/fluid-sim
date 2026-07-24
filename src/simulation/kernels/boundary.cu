
#include "simulation/kernels/boundary.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto update_boundary_columns_kernel(GridView dye) -> void {
  const auto j {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (j <= dye.height) {
    dye.at(0, j) = dye.at(1, j);
    dye.at(dye.width + 1, j) = dye.at(dye.width, j);
  }

  // Corner cells
  if (j == dye.height) {
    dye.at(0, 0) = dye.at(1, 1);                                           // bottom left
    dye.at(dye.width + 1, 0) = dye.at(dye.width, 1);                       // bottom right
    dye.at(0, dye.height + 1) = dye.at(1, dye.height);                     // top left
    dye.at(dye.width + 1, dye.height + 1) = dye.at(dye.width, dye.height); // top right
  }
}

__global__ auto update_boundary_rows_kernel(GridView dye) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (i <= dye.width) {
    dye.at(i, 0) = dye.at(i, 1);
    dye.at(i, dye.height + 1) = dye.at(i, dye.height);
  }
}

} // namespace

auto update_boundary(GridView dye) -> void {
  const auto block_size {16u};
  const auto blocks_height {dim3(static_cast<u32>(std::ceil((dye.height) / static_cast<f32>(block_size))))};
  const auto blocks_width {dim3(static_cast<u32>(std::ceil((dye.width) / static_cast<f32>(block_size))))};
  const auto threads {dim3(block_size)};

  update_boundary_columns_kernel<<<blocks_height, threads>>>(dye);
  utils::check_async_cuda_error();

  update_boundary_rows_kernel<<<blocks_width, threads>>>(dye);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

#include "simulation/kernels/initial.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto initialize_horizontal_split_kernel(GridView<f32> dye) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= dye.width && j <= dye.height) {
    dye.at(i, j) = 0.0f;

    if (const auto split {dye.width / 2u}; i < split) {
      dye.at(i, j) = 1.0f;
    }
  }
}

__global__ auto initialize_vertical_split_kernel(GridView<f32> dye) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= dye.width && j <= dye.height) {
    dye.at(i, j) = 0.0f;

    if (const auto split {dye.height / 2u}; j >= split) {
      dye.at(i, j) = 1.0f;
    }
  }
}

__global__ auto initialize_empty_kernel(GridView<f32> dye) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= dye.width && j <= dye.height) {
    dye.at(i, j) = 0.0f;
  }
}

} // namespace

auto initialize_horizontal_split(GridView<f32> dye, u32 block_size) -> void {
  const auto [blocks, threads] {utils::execution_configuration(dye.width, dye.height, block_size)};
  initialize_horizontal_split_kernel<<<blocks, threads>>>(dye);
  utils::check_async_cuda_error();
}

auto initialize_vertical_split(GridView<f32> dye, u32 block_size) -> void {
  const auto [blocks, threads] {utils::execution_configuration(dye.width, dye.height, block_size)};
  initialize_vertical_split_kernel<<<blocks, threads>>>(dye);
  utils::check_async_cuda_error();
}

auto initialize_empty(GridView<f32> dye, u32 block_size) -> void {
  const auto [blocks, threads] {utils::execution_configuration(dye.width, dye.height, block_size)};
  initialize_empty_kernel<<<blocks, threads>>>(dye);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

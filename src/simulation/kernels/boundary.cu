#include "simulation/kernels/boundary.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto update_velocity_boundary_rows_kernel(GridView<float2> velocity) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (i <= velocity.width) {
    if (i <= velocity.width - 1) {
      velocity.at(i, 0).x = velocity.at(i, 1).x;
      velocity.at(i, velocity.height + 1).x = velocity.at(i, velocity.height).x;
    }

    velocity.at(i, 0).y = 0.0f;
    velocity.at(i, velocity.height).y = 0.0f;
  }
}

__global__ auto update_pressure_boundary_rows_kernel(GridView<f32> pressure) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (i <= pressure.width) {
    pressure.at(i, 0) = pressure.at(i, 1);
    pressure.at(i, pressure.height + 1) = pressure.at(i, pressure.height);
  }
}

__global__ auto update_dye_boundary_rows_kernel(GridView<f32> dye) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (i <= dye.width) {
    dye.at(i, 0) = dye.at(i, 1);
    dye.at(i, dye.height + 1) = dye.at(i, dye.height);
  }
}

__global__ auto update_velocity_boundary_columns_kernel(GridView<float2> velocity) -> void {
  const auto j {threadIdx.x + blockIdx.x * blockDim.x + 1};
  if (j <= velocity.height) {
    velocity.at(0, j).x = 0.0f;
    velocity.at(velocity.width, j).x = 0.0f;

    if (j <= velocity.height - 1) {
      velocity.at(0, j).y = velocity.at(1, j).y;
      velocity.at(velocity.width + 1, j).y = velocity.at(velocity.width, j).y;
    }

    // Corner cells
    if (j == 1) {
      velocity.at(0, 0).x = 0.5f * velocity.at(1, 1).x;                                                               // bottom left
      velocity.at(velocity.width, 0).x = 0.5f * velocity.at(velocity.width - 1, 1).x;                                 // bottom right
      velocity.at(0, velocity.height + 1).x = 0.5f * velocity.at(1, velocity.height).x;                               // top left
      velocity.at(velocity.width, velocity.height + 1).x = 0.5f * velocity.at(velocity.width - 1, velocity.height).x; // top right

      velocity.at(0, 0).y = 0.5f * velocity.at(1, 1).y;                                                                   // bottom left
      velocity.at(velocity.width + 1, 0).y = 0.5f * velocity.at(velocity.width, 1).y;                                     // bottom right
      velocity.at(0, velocity.height).y = 0.5f * velocity.at(1, velocity.height - 1).y;                                   // top left
      velocity.at(velocity.width + 1, velocity.height).y = 0.5f * velocity.at(velocity.width + 1, velocity.height - 1).y; // top right
    }
  }
}

__global__ auto update_pressure_boundary_columns_kernel(GridView<f32> pressure) -> void {
  const auto j {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (j <= pressure.height) {
    pressure.at(0, j) = pressure.at(1, j);
    pressure.at(pressure.width + 1, j) = pressure.at(pressure.width, j);
  }

  // Corner cells
  if (j == 1) {
    pressure.at(0, 0) = pressure.at(1, 1);                                                               // bottom left
    pressure.at(pressure.width + 1, 0) = pressure.at(pressure.width, 1);                                 // bottom right
    pressure.at(0, pressure.height + 1) = pressure.at(1, pressure.height);                               // top left
    pressure.at(pressure.width + 1, pressure.height + 1) = pressure.at(pressure.width, pressure.height); // top right
  }
}

__global__ auto update_dye_boundary_columns_kernel(GridView<f32> dye) -> void {
  const auto j {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (j <= dye.height) {
    dye.at(0, j) = dye.at(1, j);
    dye.at(dye.width + 1, j) = dye.at(dye.width, j);
  }

  // Corner cells
  if (j == 1) {
    dye.at(0, 0) = dye.at(1, 1);                                           // bottom left
    dye.at(dye.width + 1, 0) = dye.at(dye.width, 1);                       // bottom right
    dye.at(0, dye.height + 1) = dye.at(1, dye.height);                     // top left
    dye.at(dye.width + 1, dye.height + 1) = dye.at(dye.width, dye.height); // top right
  }
}

} // namespace

auto update_velocity_boundary(GridView<float2> velocity) -> void {
  const auto [blocks_height, threads_height] {utils::execution_configuration(velocity.height, 16)};
  update_velocity_boundary_columns_kernel<<<blocks_height, threads_height>>>(velocity);
  utils::check_async_cuda_error();

  const auto [blocks_width, threads_width] {utils::execution_configuration(velocity.width, 16)};
  update_velocity_boundary_rows_kernel<<<blocks_width, threads_width>>>(velocity);
  utils::check_async_cuda_error();
}

auto update_pressure_boundary(GridView<f32> pressure) -> void {
  const auto [blocks_height, threads_height] {utils::execution_configuration(pressure.height, 16)};
  update_pressure_boundary_columns_kernel<<<blocks_height, threads_height>>>(pressure);
  utils::check_async_cuda_error();

  const auto [blocks_width, threads_width] {utils::execution_configuration(pressure.width, 16)};
  update_pressure_boundary_rows_kernel<<<blocks_width, threads_width>>>(pressure);
  utils::check_async_cuda_error();
}

auto update_dye_boundary(GridView<f32> dye) -> void {
  const auto [blocks_height, threads_height] {utils::execution_configuration(dye.height, 16)};
  update_dye_boundary_columns_kernel<<<blocks_height, threads_height>>>(dye);
  utils::check_async_cuda_error();

  const auto [blocks_width, threads_width] {utils::execution_configuration(dye.width, 16)};
  update_dye_boundary_rows_kernel<<<blocks_width, threads_width>>>(dye);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

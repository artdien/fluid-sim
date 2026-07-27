#include "simulation/kernels/boundary.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto update_u_boundary_rows_kernel(GridView u) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (i <= u.width - 1) {
    u.at(i, 0) = u.at(i, 1);
    u.at(i, u.height + 1) = u.at(i, u.height);
  }
}

__global__ auto update_v_boundary_rows_kernel(GridView v) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (i <= v.width) {
    v.at(i, 0) = 0.0f;
    v.at(i, v.height) = 0.0f;
  }
}

__global__ auto update_pressure_boundary_rows_kernel(GridView pressure) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (i <= pressure.width) {
    pressure.at(i, 0) = pressure.at(i, 1);
    pressure.at(i, pressure.height + 1) = pressure.at(i, pressure.height);
  }
}

__global__ auto update_dye_boundary_rows_kernel(GridView dye) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (i <= dye.width) {
    dye.at(i, 0) = dye.at(i, 1);
    dye.at(i, dye.height + 1) = dye.at(i, dye.height);
  }
}

__global__ auto update_u_boundary_columns_kernel(GridView u) -> void {
  const auto j {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (j <= u.height) {
    u.at(0, j) = 0.0f;
    u.at(u.width, j) = 0.0f;
  }

  // Corner cells
  if (j == 1) {
    u.at(0, 0) = 0.5f * u.at(1, 1);                                   // bottom left
    u.at(u.width, 0) = 0.5f * u.at(u.width - 1, 1);                   // bottom right
    u.at(0, u.height + 1) = 0.5f * u.at(1, u.height);                 // top left
    u.at(u.width, u.height + 1) = 0.5f * u.at(u.width - 1, u.height); // top right
  }
}

__global__ auto update_v_boundary_columns_kernel(GridView v) -> void {
  const auto j {threadIdx.x + blockIdx.x * blockDim.x + 1};

  if (j <= v.height - 1) {
    v.at(0, j) = v.at(1, j);
    v.at(v.width + 1, j) = v.at(v.width, j);
  }

  // Corner cells
  if (j == 1) {
    v.at(0, 0) = 0.5f * v.at(1, 1);                                       // bottom left
    v.at(v.width + 1, 0) = 0.5f * v.at(v.width, 1);                       // bottom right
    v.at(0, v.height) = 0.5f * v.at(1, v.height - 1);                     // top left
    v.at(v.width + 1, v.height) = 0.5f * v.at(v.width + 1, v.height - 1); // top right
  }
}

__global__ auto update_pressure_boundary_columns_kernel(GridView pressure) -> void {
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

__global__ auto update_dye_boundary_columns_kernel(GridView dye) -> void {
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

auto update_u_boundary(GridView u) -> void {
  const auto [blocks_height, threads_height] {utils::execution_configuration(u.height, 16)};
  update_u_boundary_columns_kernel<<<blocks_height, threads_height>>>(u);
  utils::check_async_cuda_error();

  const auto [blocks_width, threads_width] {utils::execution_configuration(u.width, 16)};
  update_u_boundary_rows_kernel<<<blocks_width, threads_width>>>(u);
  utils::check_async_cuda_error();
}

auto update_v_boundary(GridView v) -> void {
  const auto [blocks_height, threads_height] {utils::execution_configuration(v.height, 16)};
  update_v_boundary_columns_kernel<<<blocks_height, threads_height>>>(v);
  utils::check_async_cuda_error();

  const auto [blocks_width, threads_width] {utils::execution_configuration(v.width, 16)};
  update_v_boundary_rows_kernel<<<blocks_width, threads_width>>>(v);
  utils::check_async_cuda_error();
}

auto update_pressure_boundary(GridView pressure) -> void {
  const auto [blocks_height, threads_height] {utils::execution_configuration(pressure.height, 16)};
  update_pressure_boundary_columns_kernel<<<blocks_height, threads_height>>>(pressure);
  utils::check_async_cuda_error();

  const auto [blocks_width, threads_width] {utils::execution_configuration(pressure.width, 16)};
  update_pressure_boundary_rows_kernel<<<blocks_width, threads_width>>>(pressure);
  utils::check_async_cuda_error();
}

auto update_dye_boundary(GridView dye) -> void {
  const auto [blocks_height, threads_height] {utils::execution_configuration(dye.height, 16)};
  update_dye_boundary_columns_kernel<<<blocks_height, threads_height>>>(dye);
  utils::check_async_cuda_error();

  const auto [blocks_width, threads_width] {utils::execution_configuration(dye.width, 16)};
  update_dye_boundary_rows_kernel<<<blocks_width, threads_width>>>(dye);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

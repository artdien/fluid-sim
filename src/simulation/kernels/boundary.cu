#include "simulation/kernels/boundary.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

enum class BoundaryType {
  ROW,
  COLUMN,
  CORNER,
};

__global__ auto update_velocity_boundary_kernel(GridView<float2> velocity, BoundaryType type) -> void {
  if (type == BoundaryType::ROW) {
    if (const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1}; i <= velocity.width) {
      velocity.at(i, 0) = make_float2(velocity.at(i, 1).x, 0.0f);

      velocity.at(i, velocity.height + 1).x = velocity.at(i, velocity.height).x;
      velocity.at(i, velocity.height).y = 0.0f;
    }
  } else if (type == BoundaryType::COLUMN) {
    if (const auto j {threadIdx.x + blockIdx.x * blockDim.x + 1}; j <= velocity.height) {
      velocity.at(0, j) = make_float2(0.0f, velocity.at(1, j).y);

      velocity.at(velocity.width + 1, j).y = velocity.at(velocity.width, j).y;
      velocity.at(velocity.width, j).x = 0.0f;
    }
  } else if (type == BoundaryType::CORNER) {
    const auto vel {velocity.at(1, 1)};

    // bottom left
    velocity.at(0, 0).x = 0.5f * vel.x;
    velocity.at(0, 0).y = 0.5f * vel.y;

    // bottom right
    velocity.at(velocity.width, 0).x = 0.5f * velocity.at(velocity.width - 1, 1).x;
    velocity.at(velocity.width + 1, 0).y = 0.5f * velocity.at(velocity.width, 1).y;

    // top left
    velocity.at(0, velocity.height + 1).x = 0.5f * velocity.at(1, velocity.height).x;
    velocity.at(0, velocity.height).y = 0.5f * velocity.at(1, velocity.height - 1).y;

    // top right
    velocity.at(velocity.width, velocity.height + 1).x = 0.5f * velocity.at(velocity.width - 1, velocity.height).x;
    velocity.at(velocity.width + 1, velocity.height).y = 0.5f * velocity.at(velocity.width + 1, velocity.height - 1).y;
  }
}

__global__ auto update_pressure_boundary_kernel(GridView<f32> pressure, BoundaryType type) -> void {
  if (type == BoundaryType::ROW) {
    if (const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1}; i <= pressure.width) {
      pressure.at(i, 0) = pressure.at(i, 1);
      pressure.at(i, pressure.height + 1) = pressure.at(i, pressure.height);
    }
  } else if (type == BoundaryType::COLUMN) {
    if (const auto j {threadIdx.x + blockIdx.x * blockDim.x + 1}; j <= pressure.height) {
      pressure.at(0, j) = pressure.at(1, j);
      pressure.at(pressure.width + 1, j) = pressure.at(pressure.width, j);
    }
  } else if (type == BoundaryType::CORNER) {
    // bottom left
    pressure.at(0, 0) = pressure.at(1, 1);
    // bottom right
    pressure.at(pressure.width + 1, 0) = pressure.at(pressure.width, 1);
    // top left
    pressure.at(0, pressure.height + 1) = pressure.at(1, pressure.height);
    // top right
    pressure.at(pressure.width + 1, pressure.height + 1) = pressure.at(pressure.width, pressure.height);
  }
}

__global__ auto update_dye_boundary_kernel(GridView<f32> dye, BoundaryType type) -> void {
  if (type == BoundaryType::ROW) {
    if (const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1}; i <= dye.width) {
      dye.at(i, 0) = dye.at(i, 1);
      dye.at(i, dye.height + 1) = dye.at(i, dye.height);
    }
  } else if (type == BoundaryType::COLUMN) {
    if (const auto j {threadIdx.x + blockIdx.x * blockDim.x + 1}; j <= dye.height) {
      dye.at(0, j) = dye.at(1, j);
      dye.at(dye.width + 1, j) = dye.at(dye.width, j);
    }
  } else if (type == BoundaryType::CORNER) {
    // bottom left
    dye.at(0, 0) = dye.at(1, 1);
    // bottom right
    dye.at(dye.width + 1, 0) = dye.at(dye.width, 1);
    // top left
    dye.at(0, dye.height + 1) = dye.at(1, dye.height);
    // top right
    dye.at(dye.width + 1, dye.height + 1) = dye.at(dye.width, dye.height);
  }
}

} // namespace

auto update_velocity_boundary(GridView<float2> velocity) -> void {
  const auto [blocks_height, threads_height] {utils::execution_configuration(velocity.height, 16)};
  update_velocity_boundary_kernel<<<blocks_height, threads_height>>>(velocity, BoundaryType::COLUMN);
  utils::check_async_cuda_error();

  const auto [blocks_width, threads_width] {utils::execution_configuration(velocity.width, 16)};
  update_velocity_boundary_kernel<<<blocks_width, threads_width>>>(velocity, BoundaryType::ROW);
  utils::check_async_cuda_error();

  update_velocity_boundary_kernel<<<1, 1>>>(velocity, BoundaryType::CORNER);
  utils::check_async_cuda_error();
}

auto update_pressure_boundary(GridView<f32> pressure) -> void {
  const auto [blocks_height, threads_height] {utils::execution_configuration(pressure.height, 16)};
  update_pressure_boundary_kernel<<<blocks_height, threads_height>>>(pressure, BoundaryType::COLUMN);
  utils::check_async_cuda_error();

  const auto [blocks_width, threads_width] {utils::execution_configuration(pressure.width, 16)};
  update_pressure_boundary_kernel<<<blocks_width, threads_width>>>(pressure, BoundaryType::ROW);
  utils::check_async_cuda_error();

  update_pressure_boundary_kernel<<<1, 1>>>(pressure, BoundaryType::CORNER);
  utils::check_async_cuda_error();
}

auto update_dye_boundary(GridView<f32> dye) -> void {
  const auto [blocks_height, threads_height] {utils::execution_configuration(dye.height, 16)};
  update_dye_boundary_kernel<<<blocks_height, threads_height>>>(dye, BoundaryType::COLUMN);
  utils::check_async_cuda_error();

  const auto [blocks_width, threads_width] {utils::execution_configuration(dye.width, 16)};
  update_dye_boundary_kernel<<<blocks_width, threads_width>>>(dye, BoundaryType::ROW);
  utils::check_async_cuda_error();

  update_dye_boundary_kernel<<<1, 1>>>(dye, BoundaryType::CORNER);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

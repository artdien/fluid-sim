#include "simulation/kernels/diffusion.cuh"

#include "simulation/kernels/parameters.cuh"
#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto diffuse_velocity_kernel(GridView<float2> velocity_next, GridView<float2> velocity, u32 cache_size) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  const auto l_i {threadIdx.x + 1};
  const auto l_j {threadIdx.y + 1};

  extern __shared__ float2 vel_cache[];

  // Plus three instead of plus two to avoid shared memory bank conflicts
  const auto idx {[&](u32 i, u32 j) { return i + j * (cache_size + 3); }};

  // This kernel 'incorrectly' sets some boundary values for the velocity.
  // However, they get corrected when updating the boundary values.
  if (i <= velocity.width && j <= velocity.height) {
    vel_cache[idx(l_i, l_j)] = velocity.ro(i, j);

    if (l_i == 1) {
      vel_cache[idx(l_i - 1, l_j)] = velocity.ro(i - 1, j);
    }
    if (l_i == cache_size || i == velocity.width) {
      vel_cache[idx(l_i + 1, l_j)] = velocity.ro(i + 1, j);
    }
    if (l_j == 1) {
      vel_cache[idx(l_i, l_j - 1)] = velocity.ro(i, j - 1);
    }
    if (l_j == cache_size || j == velocity.height) {
      vel_cache[idx(l_i, l_j + 1)] = velocity.ro(i, j + 1);
    }

    __syncthreads();

    const auto center {vel_cache[idx(l_i, l_j)]};
    const auto right {vel_cache[idx(l_i + 1, l_j)]};
    const auto left {vel_cache[idx(l_i - 1, l_j)]};
    const auto up {vel_cache[idx(l_i, l_j + 1)]};
    const auto down {vel_cache[idx(l_i, l_j - 1)]};

    const auto sum {make_float2(right.x + left.x + up.x + down.x, right.y + left.y + up.y + down.y)};

    velocity_next.at(i, j) = make_float2((1.0f - 4.0f * viscosity_times_dt) * center.x + viscosity_times_dt * sum.x,
                                         (1.0f - 4.0f * viscosity_times_dt) * center.y + viscosity_times_dt * sum.y);
  }
}

__global__ auto diffuse_dye_kernel(GridView<f32> dye_next, GridView<f32> dye, u32 cache_size) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  const auto l_i {threadIdx.x + 1};
  const auto l_j {threadIdx.y + 1};

  extern __shared__ f32 dye_cache[];

  // Plus three instead of plus two to avoid shared memory bank conflicts
  const auto idx {[&](u32 i, u32 j) { return i + j * (cache_size + 3); }};

  if (i <= dye.width && j <= dye.height) {
    dye_cache[idx(l_i, l_j)] = dye.ro(i, j);

    if (l_i == 1) {
      dye_cache[idx(l_i - 1, l_j)] = dye.ro(i - 1, j);
    }
    if (l_i == cache_size || i == dye.width) {
      dye_cache[idx(l_i + 1, l_j)] = dye.ro(i + 1, j);
    }
    if (l_j == 1) {
      dye_cache[idx(l_i, l_j - 1)] = dye.ro(i, j - 1);
    }
    if (l_j == cache_size || j == dye.height) {
      dye_cache[idx(l_i, l_j + 1)] = dye.ro(i, j + 1);
    }

    __syncthreads();

    const auto center {dye_cache[idx(l_i, l_j)]};
    const auto right {dye_cache[idx(l_i + 1, l_j)]};
    const auto left {dye_cache[idx(l_i - 1, l_j)]};
    const auto up {dye_cache[idx(l_i, l_j + 1)]};
    const auto down {dye_cache[idx(l_i, l_j - 1)]};

    const auto sum {right + left + up + down};

    dye_next.at(i, j) = (1.0f - 4.0f * viscosity_dye_times_dt) * center + viscosity_dye_times_dt * sum;
  }
}

} // namespace

auto diffuse_velocity(GridView<float2> velocity_next, GridView<float2> velocity, u32 block_size) -> void {
  const auto cache_size {block_size};
  const auto [blocks, threads] {utils::execution_configuration(velocity.width, velocity.height, block_size)};
  diffuse_velocity_kernel<<<blocks, threads, (cache_size + 3) * (cache_size + 2) * sizeof(float2)>>>(velocity_next, velocity, cache_size);
  utils::check_async_cuda_error();
}

auto diffuse_dye(GridView<f32> dye_next, GridView<f32> dye, u32 block_size) -> void {
  const auto cache_size {block_size};
  const auto [blocks, threads] {utils::execution_configuration(dye.width, dye.height, block_size)};
  diffuse_dye_kernel<<<blocks, threads, (cache_size + 3) * (cache_size + 2) * sizeof(f32)>>>(dye_next, dye, cache_size);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

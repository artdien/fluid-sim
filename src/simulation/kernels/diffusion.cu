#include "simulation/kernels/diffusion.cuh"

#include "utils/cuda.hpp"

namespace fluidsim::simulation::kernels {

namespace {

__global__ auto diffuse_u_kernel(GridView u_next, GridView u, f32 dt, f32 viscosity) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= u.width - 1 && j <= u.height) {
    const auto center {u.at(i, j)};
    const auto stencil {u.at(i + 1, j) + u.at(i - 1, j) + u.at(i, j + 1) + u.at(i, j - 1) - 4.0f * center};

    u_next.at(i, j) = center + dt * viscosity * stencil;
  }
}

__global__ auto diffuse_v_kernel(GridView v_next, GridView v, f32 dt, f32 viscosity) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= v.width && j <= v.height - 1) {
    const auto center {v.at(i, j)};
    const auto stencil {v.at(i + 1, j) + v.at(i - 1, j) + v.at(i, j + 1) + v.at(i, j - 1) - 4.0f * center};

    v_next.at(i, j) = center + dt * viscosity * stencil;
  }
}

__global__ auto diffuse_dye_kernel(GridView dye_next, GridView dye, f32 dt, f32 viscosity) -> void {
  const auto i {threadIdx.x + blockIdx.x * blockDim.x + 1};
  const auto j {threadIdx.y + blockIdx.y * blockDim.y + 1};

  if (i <= dye.width && j <= dye.height) {
    const auto center {dye.at(i, j)};
    const auto stencil {dye.at(i + 1, j) + dye.at(i - 1, j) + dye.at(i, j + 1) + dye.at(i, j - 1) - 4.0f * center};

    dye_next.at(i, j) = center + dt * viscosity * stencil;
  }
}

} // namespace

auto diffuse_u(GridView u_next, GridView u, f32 dt, f32 viscosity) -> void {
  const auto block_size {16u};
  const auto blocks {dim3(static_cast<u32>(std::ceil((u.width) / static_cast<f32>(block_size))), //
                          static_cast<u32>(std::ceil((u.height) / static_cast<f32>(block_size))))};
  const auto threads {dim3(block_size, block_size)};

  diffuse_u_kernel<<<blocks, threads>>>(u_next, u, dt, viscosity);
  utils::check_async_cuda_error();
}

auto diffuse_v(GridView v_next, GridView v, f32 dt, f32 viscosity) -> void {
  const auto block_size {16u};
  const auto blocks {dim3(static_cast<u32>(std::ceil((v.width) / static_cast<f32>(block_size))), //
                          static_cast<u32>(std::ceil((v.height) / static_cast<f32>(block_size))))};
  const auto threads {dim3(block_size, block_size)};

  diffuse_v_kernel<<<blocks, threads>>>(v_next, v, dt, viscosity);
  utils::check_async_cuda_error();
}

auto diffuse_dye(GridView dye_next, GridView dye, f32 dt, f32 viscosity) -> void {
  const auto block_size {16u};
  const auto blocks {dim3(static_cast<u32>(std::ceil((dye_next.width) / static_cast<f32>(block_size))),
                          static_cast<u32>(std::ceil((dye_next.height) / static_cast<f32>(block_size))))};
  const auto threads {dim3(block_size, block_size)};

  diffuse_dye_kernel<<<blocks, threads>>>(dye_next, dye, dt, viscosity);
  utils::check_async_cuda_error();
}

} // namespace fluidsim::simulation::kernels

#include "simulation/grid.hpp"

#include <cuda_runtime.h>

#include "utils/cuda.hpp"

namespace fluidsim::simulation {

namespace {

auto allocate_device_memory(usize size_in_bytes) -> f32* {
  f32* pointer;
  utils::check_cuda_error(cudaMalloc(&pointer, size_in_bytes));
  return pointer;
}

} // namespace

Grid::Grid(u32 width, u32 height) : width_ {width}, height_ {height} {
  const auto deleter {[](f32* pointer) { cudaFree(pointer); }};

  data_ = std::unique_ptr<f32[], CudaMemoryDeleter>(allocate_device_memory(width_ * height_ * sizeof(f32)), deleter);
  grid_ = data_.get();
}

} // namespace fluidsim::simulation

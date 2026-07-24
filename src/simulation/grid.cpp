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

Grid::Grid(u32 width, u32 height) : width_ {width + 2}, height_ {height + 2} {
  const auto deleter {[](f32* pointer) { cudaFree(pointer); }};

  data_ = std::unique_ptr<f32[], CudaMemoryDeleter>(allocate_device_memory(width_ * height_ * sizeof(f32)), deleter);
  grid_ = data_.get();

  reset();
}

auto Grid::view() const -> GridView {
  return GridView {width_ - 2, height_ - 2, grid_};
}

auto Grid::reset() -> void {
  cudaMemset(grid_, 0, width_ * height_ * sizeof(f32));
}

DoubleGrid::DoubleGrid(u32 width, u32 height) : grids_ {Grid {width, height}, Grid {width, height}}, current_idx_ {0u} {}

auto DoubleGrid::current() const -> GridView {
  return grids_[current_idx_].view();
}

auto DoubleGrid::next() const -> GridView {
  return grids_[1 - current_idx_].view();
}

auto DoubleGrid::swap() -> void {
  current_idx_ = 1 - current_idx_;
}

auto DoubleGrid::reset() -> void {
  grids_[0].reset();
  grids_[1].reset();
}

} // namespace fluidsim::simulation

#include "simulation/grid.hpp"

#include <cuda_runtime.h>

#include "utils/cuda.hpp"

namespace fluidsim::simulation {

Grid::Grid(u32 width, u32 height) : width_ {width + 2}, height_ {height + 2} {
  const auto deleter {[](f32* pointer) { cudaFree(pointer); }};

  utils::check_cuda_error(cudaMallocPitch(&grid_, &pitch_, width_ * sizeof(f32), height_));
  data_ = std::unique_ptr<f32[], CudaMemoryDeleter>(grid_, deleter);

  reset();
}

auto Grid::view() const -> GridView {
  return GridView {width_ - 2, height_ - 2, pitch_, grid_};
}

auto Grid::reset() -> void {
  utils::check_cuda_error(cudaMemset2D(grid_, pitch_, 0, width_ * sizeof(f32), height_));
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

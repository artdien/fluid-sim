#include "simulation/grid.hpp"

#include <cuda_runtime.h>
#include <vector_types.h>

#include "utils/cuda.hpp"

namespace fluidsim::simulation {

template <typename T>
Grid<T>::Grid(u32 width, u32 height) : width_ {width + 2}, height_ {height + 2} {
  const auto deleter {[](T* pointer) { cudaFree(pointer); }};

  utils::check_cuda_error(cudaMallocPitch(&grid_, &pitch_, width_ * sizeof(T), height_));
  data_ = std::unique_ptr<T[], CudaMemoryDeleter<T>>(grid_, deleter);

  reset();
}

template <typename T>
auto Grid<T>::view() const -> GridView<T> {
  return GridView<T> {width_ - 2, height_ - 2, pitch_, grid_};
}

template <typename T>
auto Grid<T>::reset() -> void {
  utils::check_cuda_error(cudaMemset2D(grid_, pitch_, 0, width_ * sizeof(T), height_));
}

template <typename T>
DoubleGrid<T>::DoubleGrid(u32 width, u32 height) : grids_ {Grid<T> {width, height}, Grid<T> {width, height}}, current_idx_ {0u} {}

template <typename T>
auto DoubleGrid<T>::current() const -> GridView<T> {
  return grids_[current_idx_].view();
}

template <typename T>
auto DoubleGrid<T>::next() const -> GridView<T> {
  return grids_[1 - current_idx_].view();
}

template <typename T>
auto DoubleGrid<T>::swap() -> void {
  current_idx_ = 1 - current_idx_;
}

template <typename T>
auto DoubleGrid<T>::reset() -> void {
  grids_[0].reset();
  grids_[1].reset();
}

template class Grid<f32>;
template class DoubleGrid<f32>;
template class DoubleGrid<float2>;
template class DoubleGrid<float4>;

} // namespace fluidsim::simulation

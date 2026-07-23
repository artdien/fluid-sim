#pragma once

#include <functional>
#include <memory>

#include "platform/device.hpp"
#include "platform/types.hpp"

namespace fluidsim::simulation {

using CudaMemoryDeleter = std::function<void(f32*)>;

struct GridView {
  u32 width;
  u32 height;
  f32* grid;

  CUDA_HOST_DEVICE CUDA_FORCEINLINE f32& at(u32 i, u32 j) {
    return grid[i + j * width];
  }

  CUDA_HOST_DEVICE CUDA_FORCEINLINE const f32& at(u32 i, u32 j) const {
    return grid[i + j * width];
  }
};

class Grid {
public:
  /// @brief Creates a two-dimensional grid in GPU memory.
  ///
  /// @param width Number of horizontal grid points.
  /// @param height Number of vertical grid points.
  Grid(u32 width, u32 height);

  Grid(const Grid&) = delete;
  Grid(Grid&&) = delete;
  auto operator=(const Grid&) -> Grid& = delete;
  auto operator=(Grid&&) -> Grid& = delete;
  ~Grid() = default;

  /// @brief Returns a modifiable view of the grid.
  auto view() const -> GridView {
    return GridView {width_, height_, grid_};
  }

private:
  u32 width_;
  u32 height_;
  f32* grid_;
  std::unique_ptr<f32[], CudaMemoryDeleter> data_;
};

} // namespace fluidsim::simulation

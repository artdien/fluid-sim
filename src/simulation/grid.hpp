#pragma once

#include <array>
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
    return grid[i + j * (width + 2u)];
  }

  CUDA_HOST_DEVICE CUDA_FORCEINLINE const f32& at(u32 i, u32 j) const {
    return grid[i + j * (width + 2u)];
  }
};

class Grid {
public:
  /// @brief Creates a two-dimensional grid in GPU memory with a boundary strip.
  ///
  /// The initial state of the grid contains zero in all cells.
  ///
  /// @param width Number of horizontal grid points (excluding boundary strip).
  /// @param height Number of vertical grid points (excluding boundary strip).
  Grid(u32 width, u32 height);

  Grid(const Grid&) = delete;
  Grid(Grid&&) = delete;
  auto operator=(const Grid&) -> Grid& = delete;
  auto operator=(Grid&&) -> Grid& = delete;
  ~Grid() = default;

  /// @brief Returns a modifiable view of the grid.
  ///
  /// The width and height in the view contains the 'logical' size of the grid, i.e. without the boundary strip.
  /// However, the grid itself is the full grid, i.e. indexing into the boundary strip cells is possible.
  ///
  /// @return Modifiable view of grid.
  auto view() const -> GridView;

  /// @brief Resets a grid to its initial state.
  auto reset() -> void;

private:
  u32 width_;
  u32 height_;
  f32* grid_;
  std::unique_ptr<f32[], CudaMemoryDeleter> data_;
};

class DoubleGrid {
public:
  DoubleGrid(u32 width, u32 height);

  DoubleGrid(const DoubleGrid&) = delete;
  DoubleGrid(DoubleGrid&&) = delete;
  auto operator=(const DoubleGrid&) -> DoubleGrid& = delete;
  auto operator=(DoubleGrid&&) -> DoubleGrid& = delete;
  ~DoubleGrid() = default;

  auto current() const -> GridView;
  auto next() const -> GridView;
  auto swap() -> void;
  auto reset() -> void;

private:
  std::array<Grid, 2> grids_;
  u32 current_idx_;
};

} // namespace fluidsim::simulation

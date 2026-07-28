#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <memory>

#include "platform/device.hpp"
#include "platform/types.hpp"

namespace fluidsim::simulation {

template <typename T>
using CudaMemoryDeleter = std::function<void(T*)>;

struct RawGridView {
  u32 width;
  u32 height;
  usize pitch;
  void* grid;
};

template <typename T>
struct GridView {
  u32 width;
  u32 height;
  usize pitch;
  T* grid;

  explicit GridView(const RawGridView& raw) : width {raw.width}, height {raw.height}, pitch {raw.pitch}, grid {reinterpret_cast<T*>(raw.grid)} {}
  GridView(u32 width, u32 height, usize pitch, T* grid) : width {width}, height {height}, pitch {pitch}, grid {grid} {}

  CUDA_HOST auto raw() -> RawGridView {
    return {width, height, pitch, reinterpret_cast<void*>(grid)};
  }

  CUDA_DEVICE CUDA_FORCEINLINE auto at(u32 i, u32 j) -> T& {
    return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(grid) + j * pitch)[i];
  }

  CUDA_DEVICE CUDA_FORCEINLINE auto at(u32 i, u32 j) const -> const T& {
    return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(grid) + j * pitch)[i];
  }

#ifdef __CUDACC__
  // The following two methods should only be used for non-coherent loads for read-only data,
  // e.g. if reading data from a source grid which is not modified within the kernel call.

  CUDA_DEVICE CUDA_FORCEINLINE auto ro(u32 i, u32 j) -> T {
    return __ldg(&reinterpret_cast<T*>(reinterpret_cast<std::byte*>(grid) + j * pitch)[i]);
  }

  CUDA_DEVICE CUDA_FORCEINLINE auto ro(u32 i, u32 j) const -> T {
    return __ldg(&reinterpret_cast<T*>(reinterpret_cast<std::byte*>(grid) + j * pitch)[i]);
  }
#endif
};

template <typename T>
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
  auto view() const -> GridView<T>;

  /// @brief Resets a grid to its initial state.
  auto reset() -> void;

private:
  u32 width_;
  u32 height_;
  usize pitch_;
  T* grid_;
  std::unique_ptr<T[], CudaMemoryDeleter<T>> data_;
};

template <typename T>
class DoubleGrid {
public:
  DoubleGrid(u32 width, u32 height);

  DoubleGrid(const DoubleGrid&) = delete;
  DoubleGrid(DoubleGrid&&) = delete;
  auto operator=(const DoubleGrid&) -> DoubleGrid& = delete;
  auto operator=(DoubleGrid&&) -> DoubleGrid& = delete;
  ~DoubleGrid() = default;

  /// @brief Returns a view of the current grid.
  ///
  /// @return A modifiable view of the current grid.
  auto current() const -> GridView<T>;

  /// @brief Returns a view of the next grid.
  ///
  /// @return A modifiable view of the next grid.
  auto next() const -> GridView<T>;

  /// @brief Swaps the roles of the current and next grids.
  auto swap() -> void;

  /// @brief Resets both grids to their initial state.
  auto reset() -> void;

private:
  std::array<Grid<T>, 2> grids_;
  u32 current_idx_;
};

} // namespace fluidsim::simulation

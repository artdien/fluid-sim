#pragma once

#include "platform/types.hpp"
#include "simulation/grid.hpp"

namespace fluidsim::simulation {

class Solver {
public:
  /// @brief Creates a two-dimensional fluid solver.
  ///
  /// This constructor allocates GPU memory for several internal grids.
  ///
  /// @param width  Number of horizontal grid cells.
  /// @param height Number of vertical grid cells.
  Solver(u32 width, u32 height);

  Solver(const Solver&) = delete;
  Solver(Solver&&) = delete;
  auto operator=(const Solver&) -> Solver& = delete;
  auto operator=(Solver&&) -> Solver& = delete;
  ~Solver() = default;

  /// @brief Advances the fluid simulation by one time step.
  ///
  /// This method launches several asynchronous CUDA kernels.
  /// While it returns quickly, the GPU may still be processing.
  ///
  /// @note CPU-side processing after the method call must be synchronized.
  ///       GPU-side processing after the method call should not require sychronization.
  auto step() -> void;

  /// @brief Provides a view of the grid data used for visualization.
  ///
  /// @return GridView Read-only handle to the GPU buffer containing the visualization data.
  auto grid() const -> const GridView;

private:
  DoubleGrid u_;
  DoubleGrid v_;
  DoubleGrid dye_;
  DoubleGrid pressure_;
  Grid divergence_;
};

} // namespace fluidsim::simulation

#pragma once

#include <mutex>

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

  /// @brief Applies an external force to the fluid at a specific position.
  ///
  /// This method maps the provided floating-point coordinates to the underlying
  /// grid cells and adds the force components to the velocity fields.
  ///
  /// @param position_x The x-coordinate where the force is applied.
  /// @param position_y The y-coordinate where the force is applied.
  /// @param force_x    The magnitude and direction of the force along the x-axis.
  /// @param force_y    The magnitude and direction of the force along the y-axis.
  ///
  /// @note This method is thread-safe. It uses an internal mutex to prevent
  ///       race conditions when updating grid buffers during a simulation step.
  auto add_external_force(f32 position_x, f32 position_y, f32 force_x, f32 force_y) -> void;

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

  std::mutex mutex_;
};

} // namespace fluidsim::simulation

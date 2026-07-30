#pragma once

#include <memory>
#include <mutex>

#include "platform/types.hpp"
#include "simulation/grid.hpp"

namespace fluidsim::simulation {

struct SolverParameters {
  f32 dt {1.0f};
  f32 density {1.0f};
  f32 viscosity {0.2f};
  f32 viscosity_dye {0.2f};
  u32 jacobi_iterations {40u};
  f32 jacobi_weight {0.67f};
  f32 external_force_radius {15.0f};
  f32 external_dye_radius {15.0f};
  bool allow_adding_external_force {true};
  bool allow_adding_external_dye {true};
  u32 block_size {16u};
};

class Solver {
public:
  /// @brief Creates a two-dimensional fluid solver.
  ///
  /// This constructor allocates GPU memory for several internal grids.
  ///
  /// @param parameters Initial parameters for the solver.
  /// @param width  Number of horizontal grid cells.
  /// @param height Number of vertical grid cells.
  Solver(const SolverParameters& parameters, u32 width, u32 height);

  Solver(const Solver&) = delete;
  Solver(Solver&&) = delete;
  auto operator=(const Solver&) -> Solver& = delete;
  auto operator=(Solver&&) -> Solver& = delete;
  ~Solver();

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
  /// @param x   X-coordinate where the force is applied.
  /// @param y   Y-coordinate where the force is applied.
  /// @param f_x Magnitude and direction of the force along the x-axis.
  /// @param f_y Magnitude and direction of the force along the y-axis.
  ///
  /// @note This method is thread-safe. It uses an internal mutex to prevent
  ///       race conditions when updating grid buffers during a simulation step.
  auto add_external_force(f32 x, f32 y, f32 f_x, f32 f_y) -> void;

  /// @brief Adds an external dye source to the fluid at a specific position.
  ///
  /// This method maps the provided floating-point coordinates to the underlying
  /// grid cells and adds the source to the dye field.
  ///
  /// @param x X-coordinate where the dye is added.
  /// @param y Y-coordinate where the dye is added.
  /// @param r Red channel of the dye source.
  /// @param g Green channel of the dye source.
  /// @param b Blue channel of the dye source.
  ///
  /// @note This method is thread-safe. It uses an internal mutex to prevent
  ///       race conditions when updating grid buffers during a simulation step.
  auto add_external_dye(f32 x, f32 y, f32 r, f32 g, f32 b) -> void;

  /// @brief Update the parameters used for the fluid simulation.
  ///
  /// The parameters can updated between every step.
  /// Once updated, the next step will immediately use the new parameters.
  ///
  /// @param parameters New parameters for the solver.
  ///
  /// @note This method is thread-safe. It uses an internal mutex to prevent
  ///       race conditions when updating parameters during a simulation step.
  auto update_parameters(const SolverParameters parameters) -> void;

  /// @brief Provides a view of the grid data used for visualization.
  ///
  /// @return RawGridView Read-only handle to the GPU buffer containing the visualization data.
  auto grid() const -> const RawGridView;

private:
  SolverParameters parameters_;
  std::mutex mutex_;

  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace fluidsim::simulation

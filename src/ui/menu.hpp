#pragma once

// Forward declarations
namespace fluidsim::simulation {

class Solver;
enum class InitialState;
struct ExternalConfiguration;
struct SolverParameters;

} // namespace fluidsim::simulation

namespace fluidsim::ui {

class Menu {
public:
  /// @brief Constructs a menu.
  ///
  /// All parameters except the visible flag can be directly changed via the menu,
  /// hence they are passed as pointers.
  ///
  /// @param visible Flag to denote whether menu is currently visible or not.
  Menu(simulation::Solver* solver, simulation::SolverParameters* parameters, simulation::ExternalConfiguration* configuration,
       simulation::InitialState* initial_state, bool* randomize_external_dye, bool visible = false);

  Menu(const Menu&) = delete;
  Menu(Menu&&) = delete;
  auto operator=(const Menu&) -> Menu& = delete;
  auto operator=(Menu&&) -> Menu& = delete;
  ~Menu() = default;

  /// @brief Displays the menu.
  ///
  /// This method must be called within an existing OpenGL context.
  /// Does not display if current visibility is set to false.
  /// In this case it should be toggled beforehand.
  auto display() -> void;

  /// @brief Toggle the visibility of the menu.
  auto toggle() -> void;

private:
  simulation::Solver* solver_;
  simulation::SolverParameters* parameters_;
  simulation::ExternalConfiguration* configuration_;
  simulation::InitialState* initial_state_;
  bool* randomize_external_dye_;
  bool visible_;
};

} // namespace fluidsim::ui

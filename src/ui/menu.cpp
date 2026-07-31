#include "ui/menu.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace fluidsim::ui {

namespace {

constexpr auto INITIAL_STATE_DESCRIPTIONS {std::array {"Empty", "Horizontal Split", "Vertical Split"}};

auto input_field_f32(const char* label, f32* value, f32 min = 0.0f, f32 max = std::numeric_limits<f32>::infinity(), const char* format = "%.3f") -> bool {
  if (ImGui::InputScalar(label, ImGuiDataType_Float, value, nullptr, nullptr, format)) {
    if (*value < min) {
      *value = min;
    }
    if (*value > max) {
      *value = max;
    }
    return true;
  }
  return false;
}

auto input_field_u32(const char* label, u32* value, u32 min = 0u) -> bool {
  // Using data type ImGuiDataType_U32 does not prevent entering negative values,
  // they will instead be casted to unsigned values and thus wrapped around.
  // Thus, we use a temporary signed value to work around this.
  if (auto temp {static_cast<i32>(*value)}; ImGui::InputScalar(label, ImGuiDataType_S32, &temp)) {
    if (temp < 0) {
      temp = 0;
    }
    if (temp < static_cast<i32>(min)) {
      temp = static_cast<i32>(min);
    }
    *value = static_cast<u32>(temp);
    return true;
  }
  return false;
}

auto center_button(const char* label) -> void {
  const auto menu_width {ImGui::GetContentRegionAvail().x};
  const auto button_width {ImGui::CalcTextSize(label).x + 2.0f * ImGui::GetStyle().FramePadding.x};

  ImGui::SetCursorPosX(0.5f * (menu_width - button_width));
}

} // namespace

Menu::Menu(simulation::Solver* solver, simulation::SolverParameters* parameters, simulation::ExternalConfiguration* configuration,
           simulation::InitialState* initial_state, bool* randomize_external_dye, bool visible)
    : solver_ {solver}, parameters_ {parameters}, configuration_ {configuration}, initial_state_ {initial_state},
      randomize_external_dye_ {randomize_external_dye}, visible_ {visible} {
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
}

auto Menu::display() -> void {
  if (!visible_) {
    return;
  }

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::Begin("Menu", &visible_, ImGuiWindowFlags_AlwaysAutoResize);

  // --- SIMULATION PARAMETERS SECTION --- //

  ImGui::SeparatorText("Simulation Parameters");

  input_field_f32("Time Step", &parameters_->dt, 0.001f);
  input_field_f32("Density", &parameters_->density, 0.001f);
  input_field_f32("Viscosity (Velocity)", &parameters_->viscosity);
  input_field_f32("Viscosity (Dye)", &parameters_->viscosity_dye);
  input_field_u32("Jacobi Iterations", &parameters_->jacobi_iterations, 1u);
  input_field_f32("Jacobi Weight", &parameters_->jacobi_weight, 0.0f, 1.0f);
  input_field_u32("Block Size", &parameters_->block_size, 1u);

  // --- EXTERNAL CONFIGURATION SECTION --- //

  ImGui::SeparatorText("External Configuration");

  auto changed {false};

  changed |= ImGui::Checkbox("Add External Force", &configuration_->allow_adding_external_force);
  if (configuration_->allow_adding_external_force) {
    changed |= input_field_f32("Force Radius", &configuration_->external_force_radius, 0.0f);
  }

  changed |= ImGui::Checkbox("Add External Dye", &configuration_->allow_adding_external_dye);
  if (configuration_->allow_adding_external_dye) {
    ImGui::SameLine();
    changed |= ImGui::Checkbox("Randomize", randomize_external_dye_);
    changed |= input_field_f32("Dye Radius", &configuration_->external_dye_radius, 0.0f);
  }

  if (changed) {
    solver_->update_configuration(*configuration_);
  }

  // --- INITIAL STATE SECTION --- //

  ImGui::SeparatorText("Initial State");

  ImGui::Combo("Initial State", reinterpret_cast<i32*>(initial_state_), INITIAL_STATE_DESCRIPTIONS.data(), INITIAL_STATE_DESCRIPTIONS.size());

  // --- APPLY & RESET SECTION --- //

  ImGui::SeparatorText("Apply & Reset");

  center_button("Apply Parameters");
  if (ImGui::Button("Apply Parameters")) {
    solver_->update_parameters(*parameters_);
  }

  center_button("Reset Simulation");
  if (ImGui::Button("Reset Simulation")) {
    solver_->reset(*initial_state_);
  }

  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

auto Menu::toggle() -> void {
  visible_ = !visible_;
  ImGui::GetIO().WantCaptureMouse = false;
  ImGui::GetIO().WantCaptureKeyboard = false;
}

} // namespace fluidsim::ui

#include "platform/window.hpp"

#include <chrono>
#include <format>
#include <iostream>
#include <stdexcept>

#include <glad/glad.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "platform/debug.hpp"
#include "platform/input.hpp"

namespace fluidsim::platform {

namespace {

auto determine_mouse_position(GLFWwindow* window, u32 height, MousePosition last_mouse_position) -> MousePosition {
  static auto first_mouse_click {true};

  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
    auto mouse_position_x {0.0};
    auto mouse_position_y {0.0};
    glfwGetCursorPos(window, &mouse_position_x, &mouse_position_y);

    const auto current_mouse_position {MousePosition {
        .x = static_cast<f32>(mouse_position_x),
        .y = static_cast<f32>(height - mouse_position_y),
    }};

    if (first_mouse_click) {
      first_mouse_click = false;
      return current_mouse_position;
    }

    const auto position_delta {MousePositionDelta {
        .dx = current_mouse_position.x - last_mouse_position.x,
        .dy = current_mouse_position.y - last_mouse_position.y,
    }};

    add_mouse_input_event({
        .position = current_mouse_position,
        .position_delta = position_delta,
        .pressed = true,
    });

    return current_mouse_position;
  }

  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
    first_mouse_click = true;
    return last_mouse_position;
  }

  return last_mouse_position;
}

auto key_callback([[maybe_unused]] GLFWwindow* window, int key, int scan_code, int action, [[maybe_unused]] int modifiers) -> void {
  if (action == GLFW_PRESS) {
    if (key == GLFW_KEY_ESCAPE) {
      add_keyboard_input_event({.key = std::string {"esc"}});
    }
    if (const auto key_name {glfwGetKeyName(key, scan_code)}; key_name) {
      add_keyboard_input_event({.key = std::string {key_name}});
    }
  }
}

} // namespace

Window::Window(u32 width, u32 height, const std::string& title) : title_ {title}, width_ {width}, height_ {height} {
  glfwSetErrorCallback([](int error, const char* description) { std::cerr << std::format("Error initializing window: [{}] {}\n", error, description); });

  if (!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW");
  }

  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  if constexpr (DEBUG_BUILD) {
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
  }

  window_ = glfwCreateWindow(static_cast<i32>(width_), static_cast<i32>(height_), title_.c_str(), NULL, NULL);
  if (!window_) {
    glfwTerminate();
    throw std::runtime_error("Failed to create window");
  }

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);
  gladLoadGLLoader(reinterpret_cast<GLADloadproc>(&glfwGetProcAddress));

  if constexpr (DEBUG_BUILD) {
    // Can only be enabled after OpenGL context is created
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
    glDebugMessageCallback(debug_callback, nullptr);
  }

  glfwSetKeyCallback(window_, key_callback);

  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init();

  std::cout << "Initialized window with OpenGL context\n";
  std::cout << "  OpenGL: " << glGetString(GL_VERSION) << '\n';
  std::cout << "  GPU: " << glGetString(GL_RENDERER) << '\n';
}

Window::~Window() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  glfwDestroyWindow(window_);
  glfwTerminate();
}

auto Window::open(std::function<void(MouseInput, KeyboardInput, f64)> execute_per_frame) -> void {
  auto previous_time {std::chrono::steady_clock::now()};
  auto mouse_position {MousePosition {0.0f, 0.0f}};

  while (!glfwWindowShouldClose(window_)) {
    const auto current_time {std::chrono::steady_clock::now()};
    const auto elapsed_time {std::chrono::round<std::chrono::microseconds>(current_time - previous_time).count() / 1000.0};
    previous_time = current_time;

    mouse_position = determine_mouse_position(window_, height_, mouse_position);

    auto mouse_input {get_mouse_input_event()};
    if (ImGui::GetIO().WantCaptureMouse) {
      mouse_input.reset();
    }

    auto keyboard_input {get_keyboard_input_event()};
    if (ImGui::GetIO().WantCaptureKeyboard) {
      keyboard_input.reset();
    }

    execute_per_frame(mouse_input.value_or({}), keyboard_input.value_or({}), elapsed_time);

    glfwSwapBuffers(window_);
    glfwPollEvents();
  }
}

auto Window::close() -> void {
  glfwSetWindowShouldClose(window_, GL_TRUE);
}

auto Window::set_title(const std::string& title) -> void {
  title_ = title;
  glfwSetWindowTitle(window_, title_.c_str());
}

} // namespace fluidsim::platform

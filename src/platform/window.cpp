#include "platform/window.hpp"

#include <chrono>
#include <format>
#include <iostream>
#include <stdexcept>

#include <glad/glad.h>

namespace fluidsim::platform {

Window::Window(u32 width, u32 height, const std::string& title) : title_ {title}, width_ {width}, height_ {height} {
  glfwSetErrorCallback([](int error, const char* description) { std::cerr << std::format("Error initializing window: [{}] {}\n", error, description); });

  if (!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW");
  }

  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window_ = glfwCreateWindow(static_cast<i32>(width_), static_cast<i32>(height_), title_.c_str(), NULL, NULL);
  if (!window_) {
    glfwTerminate();
    throw std::runtime_error("Failed to create window");
  }

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);
  gladLoadGLLoader(reinterpret_cast<GLADloadproc>(&glfwGetProcAddress));

  std::cout << "Initialized window with OpenGL context\n";
  std::cout << "  OpenGL: " << glGetString(GL_VERSION) << '\n';
  std::cout << "  GPU: " << glGetString(GL_RENDERER) << '\n';
}

Window::~Window() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

auto Window::open(std::function<void(double)> execute_per_frame) -> void {
  auto previous_time {std::chrono::steady_clock::now()};

  while (!glfwWindowShouldClose(window_)) {
    const auto current_time {std::chrono::steady_clock::now()};
    const auto elapsed_time {std::chrono::round<std::chrono::microseconds>(current_time - previous_time).count() / 1000.0};
    previous_time = current_time;

    execute_per_frame(elapsed_time);

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

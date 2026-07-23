#pragma once

#include <glad/glad.h>

#include "platform/types.hpp"

namespace fluidsim::rendering {

class Renderer {
public:
  /// @brief Creates a renderer.
  ///
  /// This renderer is simple and only renders a texture as full-screen quad.
  ///
  /// This object must be constructed within an existing OpenGL context.
  /// Otherwise the construction will fail.
  ///
  /// @param width  Window width in pixels.
  /// @param height Window height in pixels.
  Renderer(u32 width, u32 height);

  Renderer(const Renderer&) = delete;
  Renderer(Renderer&&) = delete;
  auto operator=(const Renderer&) -> Renderer& = delete;
  auto operator=(Renderer&&) -> Renderer& = delete;
  ~Renderer();

  /// @brief Renders the provided texture as a full-screen quad.
  ///
  /// @param texture_id OpenGL handle of the texture to be rendered.
  auto render(GLuint texture_id) -> void;

private:
  GLuint vao_id_;
  GLuint program_id_;
  GLint uniform_framebuffer_;

  u32 width_;
  u32 height_;
};

} // namespace fluidsim::rendering

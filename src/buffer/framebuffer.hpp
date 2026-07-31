#pragma once

#include <memory>

#include <glad/glad.h>

#include "platform/types.hpp"

// Forward declarations
namespace fluidsim::simulation {

struct RawGridView;

} // namespace fluidsim::simulation

namespace fluidsim::buffer {

class Framebuffer {
public:
  /// @brief Initializes a new GPU framebuffer with a specific resolution.
  ///
  /// This constructor allocates a texture used as a framebuffer in GPU memory
  /// and registers it as a CUDA-interop resource.
  /// This object must be constructed within an existing OpenGL context.
  /// Otherwise the construction will fail.
  ///
  /// @param width  The width of the texture in pixels.
  /// @param height The height of the texture in pixels.
  Framebuffer(u32 width, u32 height);

  Framebuffer(const Framebuffer&) = delete;
  Framebuffer(Framebuffer&&) = delete;
  auto operator=(const Framebuffer&) -> Framebuffer& = delete;
  auto operator=(Framebuffer&&) -> Framebuffer& = delete;
  ~Framebuffer();

  /// @brief Updates the GPU framebuffer with the current state of the simulation grid.
  ///
  /// @param grid A view of the simulation data to be visualized.
  ///
  /// @note This is a synchronous operation from the perspective of the CUDA stream.
  ///       After returning, subsequent OpenGL rendering calls can be immediately issued.
  auto update(simulation::RawGridView grid) -> void;

  /// @brief Retrieves the OpenGL handle of the underlying texture.
  ///
  /// @return GLuint OpenGL texture ID.
  auto texture_id() const -> GLuint;

private:
  u32 width_;
  u32 height_;
  GLuint texture_id_;

  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace fluidsim::buffer

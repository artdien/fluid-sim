#include "rendering/renderer.hpp"

#include <string_view>

#include "utils/gl.hpp"

namespace fluidsim::rendering {

namespace {

constexpr auto VERTEX_SHADER {std::string_view {
#include "shaders/shader.vert"
}};

constexpr auto FRAGMENT_SHADER {std::string_view {
#include "shaders/shader.frag"
}};

} // namespace

Renderer::Renderer(u32 width, u32 height) : width_ {width}, height_ {height} {
  const auto vertex_shader_id {utils::compile_shader(GL_VERTEX_SHADER, VERTEX_SHADER.data())};
  const auto fragment_shader_id {utils::compile_shader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER.data())};
  program_id_ = utils::link_shaders(vertex_shader_id, fragment_shader_id);

  glUseProgram(program_id_);
  uniform_framebuffer_ = glGetUniformLocation(program_id_, "framebuffer");
  glUseProgram(0);

  glCreateVertexArrays(1, &vao_id_);

  // Resizing window is not allowed, so we set viewport only once.
  glViewport(0, 0, width_, height_);
}

Renderer::~Renderer() {
  glDeleteProgram(program_id_);
  glDeleteVertexArrays(1, &vao_id_);
}

auto Renderer::render(GLuint texture_id) -> void {
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_id);

  glUseProgram(program_id_);
  glUniform1i(uniform_framebuffer_, 0);

  glBindVertexArray(vao_id_);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glBindVertexArray(0);
  glUseProgram(0);
}

} // namespace fluidsim::rendering

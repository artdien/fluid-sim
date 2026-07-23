#include "utils/gl.hpp"

#include <vector>

#include <glad/glad.h>

#include "platform/types.hpp"

namespace fluidsim::utils {

auto compile_shader(GLuint shader_type, const char* shader) -> GLuint {
  const auto shader_id {glCreateShader(shader_type)};
  glShaderSource(shader_id, 1, &shader, nullptr);
  glCompileShader(shader_id);

  auto log_size {GLint {}};
  glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &log_size);
  if (log_size > 0) {
    auto log {std::vector<c8>(static_cast<u32>(log_size + 1))};
    glGetShaderInfoLog(shader_id, log_size, nullptr, log.data());
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_OTHER, 0, GL_DEBUG_SEVERITY_NOTIFICATION, -1, log.data());
  }

  return shader_id;
}

auto link_shaders(GLuint vertex_shader_id, GLuint fragment_shader_id) -> GLuint {
  const auto shader_program_id {glCreateProgram()};

  glAttachShader(shader_program_id, vertex_shader_id);
  glAttachShader(shader_program_id, fragment_shader_id);
  glLinkProgram(shader_program_id);

  auto log_size {GLint {}};
  glGetProgramiv(shader_program_id, GL_INFO_LOG_LENGTH, &log_size);
  if (log_size > 0) {
    auto log {std::vector<c8>(static_cast<u32>(log_size + 1))};
    glGetProgramInfoLog(shader_program_id, log_size, nullptr, log.data());
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_OTHER, 0, GL_DEBUG_SEVERITY_NOTIFICATION, -1, log.data());
  }

  glDetachShader(shader_program_id, vertex_shader_id);
  glDetachShader(shader_program_id, fragment_shader_id);
  glDeleteShader(vertex_shader_id);
  glDeleteShader(fragment_shader_id);

  return shader_program_id;
}

} // namespace fluidsim::utils

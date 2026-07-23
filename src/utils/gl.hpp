#pragma once

#include <glad/glad.h>

namespace fluidsim::utils {

/// @brief Compiles a GLSL shader from the provided source code.
///
/// This method must be called within an existing OpenGL context.
///
/// @param shader_type Type of shader to create (GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, ...).
/// @param shader      Null-terminated string containing the GLSL source code.
///
/// @return ID of the compiled shader.
auto compile_shader(GLuint shader_type, const char* shader) -> GLuint;

/// @brief Links multiple compiled shaders into a single shader program.
///
/// This method must be called within an existing OpenGL context.
/// The individual shaders are detached and deleted after linking.
///
/// @param vertex_shader_id   ID of vertex shader to be linked into the program.
/// @param fragment_shader_id ID of fragment shader to be linked into the program.
///
/// @return ID of the shader program.
auto link_shaders(GLuint vertex_shader_id, GLuint fragment_shader_id) -> GLuint;

} // namespace fluidsim::utils

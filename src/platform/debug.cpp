#include "platform/debug.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <iostream>

using namespace std::string_literals;

namespace fluidsim::platform {

namespace {

constexpr auto SUPPRESSED_MESSAGE_IDS {std::array {
    131185u, // Details about created buffers
}};

}

auto debug_callback(GLenum source, GLenum type, GLuint id, GLenum severity, [[maybe_unused]] GLsizei length, GLchar const* message,
                    [[maybe_unused]] void const* user_param) -> void {
  if (std::find(SUPPRESSED_MESSAGE_IDS.begin(), SUPPRESSED_MESSAGE_IDS.end(), id) != SUPPRESSED_MESSAGE_IDS.end()) {
    return;
  }

  const auto source_as_string {[](auto source) {
    switch (source) {
      case GL_DEBUG_SOURCE_API:
        return "API"s;
      case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        return "WINDOW SYSTEM"s;
      case GL_DEBUG_SOURCE_SHADER_COMPILER:
        return "SHADER COMPILER"s;
      case GL_DEBUG_SOURCE_THIRD_PARTY:
        return "THIRD PARTY"s;
      case GL_DEBUG_SOURCE_APPLICATION:
        return "APPLICATION"s;
      case GL_DEBUG_SOURCE_OTHER:
        return "OTHER"s;
      default:
        return std::format("UNKNOWN_SOURCE({})", source);
    }
  }(source)};

  const auto type_as_string {[](auto type) {
    switch (type) {
      case GL_DEBUG_TYPE_ERROR:
        return "ERROR"s;
      case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        return "DEPRECATED_BEHAVIOR"s;
      case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        return "UNDEFINED_BEHAVIOR"s;
      case GL_DEBUG_TYPE_PORTABILITY:
        return "PORTABILITY"s;
      case GL_DEBUG_TYPE_PERFORMANCE:
        return "PERFORMANCE"s;
      case GL_DEBUG_TYPE_MARKER:
        return "MARKER"s;
      case GL_DEBUG_TYPE_OTHER:
        return "OTHER"s;
      default:
        return std::format("UNKNOWN_TYPE({})", type);
    }
  }(type)};

  const auto severity_as_string {[](auto severity) {
    switch (severity) {
      case GL_DEBUG_SEVERITY_NOTIFICATION:
        return "NOTIFICATION"s;
      case GL_DEBUG_SEVERITY_LOW:
        return "LOW"s;
      case GL_DEBUG_SEVERITY_MEDIUM:
        return "MEDIUM"s;
      case GL_DEBUG_SEVERITY_HIGH:
        return "HIGH"s;
      default:
        return std::format("UNKNOWN_SEVERITY({})", severity);
    }
  }(severity)};

  std::cout << std::format("{} - [{}] [{}] [{}] - {}\n", severity_as_string, source_as_string, type_as_string, id, message);
}

} // namespace fluidsim::platform

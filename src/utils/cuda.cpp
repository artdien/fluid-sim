#include "utils/cuda.hpp"

#include <format>

#include <cuda_runtime.h>

namespace fluidsim::utils {

auto check_cuda_error(i32 error, const std::source_location location) -> void {
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::format("CUDA error at file {} (line {}): {}\n", location.file_name(), location.line(), cudaGetErrorString(static_cast<cudaError_t>(error))));
  }
}

auto check_async_cuda_error(const std::source_location location) -> void {
  if (const auto error {cudaGetLastError()}; error != cudaSuccess) {
    throw std::runtime_error(std::format("CUDA error at file {} (line {}): {}\n", location.file_name(), location.line(), cudaGetErrorString(error)));
  }
}

} // namespace fluidsim::utils

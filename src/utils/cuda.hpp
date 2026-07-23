#pragma once

#include <source_location>

#include "platform/types.hpp"

namespace fluidsim::utils {

/// @brief Validates a CUDA runtime return code.
///
/// This function checks if the provided error code indicates success.
/// If an error is detected, a runtime exception is thrown.
///
/// @param error    cudaError_t (passed as int) returned by a CUDA API call.
/// @param location Source location where error occurred (automatically populated).
///
/// @throws std::runtime_error If @p error is not equal to cudaSuccess.
///
/// @note Used for synchronous CUDA calls (cudaMalloc, cudaMemcpy, ...).
auto check_cuda_error(i32 error, const std::source_location location = std::source_location::current()) -> void;

/// @brief Checks GPU for any pending asynchronous errors.
///
/// This function checks for asynchronous CUDA errors.
/// Some CUDA operations, e.g. kernel launches, do not return an error code immediately.
/// This function queries the CUDA runtime for the most recent error that occurred on the device
/// since the last time this (or a similar) check was performed.
/// If an error is detected, a runtime exception is thrown.
///
/// @param location Source location where the check is performed (automatically populated).
///
/// @throws std::runtime_error If the last GPU operation resulted in an error.
///
/// @note Used for asynchronous CUDA calls (e.g. kernel launches).
///       Typically called immediately after a kernel launch.
auto check_async_cuda_error(const std::source_location location = std::source_location::current()) -> void;

} // namespace fluidsim::utils

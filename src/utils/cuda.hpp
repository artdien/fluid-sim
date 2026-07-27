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

#ifdef __CUDACC__

struct ExecutionConfiguration {
  dim3 blocks;
  dim3 threads;
};

///@brief Calculates the execution configuration for a 1D CUDA kernel.
///
/// @param size       Total number of elements to process.
/// @param block_size Number of threads per block.
///
/// @return An ExecutionConfiguration containing the calculated grid and block dimensions.
inline auto execution_configuration(u32 width, u32 height, u32 block_size) -> ExecutionConfiguration {
  return {
      .blocks = dim3((width + block_size - 1) / block_size, (height + block_size - 1) / block_size),
      .threads = dim3(block_size, block_size),
  };
}

/// @brief Calculates the execution configuration for a 2D CUDA kernel.
///
/// @param width      Width of the 2D domain in elements.
/// @param height     Height of the 2D domain in elements.
/// @param block_size Dimension of the square thread blocks.
///                   Results in a block size of block_size * block_size threads.
///
/// @return An ExecutionConfiguration containing the calculated grid and block dimensions.
inline auto execution_configuration(u32 size, u32 block_size) -> ExecutionConfiguration {
  return {
      .blocks = dim3((size + block_size - 1) / block_size),
      .threads = dim3(block_size),
  };
}

#endif

} // namespace fluidsim::utils

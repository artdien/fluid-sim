#pragma once

#ifdef __CUDACC__
  #define CUDA_HOST __host__
  #define CUDA_DEVICE __device__
  #define CUDA_HOST_DEVICE __host__ __device__
  #define CUDA_FORCEINLINE __forceinline__
#else
  #define CUDA_HOST
  #define CUDA_DEVICE
  #define CUDA_HOST_DEVICE
  #define CUDA_FORCEINLINE inline
#endif

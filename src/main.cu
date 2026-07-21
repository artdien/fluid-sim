#include <cstdio>
#include <iostream>

__global__ auto hello() -> void {
  std::printf("Hello, Device!\n");
}

auto main() -> int {
  std::cout << "Hello, Host!\n";
  hello<<<1, 1>>>();
  cudaDeviceSynchronize();

  return 0;
}

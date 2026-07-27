find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(fluid-sim)
target_link_libraries(
  fluid-sim
  PRIVATE
    glfw
    glad::glad
    CUDA::cudart
)
target_include_directories(
  fluid-sim
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(
  fluid-sim
  PRIVATE
    GLFW_INCLUDE_NONE
)
target_sources(
  fluid-sim
  PRIVATE
    buffer/kernels/update_framebuffer.cu
    buffer/framebuffer.cpp
    platform/debug.cpp
    platform/window.cpp
    rendering/renderer.cpp
    simulation/kernels/advection.cu
    simulation/kernels/boundary.cu
    simulation/kernels/diffusion.cu
    simulation/kernels/initial.cu
    simulation/kernels/projection.cu
    simulation/grid.cpp
    simulation/solver.cpp
    utils/cuda.cpp
    utils/gl.cpp
    main.cpp
  )

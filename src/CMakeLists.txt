find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

add_executable(fluid-sim)
target_link_libraries(
  fluid-sim
  PRIVATE
    glfw
    glad::glad
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
    platform/debug.cpp
    platform/window.cpp
    main.cpp
  )

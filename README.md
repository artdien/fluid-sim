# fluid-sim

A real-time fluid simulation written in C++ for solving the Navier-Stokes equations, accelerated by CUDA on the GPU. Below is a demonstration of this simulation, showing real-time rendering of colorful dye with vortices:

<p align="center">
  <img src="docs/demo.gif" alt="animated" />
</p>

## Overview

This project simulates fluids in two dimensions. It implements a numerical integrator to solve the Navier-Stokes equations using a finite-difference scheme on a staggered grid, written in C++ and accelerated via CUDA on the GPU. The simulation is rendered in real-time using OpenGL. Rather than striving for a physically accurate CFD (Computational Fluid Dynamics) simulation, the goal of this project was to create a visually pleasing result typical of computer graphics applications. To achieve this, it employs the well-known 'stable fluids' pipeline, ensuring unconditional stability for long-term simulations.

More information about this project can be found in the [documentation](docs/documentation.md).

### Key Features

- Real-time visualization via OpenGL and incompressible Navier-Stokes solving via CUDA.
- Real-time injection of velocity or dye using mouse input.
- **Numerical Stability**: 
  - Staggered grid implementation to prevent pressure oscillations.
  - Semi-Lagrangian advection scheme for long-term stability.
  - Weighted Jacobi iteration for solving the pressure equation.
- **Optional Simulation Effects**:
  - Diffusion of velocity and dye.
  - Vorticity confinement to preserve small-scale vortices.
- **Preset Initial States**: 
  - Empty simulation domain.
  - Horizontally split domain.
  - Vertically split domain.

## Getting Started

### Prerequisites

To build this project, you will need:
- A C++ compiler supporting C++20.
- CMake 4.0.0 or newer (with a supported build tool such as Ninja or Make).
- A GPU supporting OpenGL 4.6 with CUDA compatibility 7.5 or newer.
- CUDA Toolkit
- vcpkg
- git

The default build settings use Clang as the C++ compiler and Ninja as the build tool. However, this can be changed either by adapting `CMakePresets.json` or creating a user-specific `CMakeUserPresets.json`.

### Building and Running

1. Clone the repository:
   ```bash
   git clone https://github.com/artdien/fluid-sim.git
   cd fluid-sim
   ```

2. Build the project (this will automatically download all necessary dependencies via vcpkg):
   ```bash
   export VCPKG_ROOT="/path/to/vcpkg"
   cmake --workflow release # alternatively: cmake --workflow debug
   cmake --build build
   ```

3. Run the simulation:
   ```bash
   # Arguments are optional, default values (shown here) are used if omitted.
   ./build/src/fluid-sim -width 1920 -height 1080
   ```

### Usage

Launching the application opens a window and immediately starts the simulation. The initial window size can be specified via command-line arguments as shown above.

By default, the simulation starts empty, so no fluid will be visible initially. Use the mouse to inject dye and velocity into the domain.

The general controls are:

- **Left Mouse Button (Click and Hold)**: Add dye and/or velocity (configurable in the menu).
- **Key 'ESC'**: Close application.
- **Key 'm'**: Toggle the menu.

The menu settings are:

- *Simulation Parameters* (changes take effect after applying parameters):
  - **Time Step**: The size of the time step $dt$.
  - **Density**: The constant density used in the Navier-Stokes equations.
  - **Viscosity (Velocity)**: Constant viscosity for velocity diffusion (set to zero to disable).
  - **Viscosity (Dye)**: Constant viscosity for dye diffusion (set to zero to disable).
  - **Vorticity Confinement**: Strength of vorticity confinement (set to zero to disable).
  - **Jacobi Iterations**: Number of iterations performed by the weighted Jacobi solver for the pressure equation.
  - **Jacobi Weight**: Weight used by the weighted Jacobi solver.
  - **Block Size**: Side length of square block used for launching CUDA kernels (e.g. 16 for launching grid with 16x16 blocks).
- *External Configuration* (changes take effect immediately):
  - **Add External Force**: Toggle whether velocity can be added via mouse.
  - **Force Radius**: Radius of effect for added velocity.
  - **Add External Dye**: Toggle whether dye can be added via mouse.
  - **Randomize**: Toggle whether added dye has randomized colors.
  - **Dye Radius**: Radius of effect for added dye.
- *Initial State* (applied only upon resetting the simulation):
  - **Initial State**: Initial values to apply to the dye.
- Button **Apply Parameters**: Apply current simulation parameter changes.
- Button **Reset Simulation**: Reset the simulation grids using the specified initial state.

### Dependencies

This project uses the following dependencies, managed via vcpkg:

* [GLFW](https://github.com/glfw/glfw): Used for window creation and the handling of keyboard and mouse input.
* [glad](https://github.com/dav1dde/glad): An OpenGL loader used to access modern function pointers.
* [Dear ImGui](https://github.com/ocornut/imgui): Used to implement a menu, allowing for the adjustment of simulation parameters and rendering settings.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
## Introduction

This project simulates fluids in two dimensions. It implements a numerical integrator to solve the Navier-Stokes equations using a finite-difference scheme on a staggered grid, written in C++ and accelerated via CUDA on the GPU. The simulation is rendered in real-time using OpenGL. Rather than striving for a physically accurate CFD (Computational Fluid Dynamics) simulation, the goal of this project was to create a visually pleasing result typical of computer graphics applications. To achieve this, it employs the well-known 'stable fluids' pipeline, ensuring unconditional stability for long-term simulations.

This document is divided into two main sections:

* [Theoretical Foundation](#theoretical-foundation): This section provides the necessary background to understand the implementation, covering everything from the physics of the Navier-Stokes equations to the mathematical tools used to solve them.
* [Implementation](#implementation): This section describes the software architecture and explains the reasoning behind key technical decisions.

## Theoretical Foundation

### Notation

To state the Navier-Stokes equations concisely, we establish the following notation for the remainder of this document:

* Vectors or vector fields are denoted in **boldface**.
* The domain is denoted by $\Omega \subset \mathbb{R}^2$ with boundary $\Gamma = \partial \Omega$.
* Time is denoted by $t \in \mathbb{R}_{\ge 0}$.
* The constant density is denoted by $\rho \in \mathbb{R}_{\gt 0}$.
* The constant kinematic viscosity is denoted by $\nu \in \mathbb{R}_{\ge 0}$.
* The velocity field is a vector field and will be denoted by
  $$\mathbf{u}\colon \Omega \times \mathbb{R}_{\ge 0} \to \mathbb{R}^2, \quad (\mathbf{x},t) \mapsto \mathbf{u}(\mathbf{x},t)$$
  When decomposed into scalar components, the horizontal velocity (first component) is denoted by $u(\mathbf{x},t)$ and the vertical velocity (second component) by $v(\mathbf{x},t)$. Thus, $\mathbf{u}(\mathbf{x},t) = (u(\mathbf{x},t),v(\mathbf{x},t))$. While using the letter u for both the vector field and its first scalar component may seem ambiguous, this is a standard convention in fluid dynamics.
* The pressure field is a scalar field and will be denoted by
  $$p\colon \Omega \times \mathbb{R}_{\ge 0} \to \mathbb{R}, \quad (\mathbf{x},t) \mapsto p(\mathbf{x},t)$$
* External forces acting on the fluid will be denoted by $\mathbf{F}$.
* Other simulated quantities can be either scalar or vector fields and will correspondingly be denoted by: 
  $$q\colon \Omega \times \mathbb{R}_{\ge 0} \to \mathbb{R}, \quad (\mathbf{x},t) \mapsto q(\mathbf{x},t) \quad \text{or} \quad \mathbf{q}\colon \Omega \times \mathbb{R}_{\ge 0} \to \mathbb{R}^n, \quad (\mathbf{x},t) \mapsto \mathbf{q}(\mathbf{x},t)$$

### Navier-Stokes Equations

Fluid simulations (e.g. for water or smoke) typically rely on the Navier-Stokes equations, which describe the dynamics of velocity and pressure fields via partial differential equations. In this project, we focus on a specific form: the incompressible Navier-Stokes equations with constant viscosity and constant density.

These equations are defined as:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = \nu \Delta \mathbf{u} - \frac{1}{\rho} \nabla p + \frac{1}{\rho} \mathbf{F}
\quad \text{subject to} \quad
\nabla \cdot \mathbf{u} = 0
$$

The first equation is also known as the momentum equation, describing the time evolution of the velocity field.

The second is also known as the continuity equation or incompressibility constraint, ensuring that the velocity field is divergence-free. Rather than describing evolution over time, it ensures that the fluid's volume remains constant, i.e. it cannot be compressed or expanded. While this is an approximation, as gases and even liquids are technically compressible, it is an accurate assumption for most 'normal' conditions encountered in computer graphics.

To solve these equations, we must define an initial-boundary value problem by specifying initial conditions as well as boundary conditions.

For the initial conditions, we simply set initial values $\mathbf{u}_{0,x} \in \mathbb{R}^2$ at time $t=0$, i.e. $\mathbf{u}(\mathbf{x},0) = \mathbf{u}_{0,x}$ for all $\mathbf{x}$.

For the boundary conditions, we decompose $\mathbf{u}$ into a component $\mathbf{u}_n$ normal to the domain boundary $\Gamma$ and a component $\mathbf{u}_t$ tangential to $\Gamma$. The normalized normal vector $\hat{\mathbf{n}}$ is defined as pointing outwards from $\Gamma$. Using this decomposition, we can then set different boundary conditions which must hold for all $t > 0$. There are many different boundary conditions, but in this project, we employ free-slip boundary conditions, which allow the fluid to 'slide' along the boundary without friction:

$$
\begin{aligned}
  \mathbf{u}_n &= \mathbf{0} \\
  \frac{\partial \mathbf{u}_t}{\partial \mathbf{n}} = \nabla \mathbf{u}_t \cdot \hat{\mathbf{n}} &= \mathbf{0}
\end{aligned}
$$

These are a combination of Dirichlet and Neumann boundary conditions that prevent the fluid from penetrating the wall ($\mathbf{u}_n = 0$) while ensuring there is no tangential shear stress.

While the velocity and pressure fields describe the underlying physics of the fluid, they are essentially invisible. To produce a visual result, it is necessary to simulate additional quantities that flow within the fluid, such as smoke density or dye. These quantities can be either scalar-valued (e.g. smoke density) or vector-valued (e.g. RGB color of dye). For the sake of generality, we treat these as vector fields $\mathbf{q}$, noting that scalar quantities can be viewed as one-dimensional vector fields.

In fluid dynamics, a distinction is made between active and passive quantities. Active quantities exert a feedback effect on the fluid by altering its density $\rho$ or viscosity $\nu$. In contrast, passive quantities are simply transported by the velocity field without influencing the fluid's motion. This project focuses exclusively on passive quantities.

The dynamics of a passive quantity are governed by the following equation:

$$
\frac{\partial \mathbf{q}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{q} = \nu_q \Delta \mathbf{q} + \mathbf{Q}
$$

where $\nu_q \in \mathbb{R}_{\ge 0}$ is the viscosity specific to the quantity $\mathbf{q}$, and $\mathbf{Q}$ represents an external source term. Structurally, this equation is similar to the momentum equation of the Navier-Stokes equations. However, it lacks the pressure gradient term and is not subject to the divergence-free constraint. Instead, the velocity field $\mathbf{u}$, solved via the Navier-Stokes equations, acts as the transport mechanism that 'carries' the quantity across the domain.

To complete the initial-boundary value problem for these quantities, we specify an initial state for all $\mathbf{x}$ at $t = 0$, i.e. $\mathbf{q}(\mathbf{x}, 0) = \mathbf{q}_{0,x}$ for $\mathbf{q}_{0,x} \in \mathbb{R}^n$. For the boundary conditions, we employ a similar approach to the velocity field to ensure consistency at the domain boundary for all $t > 0$:

$$
\frac{\partial \mathbf{q}}{\partial \mathbf{n}} = \nabla \mathbf{q} \cdot \hat{\mathbf{n}} = \mathbf{0}
$$

### Discretization

#### Splitting Navier-Stokes Equations

Solving the incompressible Navier-Stokes equations is computationally challenging due to the coupling of velocity and pressure. To simplify this, we employ a technique known as operator splitting. Instead of solving the full equations at once, we decompose them into several simpler components and solve each sequentially.

The momentum equation is thus broken down into the following parts:

* **External Forces**: This step accounts for how external forces $\mathbf{F}$ influence the velocity field over time:
  $$\frac{\partial \mathbf{u}}{\partial t} = \frac{1}{\rho} \mathbf{F}$$
* **Advection**: This describes the process of 'transport', where the velocity field moves itself along its own flow:
  $$\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} = \mathbf{0}$$
* **Diffusion**: This accounts for the viscous spreading of velocity, effectively smoothing out the velocity field:
  $$\frac{\partial \mathbf{u}}{\partial t} = \nu \Delta \mathbf{u}$$
* **Projection**: This step ensures that the fluid remains incompressible. Since the incompressibility constraint is enforced via the pressure equation, this process is often referred to as projection. It describes the influence of pressure on velocity:
  $$\frac{\partial \mathbf{u}}{\partial t} = -\frac{1}{\rho} \nabla p \quad \text{subject to} \quad \nabla \cdot \mathbf{u} = 0$$

Similarly, for the passive quantities $\mathbf{q}$, we break the evolution down into the following parts:

* **External Sources**: This step accounts for how external sources $\mathbf{Q}$ are added to the quantity over time:
  $$\frac{\partial \mathbf{q}}{\partial t} = \mathbf{Q}$$
* **Advection**: This describes the process of 'transport', where the quantity is carried by the velocity field along its flow:
  $$\frac{\partial \mathbf{q}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{q} = \mathbf{0}$$
* **Diffusion**: This accounts for the viscous spreading of the quantity, effectively smoothing it out:
  $$\frac{\partial \mathbf{q}}{\partial t} = \nu_q \Delta \mathbf{q}$$

In this splitting scheme, the output of one component serves as the input for the next. Consequently, the order in which these components are solved is critical for the stability and behavior of the simulation. Our implementation follows the 'stable fluids' pipeline proposed by Jos Stam (see [2]), applying the operations in the following sequence:

$$
\text{External Forces} \to \text{Advection of } \mathbf{u} \to \text{Diffusion of } \mathbf{u} \to \text{Projection of } \mathbf{u}
$$

and then, using the updated values of $\mathbf{u}$:

$$
\text{External Sources} \to \text{Advection of } \mathbf{q} \to \text{Diffusion of } \mathbf{q}
$$

This specific pipeline is chosen because it allows for an unconditionally stable simulation, which is especially important for long-term simulations within computer graphics where visual stability is prioritized over strict physical precision.

#### Spatial, Temporal and Field Discretization

The Navier-Stokes equations are defined as partial differential equations involving fields that depend continuously on space $\mathbf{x}$ and time $t$. To implement these numerically, we must discretize the spatial domain $\Omega$ and the temporal domain $\mathbb{R}_{\geq 0}$, which in turn allows us to discretize the fields.

###### Spatial Discretization

We assume a rectangular domain $\Omega = [0, \Omega_w] \times [0, \Omega_h]$ aligned with the coordinate axes, where $\Omega_w$ and $\Omega_h$ represent the width and height of the domain, respectively. The coordinate system is defined such that the positive x-axis points to the right and the positive y-axis points upwards.

The continuous domain $\Omega$ is discretized into a grid $G = (G_w, G_h)$, where $G_w$ and $G_h$ denote the number of grid cells in the horizontal and vertical directions. The width $\delta x$ and height $\delta y$ of an individual grid cell are given by:

$$
\delta x = \frac{\Omega_w}{G_w} \quad \text{and} \quad \delta y = \frac{\Omega_h}{G_h}
$$

###### Temporal Discretization

For the temporal domain $\mathbb{R}_{\geq 0}$, we assume a fixed time step $\delta t$. This results in a sequence of discrete time points:

$$
t_n = n\,\delta t, \quad n = 0, 1, 2, \dots
$$

###### Field Discretization

With the spatial and temporal domains discretized, we can now discretize the fields. We introduce the notation $[\,\cdot\,]_{i,j}$ as the discretization operator for space and $[\,\cdot\,]^{(n)}$ for time. These can be combined to discretize a field with respect to both variables simultaneously via $[\,\cdot\,]_{i,j}^{(n)}$.

Taking the velocity field $\mathbf{u}$ as an example, we define:

$$
\left[ \mathbf{u} \right]_{i,j}^{(n)} = \mathbf{u}(\mathbf{x}_{i,j}, t_n) \quad \text{where} \quad \mathbf{x}_{i,j} = (x_i, y_j)
$$

For brevity in the following sections, we will often omit the brackets and simply write $\mathbf{u}_{i,j}^{(n)}$ to refer to the discretized velocity. Similarly, the pressure field $p$ is discretized as $[p]_{i,j}^{(n)} = p(\mathbf{x}_{i,j}, t_n)$, or more simply, $p_{i,j}^{(n)}$, and a quantity $\mathbf{q}$ as $[\mathbf{q}]_{i,j}^{(n)} = \mathbf{q}(\mathbf{x}_{i,j}, t_n)$, or more simply, $\mathbf{q}_{i,j}^{(n)}$.

##### Staggered Grid

An important detail yet to be specified is the placement of the grid points $\mathbf{x}_{i,j}$ inside each cell. We take the top-right corner of each cell as a baseline, i.e. $(i\,\delta x, j\,\delta y)$. While one could place all grid points for each field at the center of each cell (a collocated grid), i.e. $\mathbf{x}_{i,j} = ((i - 0.5)\,\delta x, (j - 0.5)\,\delta y)$, fluid simulations commonly employ a staggered grid, where each grid is shifted differently.

The staggered grid is ideal for computing the divergence required by the incompressibility constraint, as it avoids pressure oscillations that can occur on collocated grids. To achieve this, we use three distinct types of grids based on the field being stored:

* The pressure and quantity grid points are stored at the center of each cell. The grid points are thus located at $\mathbf{x}_{i,j} = ((i - 0.5)\,\delta x, (j - 0.5)\,\delta y)$.
* The horizontal velocity grid points are stored on the vertical edge lines, i.e. shifted $\frac{\delta y}{2}$ downwards from the cell corner. The grid points are thus located at $\mathbf{x}_{i,j} = (i\delta x, (j - 0.5)\,\delta y)$.
* The vertical velocity grid points are stored on the horizontal edge lines, i.e. shifted $\frac{\delta x}{2}$ to the left of the cell corner. The grid points are thus located at $\mathbf{x}_{i,j} = ((i - 0.5)\delta x, j\,\delta y)$.

Since the velocity grid points are stored at edges, we require an additional column or row for these grids to cover the entire simulation domain boundary. This arrangement results in different grid dimensions for the respective fields:

* A $(G_w, G_h)$ grid for the pressure field $p$ and quantity field $\mathbf{q}$.
* A $(G_w + 1, G_h)$ grid for the horizontal velocity $u$.
* A $(G_w, G_h + 1)$ grid for the vertical velocity $v$.

Note that by abuse of notation the width and height of each grid cell is the same for all fields, although each field has a different number of grid cells.

##### Boundary Strip

To implement boundary conditions, we require values on the domain boundary. However, due to the staggered grid, horizontal velocities only exist at vertical boundary edges and vertical velocities only exist at horizontal boundary edges. To be able to compute horizontal velocities on the horizontal boundary edges and vertical velocities on the vertical boundary edges, we enlarge the grids by adding an additional 'boundary strip' around the perimeter. This allows us to compute these velocities on the boundary by averaging the values from the boundary strip and the outermost values within the domain. The cells part of the boundary strip are sometimes also called ghost cells.

This introduces one additional column on both the left and right sides of each grid, as well as one additional row on both the top and bottom. Since we already have an additional column or row for the velocity fields, these would theoretically turn them into grids of size $(G_w + 3, G_h + 2)$ or $(G_w + 2, G_h + 3)$, respectively. However, it turns out that $(G_w + 2, G_h + 2)$ grids are sufficient in these cases. Because the additional column or row was only introduced to have the velocities lie on the corresponding domain boundary edges, we do not require ghost cells beyond those existing columns or rows. However, ghost cells are still required for components that do not lie exactly on the domain boundary. To summarize, the resulting dimensions for the fields are:

* A $(G_w + 2, G_h + 2)$ grid for the pressure field $p$ and quantity field $\mathbf{q}$.
* A $(G_w + 2, G_h + 2)$ grid for the horizontal velocity $u$.
* A $(G_w + 2, G_h + 2)$ grid for the vertical velocity $v$.

Note that this does not change the size of the domain itself. The grid points within the boundary strip (except the velocities lying exactly on the domain boundary) are outside the domain.

As previously established, we maintain a consistent notation where the width and height of each cell are assumed to be the same across all fields. With these final grid sizes determined, we can specify the exact spatial placement of the required grid points $\mathbf{x}_{i,j}$ for simulation and their range for each field within the domain and boundary strip:

* **Pressure field and quantity field**:
  $$\mathbf{x}_{i,j} = ((i - 0.5)\,\delta x, (j - 0.5)\,\delta y) \quad \text{for} \quad i = 0, \dots,G_w + 1 \quad \text{and} \quad j = 0, \dots,G_h + 1$$
* **Horizontal velocity field**:
  $$\mathbf{x}_{i,j} = (i\,\delta x, (j - 0.5)\,\delta y) \quad \text{for} \quad i = 0, \dots,G_w \quad \text{and} \quad j = 0, \dots,G_h + 1$$
* **Vertical velocity field**:
  $$\mathbf{x}_{i,j} = ((i - 0.5)\,\delta x, j\,\delta y) \quad \text{for} \quad i = 0, \dots,G_w + 1 \quad \text{and} \quad j = 0, \dots,G_h$$

It is important to note that the allocated memory for the horizontal and vertical velocity grids is slightly larger than the range of grid points specified above. For horizontal velocities, values for $i = G_w + 1$ could also be placed within the grid and similarly, for vertical velocities, values for $j = G_h + 1$. While these additional grid points are not required for the simulation calculations, allowing a potential opportunity for storage optimization, we utilize the larger grids to simplify the indexing logic throughout the implementation.

#### Finite-Difference Scheme

Having discretized the spatial and temporal domains as well as the fields themselves, we must now discretize the differential operators. The discrete representations derived in the previous sections allow us to replace continuous derivatives with finite-difference approximations, enabling the numerical solution of the split equations.

Applying the discretization operator to the four components of the splitting scheme yields the following set of discrete equations:

1. **External Forces and Sources**: 
   $$\left[ \frac{\partial \mathbf{u}}{\partial t} \right]^{(n)}_{i,j} = \frac{1}{\rho} \mathbf{F} \quad \text{and} \quad \left[ \frac{\partial \mathbf{q}}{\partial t} \right]^{(n)}_{i,j} = \mathbf{Q}$$
2. **Advection**: 
   $$\left[ \frac{\partial \mathbf{u}}{\partial t} \right]^{(n)}_{i,j} + \left[ \mathbf{u} \cdot \nabla \mathbf{u} \right]^{(n)}_{i,j} = \mathbf{0} \quad \text{and} \quad \left[ \frac{\partial \mathbf{q}}{\partial t} \right]^{(n)}_{i,j} + \left[ \mathbf{u} \cdot \nabla \mathbf{q} \right]^{(n)}_{i,j} = \mathbf{0}$$
3. **Diffusion**: 
   $$\left[ \frac{\partial \mathbf{u}}{\partial t} \right]^{(n)}_{i,j} = \nu \left[ \Delta \mathbf{u} \right]^{(n)}_{i,j} \quad \text{and} \quad \left[ \frac{\partial \mathbf{q}}{\partial t} \right]^{(n)}_{i,j} = \nu_q \left[ \Delta \mathbf{q} \right]^{(n)}_{i,j}$$
4. **Projection**: 
   $$\left[ \frac{\partial \mathbf{u}}{\partial t} \right]^{(n)}_{i,j} = -\frac{1}{\rho} \left[ \nabla p \right]^{(n)}_{i,j} \quad \text{subject to} \quad \left[ \nabla \cdot \mathbf{u} \right]^{(n)}_{i,j} = 0$$

The partial derivatives with respect to time $t$ are discretized using a simple forward finite-difference approximation:

$$\left[ \frac{\partial \mathbf{u}}{\partial t} \right]^{(n)}_{i,j} = \frac{\mathbf{u}^{(n+1)}_{i,j} - \mathbf{u}^{(n)}_{i,j}}{\delta t} \quad \text{and} \quad \left[ \frac{\partial \mathbf{q}}{\partial t} \right]^{(n)}_{i,j} = \frac{\mathbf{q}^{(n+1)}_{i,j} - \mathbf{q}^{(n)}_{i,j}}{\delta t}$$

The specific discretizations for the remaining differential operators, i.e. gradient, divergence, and Laplacian, are detailed in the subsequent subsections.

Once all components have been discretized, the simulation can be solved iteratively. Starting with the given initial conditions $\mathbf{u}_{i,j}^{(0)} = \mathbf{u}_{0,x}$ and $\mathbf{q}_{i,j}^{(0)} = \mathbf{q}_{0,x}$ at $t=0$, we can calculate the states of the velocity field and quantity field at the next time step $\mathbf{u}_{i,j}^{(n+1)}$ and $\mathbf{q}_{i,j}^{(n+1)}$ based on their current values $\mathbf{u}_{i,j}^{(n)}$ and $\mathbf{q}_{i,j}^{(n)}$ for all discrete time points $t_n$.

##### External Forces

Since the partial derivative with respect to time has been discretized using a forward finite-difference approximation, and given that external forces $\mathbf{F}$ contain no spatial differential operators, the update for the velocity field is algebraically straightforward. The state of the velocity components at the next time step, $u^{(n+1)}_{i,j}$ and $v^{(n+1)}_{i,j}$, is calculated as follows:

$$
\begin{aligned}
  u^{(n+1)}_{i,j} &= u^{(n)}_{i,j} + \delta t \frac{1}{\rho} F_1 \\
  v^{(n+1)}_{i,j} &= v^{(n)}_{i,j} + \delta t \frac{1}{\rho} F_2
\end{aligned}
$$

where $F_1$ and $F_2$ denote the first and second components of the external force vector $\mathbf{F}$ at the respective grid point. These updates are applied to all grid points within the simulation domain. Specifically, for the horizontal velocity $u$, the update is computed for indices $(i, j)$ where $i = 1, \dots, G_w - 1$ and $j = 1, \dots, G_h$. For the vertical velocity $v$, the update covers grid points $(i, j)$ where $i = 1, \dots, G_w$ and $j = 1, \dots, G_h - 1$.

A similar approach is applied to the quantity field $\mathbf{q}$. For each scalar component $q$ of the field $\mathbf{q}$ and its corresponding scalar component $Q$ of the external source term $\mathbf{Q}$ at the respective grid point, the value for the next time step is given by:

$$
q^{(n+1)}_{i,j} = q^{(n)}_{i,j} + \delta t\,Q
$$

This update is computed for all grid points in the domain, specifically where $i = 1, \dots, G_w$ and $j = 1, \dots, G_h$.

##### Advection

To solve the advection component, we must address the term $\mathbf{u} \cdot \nabla \mathbf{u}$ (for velocity) and $\mathbf{u} \cdot \nabla \mathbf{q}$ (for quantities). While it is possible to discretize the gradient operator $\nabla$ using standard finite differences, doing so often leads to numerical instabilities. Specifically, such a simulation would be prone to 'blowing up' unless the time step $\delta t$ is kept extremely small to satisfy the Courant-Friedrichs-Lewy (CFL) condition.

To avoid this limitation and ensure unconditional stability, we employ the Semi-Lagrangian method. This approach is rooted in the two primary ways of describing fluid flow:

1. **Eulerian Approach**: This perspective treats the fluid as a field over a fixed grid. We observe properties (like velocity) at specific coordinate positions $\mathbf{x}$ as time $t$ passes, without tracking individual fluid particles.
2. **Lagrangian Approach**: This perspective focuses on the trajectories of individual fluid particles, tracking their properties as they move through space.

While the Navier-Stokes equations are typically expressed in the Eulerian framework for computational efficiency, the Semi-Lagrangian method acts as a hybrid. It utilizes a fixed Eulerian grid but determines the value at each grid point by 'tracing back' along a Lagrangian trajectory to find where the fluid particle currently at $\mathbf{x}_{i,j}$ originated from during the previous time step.

This is how the Semi-Lagrangian method works conceptually:

1. Start at a grid point $\mathbf{x}_{i,j}$ and assume a particle exists there with velocity $\mathbf{u}^{(n)}_{i,j}$.
2. Trace this particle's path backward in time to find its previous position: $\tilde{\mathbf{x}}_{i,j} = \mathbf{x}_{i,j} - \delta t \, \mathbf{u}^{(n)}_{i,j}$.
3. The value of the property at $\mathbf{x}_{i,j}$ for the next time step is simply the value that existed at $\tilde{\mathbf{x}}_{i,j}$ at the previous time step. This assumes that the particle preserves its properties (e.g. velocity or dye concentration) as it travels from $\tilde{\mathbf{x}}$ to $\mathbf{x}$.

For simplicity, we use a linear backtracing scheme. However, higher-order integrators such as Runge-Kutta (RK4) could be used for increased accuracy. Because the backtraced position $\tilde{\mathbf{x}}_{i,j}$ rarely coincides exactly with a grid point, we use bilinear interpolation between the four nearest neighbors to determine the value. This interpolation is precisely what grants the method its unconditional stability: since the new value is always a weighted average of existing values, it can never exceed the local maximum or minimum, preventing numerical instabilities.

For a more realistic simulation, one should take an average of the neighboring velocities when calculating the backtraced positions. However, given that a linear backtracing scheme is employed and visual stability is prioritized over strict physical accuracy in this project, this approximation is acceptable.

We now formulate the Semi-Lagrangian method by applying it separately to each component of the velocity field and the quantity field, accounting for their specific placements on the staggered grid, to calculate the next time step. During this process, we use the ceiling function $\lceil \cdot \rceil$, which computes the least integer greater than or equal to its argument.

**Horizontal Velocity Advection**

1. Determine the location of grid points $(i, j)$ within the domain for horizontal velocities $u^{(n)}_{i,j}$. These are $x_i = i\,\delta x$ and $y_j = (j - 0.5)\,\delta y$ for  $i = 1, \dots, G_w - 1$ and $j = 1, \dots, G_h$.
2. Trace back the position for each of those grid points:
   $$
   \begin{aligned}
     \tilde{x}_i &= x_i - \delta t \, u^{(n)}_{i,j} = i\,\delta x - \delta t \, u^{(n)}_{i,j} \\
     \tilde{y}_j &= y_j - \delta t \, v^{(n)}_{i,j} = (j - 0.5)\,\delta y - \delta t \, v^{(n)}_{i,j}
   \end{aligned}
   $$
   To ensure they stay within the domain, clamp $(\tilde{x}_i,\tilde{y}_{j})$ to $[0, \Omega_w] \times [0, \Omega_h]$.
3. Find the top-right index for each traced-back position by:
   $$
   \tilde{i} = \left\lceil \frac{\tilde{x}_i}{\delta x} \right\rceil \quad \text{and} \quad \tilde{j} = \left\lceil \frac{\tilde{y}_j}{\delta y} + 0.5 \right\rceil
   $$
   Clamp $\tilde{i}$ to $\tilde{i} = 1, \dots, G_w$ to ensure it is the correct index at $\tilde{x}_i = \Omega_w$. Using the four nearest neighbors:
   $$
   \begin{aligned}
     x_l &= (\tilde{i} - 1.0) \, \delta x, \quad x_r = \tilde{i} \, \delta x \\
     y_b &= (\tilde{j} - 1.5) \, \delta y, \quad y_t = (\tilde{j} - 0.5) \, \delta y
   \end{aligned}
   $$
   we can calculate $u^{(n+1)}_{i,j}$ by bilinear interpolation:
   $$
   u^{(n+1)}_{i,j} = \frac{1}{\delta x \, \delta y}
   \left(
     (x_r - \tilde{x}_i)(y_t - \tilde{y}_j) u^{(n)}_{\tilde{i},\tilde{j}} +
     (x_r - \tilde{x}_i)(\tilde{y}_j - y_b) u^{(n)}_{\tilde{i},\tilde{j}+1} +
     (\tilde{x}_i - x_l)(y_t - \tilde{y}_j) u^{(n)}_{\tilde{i}+1,\tilde{j}} +
     (\tilde{x}_i - x_l)(\tilde{y}_j - y_b) u^{(n)}_{\tilde{i}+1,\tilde{j}+1}
   \right)
   $$

**Vertical Velocity Advection**

1. Determine the location of grid points $(i, j)$ within the domain for vertical velocities $v^{(n)}_{i,j}$. These are $x_i = (i - 0.5)\,\delta x$ and $y_j = j\,\delta y$ for  $i = 1, \dots, G_w$ and $j = 1, \dots, G_h - 1$.
2. Trace back the position for each of those grid points:
   $$
   \begin{aligned}
     \tilde{x}_i &= x_i - \delta t \, u^{(n)}_{i,j} = (i - 0.5)\,\delta x - \delta t \, u^{(n)}_{i,j} \\
     \tilde{y}_j &= y_j - \delta t \, v^{(n)}_{i,j} = j\,\delta y - \delta t \, v^{(n)}_{i,j}
   \end{aligned}
   $$
   To ensure they stay within the domain, clamp $(\tilde{x}_i,\tilde{y}_{j})$ to $[0, \Omega_w] \times [0, \Omega_h]$.
3. Find the top-right index for each traced-back position by:
   $$
   \tilde{i} = \left\lceil \frac{\tilde{x}_i}{\delta x} + 0.5 \right\rceil \quad \text{and} \quad \tilde{j} = \left\lceil \frac{\tilde{y}_j}{\delta y} \right\rceil
   $$
   Clamp $\tilde{j}$ to $\tilde{j} = 1, \dots, G_h$ to ensure it is the correct index at $\tilde{y}_j = \Omega_h$. Using the four nearest neighbors:
   $$
   \begin{aligned}
     x_l &= (\tilde{i} - 1.5) \, \delta x, \quad x_r = (\tilde{i} - 0.5) \, \delta x \\
     y_b &= (\tilde{j} - 1.0) \, \delta y, \quad y_t = \tilde{j} \, \delta y
   \end{aligned}
   $$
   we can calculate $v^{(n+1)}_{i,j}$ by bilinear interpolation:
   $$
   v^{(n+1)}_{i,j} = \frac{1}{\delta x \, \delta y}
   \left(
     (x_r - \tilde{x}_i)(y_t - \tilde{y}_j) v^{(n)}_{\tilde{i},\tilde{j}} +
     (x_r - \tilde{x}_i)(\tilde{y}_j - y_b) v^{(n)}_{\tilde{i},\tilde{j}+1} +
     (\tilde{x}_i - x_l)(y_t - \tilde{y}_j) v^{(n)}_{\tilde{i}+1,\tilde{j}} +
     (\tilde{x}_i - x_l)(\tilde{y}_j - y_b) v^{(n)}_{\tilde{i}+1,\tilde{j}+1}
   \right)
   $$

**Quantity Advection**

1. Determine the location of grid points $(i, j)$ within the domain for the quantity field $q^{(n)}_{i,j}$. These are $x_i = (i - 0.5)\,\delta x$ and $y_j = (j - 0.5)\,\delta y$ for  $i = 1, \dots, G_w$ and $j = 1, \dots, G_h$.
2. Trace back the position for each of those grid points:
   $$
   \begin{aligned}
     \tilde{x}_i &= x_i - \delta t \, u^{(n)}_{i,j} = (i - 0.5)\,\delta x - \delta t \, u^{(n)}_{i,j} \\
     \tilde{y}_j &= y_j - \delta t \, v^{(n)}_{i,j} = (j - 0.5)\,\delta y - \delta t \, v^{(n)}_{i,j}
   \end{aligned}
   $$
   To ensure they stay within the domain, clamp $(\tilde{x}_i,\tilde{y}_{j})$ to $[0, \Omega_w] \times [0, \Omega_h]$.
3. Find the top-right index for each traced-back position by:
   $$
   \tilde{i} = \left\lceil \frac{\tilde{x}_i}{\delta x} + 0.5 \right\rceil \quad \text{and} \quad \tilde{j} = \left\lceil \frac{\tilde{y}_j}{\delta y} + 0.5 \right\rceil
   $$
   Clamping indices is not necessary here. Using the four nearest neighbors:
   $$
   \begin{aligned}
     x_l &= (\tilde{i} - 1.5) \, \delta x, \quad x_r = (\tilde{i} - 0.5) \, \delta x \\
     y_b &= (\tilde{j} - 1.5) \, \delta y, \quad y_t = (\tilde{j} - 0.5) \, \delta y
   \end{aligned}
   $$
   we can calculate $q^{(n+1)}_{i,j}$ by bilinear interpolation:
   $$
   q^{(n+1)}_{i,j} = \frac{1}{\delta x \, \delta y}
   \left(
     (x_r - \tilde{x}_i)(y_t - \tilde{y}_j) q^{(n)}_{\tilde{i},\tilde{j}} +
     (x_r - \tilde{x}_i)(\tilde{y}_j - y_b) q^{(n)}_{\tilde{i},\tilde{j}+1} +
     (\tilde{x}_i - x_l)(y_t - \tilde{y}_j) q^{(n)}_{\tilde{i}+1,\tilde{j}} +
     (\tilde{x}_i - x_l)(\tilde{y}_j - y_b) q^{(n)}_{\tilde{i}+1,\tilde{j}+1}
   \right)
   $$

##### Diffusion

The diffusion step accounts for the viscous spreading of velocity and passive quantities, effectively smoothing the fields over time. This requires the discretization of the Laplace operator $\Delta$ for both the velocity field (i.e. $\Delta \mathbf{u}$) and the quantity field (i.e. $\Delta \mathbf{q}$).

It is important to note that in a Semi-Lagrangian framework, the bilinear interpolation used during the advection step introduces an implicit form of 'numerical diffusion'. While this effect is independent of the actual viscosity coefficients $\nu$ or $\nu_q$, it inherently smooths the result. Consequently, for simulations involving low-viscosity fluids, this explicit diffusion step can be omitted to increase performance without significant visual loss in quality.

For higher physical accuracy, we solve the diffusion equation. However, we use an explicit method instead of an implicit one. This means that our solution is only conditionally stable and requires a small time step for convergence. Using an implicit method would avoid this but requires iteratively solving a linear system, which is computationally expensive. Since we have 'numerical diffusion' anyway, we opt here for the less expensive method. By default, we leave diffusion disabled in this project, and it can be enabled only if the user so wishes.

Since the Laplace operator is a linear operator, it is applied to each vector component independently. We discretize the second-order partial derivatives using a second-order central finite difference scheme.

**Velocity Diffusion**

Starting with the horizontal velocity $u$, the discrete Laplacian $\left[ \Delta u \right]^{(n)}_{i,j}$ is defined as the sum of the second derivatives in the $x$ and $y$ directions:

$$
\left[ \Delta u \right]^{(n)}_{i,j}
= \left[ \frac{\partial^2 u}{\partial x^2} \right]^{(n)}_{i,j} + \left[ \frac{\partial^2 u}{\partial y^2} \right]^{(n)}_{i,j}
= \frac{u^{(n)}_{i+1,j} - 2u^{(n)}_{i,j} + u^{(n)}_{i-1,j}}{\delta x^2} + \frac{u^{(n)}_{i,j+1} - 2u^{(n)}_{i,j} + u^{(n)}_{i,j-1}}{\delta y^2}
$$

To compute this for grid points within the domain, i.e. $i = 1, \dots, G_w - 1$ and $j = 1, \dots, G_h$, we require values from the boundary edges where $i \in \{ 0, G_w \}$ and $j = 1, \dots, G_h$, and ghost cells outside the domain where $i = 1, \dots, G_w - 1$ and $j \in \{ 0, G_h + 1 \}$. These auxiliary values are provided by the boundary condition step.

Similarly, for the vertical velocity $v$, the Laplacian is:

$$
\left[ \Delta v \right]^{(n)}_{i,j}
= \left[ \frac{\partial^2 v}{\partial x^2} \right]^{(n)}_{i,j} + \left[ \frac{\partial^2 v}{\partial y^2} \right]^{(n)}_{i,j}
= \frac{v^{(n)}_{i+1,j} - 2v^{(n)}_{i,j} + v^{(n)}_{i-1,j}}{\delta x^2} + \frac{v^{(n)}_{i,j+1} - 2v^{(n)}_{i,j} + v^{(n)}_{i,j-1}}{\delta y^2}
$$

This is calculated for interior grid points where $i = 1, \dots, G_w$ and $j = 1, \dots, G_h - 1$, necessitating boundary values for $i = 1, \dots, G_w$ and $j \in \{ 0, G_h \}$, and ghost cell values for $i \in \{ 0, G_w + 1 \}$ and $j = 1, \dots, G_h - 1$.

Integrating these into the momentum equation using a forward Euler step, we obtain the updated velocity components for the next time step:

$$
\begin{aligned}
  u^{(n+1)}_{i,j} &= u^{(n)}_{i,j} + \delta t \, \nu \left( \frac{u^{(n)}_{i+1,j} - 2u^{(n)}_{i,j} + u^{(n)}_{i-1,j}}{\delta x^2} + \frac{u^{(n)}_{i,j+1} - 2u^{(n)}_{i,j} + u^{(n)}_{i,j-1}}{\delta y^2} \right) \\
  v^{(n+1)}_{i,j} &= v^{(n)}_{i,j} + \delta t \, \nu \left( \frac{v^{(n)}_{i+1,j} - 2v^{(n)}_{i,j} + v^{(n)}_{i-1,j}}{\delta x^2} + \frac{v^{(n)}_{i,j+1} - 2v^{(n)}_{i,j} + v^{(n)}_{i,j-1}}{\delta y^2} \right)
\end{aligned}
$$

**Quantity Diffusion**

For each scalar component $q$ of the passive quantity $\mathbf{q}$, the diffusion process is identical in structure. The discrete Laplacian is given by:

$$
\left[ \Delta q \right]^{(n)}_{i,j}
= \left[ \frac{\partial^2 q}{\partial x^2} \right]^{(n)}_{i,j} + \left[ \frac{\partial^2 q}{\partial y^2} \right]^{(n)}_{i,j}
= \frac{q^{(n)}_{i+1,j} - 2q^{(n)}_{i,j} + q^{(n)}_{i-1,j}}{\delta x^2} + \frac{q^{(n)}_{i,j+1} - 2q^{(n)}_{i,j} + q^{(n)}_{i,j-1}}{\delta y^2}
$$

This is calculated for all interior grid points $i = 1, \dots, G_w$ and $j = 1, \dots, G_h$, necessitating ghost cell values for $i = 1, \dots, G_w$ and $j \in \{ 0, G_h + 1\}$, as well as $i \in \{ 0, G_w + 1 \}$ and $j = 1, \dots, G_h$.

The updated quantity field is then computed as:

$$
q^{(n+1)}_{i,j} = q^{(n)}_{i,j} + \delta t \, \nu_q \left( \frac{q^{(n)}_{i+1,j} - 2q^{(n)}_{i,j} + q^{(n)}_{i-1,j}}{\delta x^2} + \frac{q^{(n)}_{i,j+1} - 2q^{(n)}_{i,j} + q^{(n)}_{i,j-1}}{\delta y^2} \right)
$$

##### Projection

Since the output of each preceding step in the pipeline is used as input for the next, it is unlikely that the velocity field remains divergence-free after the addition of external forces, advection, and diffusion. Simply discretizing the momentum equation and adding a pressure component does not automatically satisfy the incompressibility constraint $\nabla \cdot \mathbf{u} = 0$. To incorporate this continuity equation into the simulation, we employ the Chorin projection method (see [3]).

The method assumes that $\bar{\mathbf{u}}_{i,j}^{(n)}$ represents an intermediate velocity field after external forces have been applied and advection and diffusion have been performed. We then seek to calculate the final velocity for the next time step $\mathbf{u}_{i,j}^{(n+1)}$ as follows:

$$
\left[ \frac{\partial \mathbf{u}}{\partial t} \right]^{(n)}_{i,j} = \frac{\mathbf{u}^{(n+1)}_{i,j} - \bar{\mathbf{u}}^{(n)}_{i,j}}{\delta t} = -\frac{1}{\rho} \left[ \nabla p \right]^{(n)}_{i,j}
\quad \Rightarrow \quad
\mathbf{u}_{i,j}^{(n+1)} = \bar{\mathbf{u}}_{i,j}^{(n)} - \delta t \, \frac{1}{\rho} \left[ \nabla p \right]^{(n)}_{i,j}
$$

To ensure that the resulting field $\mathbf{u}_{i,j}^{(n+1)}$ is divergence-free, we require that $\left[ \nabla \cdot \mathbf{u} \right]^{(n+1)}_{i,j} = 0$. By applying the divergence operator to both sides of the equation above and rearranging the terms, we derive the following Poisson equation for pressure:

$$
\left[ \Delta p \right]^{(n)}_{i,j} = \frac{\rho}{\delta t} \left[ \nabla \cdot \bar{\mathbf{u}} \right]^{(n)}_{i,j}
$$

To determine the boundary conditions for this equation, we multiply both sides by the outward normal vector $\hat{\mathbf{n}}$ on the domain boundary $\Gamma$:

$$
\left[ \nabla p \right]^{(n)}_{i,j} \cdot \hat{\mathbf{n}} = \frac{\rho}{\delta t} \left( \bar{\mathbf{u}}_{i,j}^{(n)} - \mathbf{u}_{i,j}^{(n+1)} \right) \cdot \hat{\mathbf{n}} = 0,
\quad \text{i.e.} \quad
\frac{\partial p^{(n)}_{i,j}}{\partial \mathbf{n}} = 0
$$

This result follows from the boundary conditions of the velocity field, which imply that both the intermediate and final velocities have no component normal to the boundary, i.e. $\bar{\mathbf{u}}_{i,j}^{(n)} \cdot \hat{\mathbf{n}} = 0$ and $\mathbf{u}_{i,j}^{(n+1)} \cdot \hat{\mathbf{n}} = 0$. This effectively establishes a Neumann boundary condition for the pressure field.

To solve the Poisson equation numerically, we discretize the involved operators. We employ a second-order central difference scheme for the Laplace operator:

$$
\left[ \Delta p \right]^{(n)}_{i,j}
= \left[ \frac{\partial^2 p}{\partial x^2} \right]^{(n)}_{i,j} + \left[ \frac{\partial^2 p}{\partial y^2} \right]^{(n)}_{i,j}
= \frac{p^{(n)}_{i+1,j} - 2p^{(n)}_{i,j} + p^{(n)}_{i-1,j}}{\delta x^2} + \frac{p^{(n)}_{i,j+1} - 2p^{(n)}_{i,j} + p^{(n)}_{i,j-1}}{\delta y^2}
$$

For the divergence, we avoid the standard backward finite difference and instead leverage the staggered grid. Because the velocity components are positioned on cell edges, their averages align with the center of the grid point exactly, allowing for a more natural discretization:

$$
\left[ \nabla \cdot \bar{\mathbf{u}} \right]^{(n)}_{i,j}
= \left[ \frac{\partial \bar{u}}{\partial x} \right]^{(n)}_{i,j} + \left[ \frac{\partial \bar{v}}{\partial y} \right]^{(n)}_{i,j}
= \frac{u^{(n)}_{i,j} - u^{(n)}_{i-1,j}}{\delta x} + \frac{v^{(n)}_{i,j} - v^{(n)}_{i,j-1}}{\delta y}
$$

The resulting system of linear equations is solved using a weighted Jacobi method. This iterative solver is chosen because it can be easily parallelized on the GPU. The method introduces a weight $w \in [0,1]$ to improve stability. To apply the solver, we rearrange the discretized Poisson equation to isolate $p^{(n)}_{i,j}$:

$$
p^{(n),k+1}_{i,j} = (1 - w) \, p^{(n),k}_{i,j} + w \, \left( \frac{1}{2} \frac{1}{\delta x^2 + \delta y^2} \left( \delta y^2 \left( p^{(n),k}_{i+1,j} + p^{(n),k}_{i-1,j} \right) + \delta x^2 \left( p^{(n),k}_{i,j+1} + p^{(n),k}_{i,j-1} \right) - \delta x^2 \, \delta y^2 \, \frac{\rho}{\delta t} \left[ \nabla \cdot \bar{\mathbf{u}} \right]^{(n)}_{i,j} \right) \right)
$$

In this formula, the superscript $k$ denotes the current iteration step. The iteration is terminated after a configurable number of steps. While one could theoretically stop when the difference between successive iterations falls below a certain threshold (indicating convergence), this typically requires too many iterations for a real-time simulation. 

For the first simulation step, we initialize the pressure with $p^{(n), 0}_{i,j} = 0$. For all subsequent steps, we use the pressure values from the previous time step as the starting point to accelerate convergence. Note that since $\bar{\mathbf{u}}$ is fixed during this process, the divergence term remains unchanged across all iteration steps. Regarding the weight $w$, a value of $w=1$ corresponds to the standard Jacobi method, while $w < 1$ ensures more stable behavior without significantly slowing down convergence.

The iterative solver is calculated for all interior grid points $i = 1, \dots, G_w$ and $j = 1, \dots, G_h$. This requires ghost cell values for the pressure field at the boundaries: specifically for $i = 1, \dots, G_w$ with $j \in \{ 0, G_h + 1\}$, and for $j = 1, \dots, G_h$ with $i \in \{ 0, G_w + 1 \}$.

Once the pressure field $p^{(n)}_{i,j}$ has been computed, we perform the actual projection to make the velocity field divergence-free. This requires discretizing the pressure gradient:

$$
\left[ \nabla p \right]^{(n)}_{i,j}
= \left( \left[ \frac{\partial p}{\partial x} \right]^{(n)}_{i,j} , \left[ \frac{\partial p}{\partial y} \right]^{(n)}_{i,j} \right)
= \left( \frac{p^{(n)}_{i+1,j} - p^{(n)}_{i,j}}{\delta x}, \frac{p^{(n)}_{i,j+1} - p^{(n)}_{i,j}}{\delta y} \right)
$$

Plugging this into the update equation yields the final velocity components for the next time step:

$$
\begin{aligned}
  u_{i,j}^{(n+1)} &= \bar{u}_{i,j}^{(n)} - \delta t \, \frac{1}{\rho} \frac{p^{(n)}_{i+1,j} - p^{(n)}_{i,j}}{\delta x} \\
  v_{i,j}^{(n+1)} &= \bar{v}_{i,j}^{(n)} - \delta t \, \frac{1}{\rho} \frac{p^{(n)}_{i,j+1} - p^{(n)}_{i,j}}{\delta y}
\end{aligned}
$$

The projection is applied to all interior points. Specifically, $u_{i,j}^{(n+1)}$ is calculated for $i = 1, \dots, G_w - 1$ and $j = 1, \dots, G_h$, while $v_{i,j}^{(n+1)}$ is calculated for $i = 1, \dots, G_w$ and $j = 1, \dots, G_h - 1$. This process only requires access to the interior points of the pressure grid.

##### Boundary Conditions

As established in the theoretical foundation, we employ free-slip boundary conditions to simulate the interaction between the fluid and the domain boundary. 

It is critical that these boundary conditions are applied every time the interior grid values of a field are updated. This occurs after each major step in the pipeline, such as advection or diffusion. In the case of the pressure field, since it is solved iteratively, the boundary values must be updated after every single iteration of the weighted Jacobi solver to ensure correctness.

While all fields in this project effectively follow Neumann-type constraints, the specific discretizations required to implement these conditions differ depending on the placement of the grid points for each field. For the velocity fields, we must compute values both directly on the domain boundary and within the ghost cells of the boundary strip.

**Horizontal Velocity Boundary Conditions**

The horizontal velocity $u$ is defined on a $(G_w + 2, G_h + 2)$ grid. The grid points located exactly on the vertical boundaries are those where $i \in \{ 0, G_w \}$ and $j = 1, \dots, G_h$. Since these velocity components are normal to the boundary, the condition $\mathbf{u}_n = \mathbf{0}$ implies:

$$
u^{(n)}_{i,j} = 0 \quad \text{for} \quad i \in \{ 0, G_w \}
$$

The ghost cells for the horizontal velocity are located at $i = 1, \dots, G_w - 1$ and $j \in \{ 0, G_h + 1 \}$. At these positions, the horizontal velocity is tangential to the boundary. Applying the condition $\frac{\partial \mathbf{u}_t}{\partial \mathbf{n}} = 0$ for the normal vectors $\hat{\mathbf{n}} = (0, 1)$ and $\hat{\mathbf{n}} = (0, -1)$, we obtain:

$$
\begin{aligned}
  \frac{u^{(n)}_{i,G_h+1} - u^{(n)}_{i,G_h}}{\delta y} &= 0 \quad \Rightarrow \quad u^{(n)}_{i,G_h+1} = u^{(n)}_{i,G_h} \\
  \frac{u^{(n)}_{i,0} - u^{(n)}_{i,1}}{\delta y} &= 0 \quad \Rightarrow \quad u^{(n)}_{i,0} = u^{(n)}_{i,1}
\end{aligned}
$$

Finally, the values for the corner cells are computed by averaging their neighbors while accounting for the boundary conditions already established:

$$
\begin{aligned}
  u^{(n)}_{0,0} &= 0.5 \left( u^{(n)}_{1,0} + u^{(n)}_{0,1} \right) = 0.5 \, u^{(n)}_{1,0} = 0.5 \, u^{(n)}_{1,1} \\
  u^{(n)}_{G_w,0} &= 0.5 \left( u^{(n)}_{G_w-1,0} + u^{(n)}_{G_w,1} \right) = 0.5 \, u^{(n)}_{G_w-1,0} = 0.5 \, u^{(n)}_{G_w-1,1} \\
  u^{(n)}_{0,G_h+1} &= 0.5 \left( u^{(n)}_{0,G_h} + u^{(n)}_{1,G_h+1} \right) = 0.5 \, u^{(n)}_{1,G_h+1} = 0.5 \, u^{(n)}_{1,G_h} \\
  u^{(n)}_{G_w,G_h+1} &= 0.5 \left( u^{(n)}_{G_w-1,G_h+1} + u^{(n)}_{G_w,G_h} \right) = 0.5 \, u^{(n)}_{G_w-1,G_h+1}  = 0.5 \, u^{(n)}_{G_w-1,G_h} \\
\end{aligned}
$$

**Vertical Velocity Boundary Conditions**

The vertical velocity $v$ is similarly handled. The grid points on the horizontal boundaries are those where $i = 1, \dots, G_w$ and $j \in \{ 0, G_h \}$. Because these components are normal to the boundary, we have:

$$
v^{(n)}_{i,j} = 0 \quad \text{for} \quad j \in \{ 0, G_h \}
$$

The ghost cells are located at $i \in \{ 0, G_w + 1 \}$ and $j = 1, \dots, G_h - 1$. Here, the vertical velocity is tangential to the boundary. Applying $\frac{\partial \mathbf{u}_t}{\partial \mathbf{n}} = 0$ for $\hat{\mathbf{n}} = (1, 0)$ and $\hat{\mathbf{n}} = (-1, 0)$, we get:

$$
\begin{aligned}
  \frac{v^{(n)}_{G_w+1,j} - v^{(n)}_{G_w,j}}{\delta x} &= 0 \quad \Rightarrow \quad v^{(n)}_{G_w+1,j} = v^{(n)}_{G_w,j} \\
  \frac{v^{(n)}_{0,j} - v^{(n)}_{1,j}}{\delta x} &= 0 \quad \Rightarrow \quad v^{(n)}_{0,j} = v^{(n)}_{1,j}
\end{aligned}
$$

The corner cells are then averaged from their neighbors:

$$
\begin{aligned}
  v^{(n)}_{0,0} &= 0.5 \left( v^{(n)}_{0,1} + v^{(n)}_{1,0} \right) = 0.5 \, v^{(n)}_{0,1} = 0.5 \, v^{(n)}_{1,1} \\
  v^{(n)}_{G_w+1,0} &= 0.5 \left( v^{(n)}_{G_w,0} + v^{(n)}_{G_w+1,1} \right) = 0.5 \, v^{(n)}_{G_w+1,1} = 0.5 \, v^{(n)}_{G_w,1} \\
  v^{(n)}_{0,G_h} &= 0.5 \left( v^{(n)}_{0,G_h-1} + v^{(n)}_{1,G_h} \right) = 0.5 \, v^{(n)}_{0,G_h-1} = 0.5 \, v^{(n)}_{1,G_h-1}  \\
  v^{(n)}_{G_w+1,G_h} &= 0.5 \left( v^{(n)}_{G_w+1,G_h-1} + v^{(n)}_{G_w,G_h} \right) = 0.5 \, v^{(n)}_{G_w+1,G_h-1} = 0.5 \, v^{(n)}_{G_w,G_h-1} \\
\end{aligned}
$$

**Quantity and Pressure Boundary Conditions**

Since the grid points for quantity fields $\mathbf{q}$ and the pressure field $p$ are positioned at the center of each cell, there are no grid points located exactly on the domain boundary. Consequently, we only need to compute values for the ghost cells in the boundary strip. 

The following derivations use the quantity field $\mathbf{q}$, but it should be noted that identical conditions apply to the pressure field $p$ by simply replacing $\mathbf{q}$ with $p$. We calculate the values for indices $i = 1, \dots, G_w, j \in \{ 0, G_h + 1\}$ and $i \in \{ 0, G_w + 1 \}, j = 1, \dots, G_h$ using the condition $\frac{\partial \mathbf{q}_t}{\partial \mathbf{n}} = \mathbf{0}$ for all four boundary normal vectors:

$$
\begin{aligned}
  \frac{\mathbf{q}^{(n)}_{0,j} - \mathbf{q}^{(n)}_{1,j}}{\delta x} &= \mathbf{0} \quad \Rightarrow \quad \mathbf{q}^{(n)}_{0,j} = \mathbf{q}^{(n)}_{1,j} \\
  \frac{\mathbf{q}^{(n)}_{G_w+1,j} - \mathbf{q}^{(n)}_{G_w,j}}{\delta x} &= \mathbf{0} \quad \Rightarrow \quad \mathbf{q}^{(n)}_{G_w+1,j} = \mathbf{q}^{(n)}_{G_w,j} \\
  \frac{\mathbf{q}^{(n)}_{i,G_h+1} - \mathbf{q}^{(n)}_{i,G_h}}{\delta y} &= \mathbf{0} \quad \Rightarrow \quad \mathbf{q}^{(n)}_{i,G_h+1} = \mathbf{q}^{(n)}_{i,G_h} \\
  \frac{\mathbf{q}^{(n)}_{i,0} - \mathbf{q}^{(n)}_{i,1}}{\delta y} &= \mathbf{0} \quad \Rightarrow \quad \mathbf{q}^{(n)}_{i,0} = \mathbf{q}^{(n)}_{i,1}
\end{aligned}
$$

The corner cells are derived by averaging their neighbors. Given the equalities above, these corners effectively simplify to the value of the nearest interior cell:

$$
\begin{aligned}
  \mathbf{q}^{(n)}_{0,0} &= 0.5 \left( \mathbf{q}^{(n)}_{0,1} + \mathbf{q}^{(n)}_{1,0} \right) = 0.5 \left( \mathbf{q}^{(n)}_{1,1} + \mathbf{q}^{(n)}_{1,1} \right) = \mathbf{q}^{(n)}_{1,1} \\
  \mathbf{q}^{(n)}_{G_w+1,0} &= 0.5 \left( \mathbf{q}^{(n)}_{G_w,0} + \mathbf{q}^{(n)}_{G_w+1,1} \right) = 0.5 \left( \mathbf{q}^{(n)}_{G_w,1} + \mathbf{q}^{(n)}_{G_w,1} \right) = \mathbf{q}^{(n)}_{G_w,1} \\
  \mathbf{q}^{(n)}_{0,G_h+1} &= 0.5 \left( \mathbf{q}^{(n)}_{0,G_h} + \mathbf{q}^{(n)}_{1,G_h+1} \right) = 0.5 \left( \mathbf{q}^{(n)}_{1,G_h} + \mathbf{q}^{(n)}_{1,G_h} \right) = \mathbf{q}^{(n)}_{1,G_h}  \\
  \mathbf{q}^{(n)}_{G_w+1,G_h+1} &= 0.5 \left( \mathbf{q}^{(n)}_{G_w+1,G_h} + \mathbf{q}^{(n)}_{G_w,G_h+1} \right) = 0.5 \left( \mathbf{q}^{(n)}_{G_w,G_h} + \mathbf{q}^{(n)}_{G_w,G_h} \right) = \mathbf{q}^{(n)}_{G_w,G_h} \\
\end{aligned}
$$

### Vorticity Confinement

As noted in the discussion on advection, the Semi-Lagrangian method introduces significant numerical diffusion. This effect causes the fluid to 'lose' fine details over time, as smaller vortices are often dissolved by the interpolation process before they have a chance to fully develop. To mitigate this and re-introduce these missing details, we employ a technique known as vorticity confinement (see [4]). While this approach is not strictly physically accurate, it produces the swirling visual effects typically expected in high-quality computer graphics simulations.

The confinement process is implemented as an additional external force $\mathbf{F}_{\mathrm{vc}}$, which is added to the momentum equation alongside other external forces. To compute this force, we first determine the vorticity $\omega$ of the velocity field $\mathbf{u}$:

$$
\omega = \nabla \times \mathbf{u} = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
$$

Because the simulation is two-dimensional, the vorticity is treated as a scalar field representing the local rotation of the fluid. To 'confine' these rotations, we identify areas with high vorticity gradients and apply a force that strengthens the existing curls around those regions:

$$
\mathbf{F}_{\mathrm{vc}}
= \alpha_{\mathrm{vc}} \left( \frac{\nabla \Vert \omega \Vert}{\Vert \nabla \Vert \omega \Vert \Vert} \times \begin{pmatrix} 0 \\ 0 \\ \omega \end{pmatrix} \right)
= \frac{\alpha_{\mathrm{vc}}}{\Vert \nabla \Vert \omega \Vert \Vert} \begin{pmatrix} \phantom{-}\frac{\partial \Vert \omega \Vert}{\partial y} \, \omega \\ -\frac{\partial \Vert \omega \Vert}{\partial x} \, \omega \end{pmatrix}
$$

In this expression, $\alpha_{\mathrm{vc}} \in \mathbb{R}_{\geq 0}$ is a coefficient used to control the strength of the confinement effect. Since $\omega$ is a scalar in our 2D domain, we treat it mathematically as a vector field where only the third component is non-zero. This represents vortices that 'swirl' around the z-axis, which is orthogonal to the simulation plane. The resulting cross product produces a vector with a zero third component, effectively returning us to a two-dimensional force vector.

To implement this numerically, we place the grid points for the vorticity field at the center of each cell, similar to the quantity and pressure fields. We discretize $\omega$ using a second-order central finite difference scheme:

$$
\left[ \omega \right]^{(n)}_{i,j}
= \left[ \frac{\partial v}{\partial x} \right]^{(n)}_{i,j} - \left[ \frac{\partial u}{\partial y} \right]^{(n)}_{i,j}
= \frac{v_{i+1,j} - v_{i-1,j}}{2\,\delta x} - \frac{u_{i,j+1} - u_{i,j-1}}{2\,\delta y}
$$

To calculate the gradient of the vorticity magnitude, we employ a forward difference approximation:

$$
\left[ \nabla \Vert \omega \Vert \right]^{(n)}_{i,j}
= \left( \left[ \frac{\partial \Vert \omega \Vert}{\partial x} \right]^{(n)}_{i,j}, \left[ \frac{\partial \Vert \omega \Vert}{\partial y} \right]^{(n)}_{i,j} \right)
= \left( \frac{{\Vert \omega \Vert}_{i+1,j} - {\Vert \omega \Vert}_{i,j}}{\delta x}, \frac{{\Vert \omega \Vert}_{i,j+1} - {\Vert \omega \Vert}_{i,j}}{\delta y} \right)
$$

Using these two discretizations, we can compute the force $\left[ \mathbf{F}_{\mathrm{vc}} \right]^{(n)}_{i,j}$ for all interior grid points where $i = 1, \dots, G_w$ and $j = 1, \dots, G_h$. This computation necessitates ghost cell values for indices $i = 1, \dots, G_w$ with $j \in \{ 0, G_h + 1\}$, as well as $j = 1, \dots, G_h$ with $i \in \{ 0, G_w + 1 \}$.

To obtain these ghost cell values, we apply the same boundary conditions used for the pressure and quantity fields. This ensures consistency across all fields whose grid points are positioned at the center of each cell.

## Implementation

#### Tech Stack

The software is implemented using C++20. Newer standards such as C++23 or C++26 were not adopted because the NVCC compiler does not yet provide full support for these versions. Build orchestration is handled by CMake, while vcpkg is used for dependency management.

The simulation core is accelerated via CUDA and requires a GPU with compute capability 7.5 or higher. For visualization, OpenGL 4.6 is employed in conjunction with CUDA-OpenGL interoperability to allow the GPU to handle both simulation and rendering without intermediate host intervention.

To keep the project lightweight and maintainable, third-party dependencies are kept to a minimum and standard library features are prioritized. Outside of the CUDA Toolkit, external libraries are used only where implementing equivalent functionality would be counterproductive to the project's primary goals:

* **GLFW**: Used for window creation and the handling of keyboard and mouse input.
* **glad**: Used as an OpenGL loader to access modern function pointers.
* **Dear ImGui**: Used to implement a menu, allowing for the adjustment of simulation parameters.

#### Design Goals

The implementation is guided by two primary design goals:

1. **Decoupling of Logic**: The simulation and rendering logic should be as decoupled as possible. This ensures a clean separation of concerns, where the simulation core remains agnostic of how the data is visualized.
2. **GPU Residency**: All computations must occur on the GPU without copying data back to the host. Ideally, the rendering logic should consume results directly from GPU memory to avoid the performance bottleneck associated with transfers between host and device.

#### Technical Realization

To achieve these goals, a shared resource in the form of a framebuffer is used as the primary interface between the simulation and the renderer. This architecture allows the simulation logic to solve the Navier-Stokes equations and write the resulting fields into the framebuffer, while the rendering logic reads from that same buffer for visualization. This approach successfully decouples the two systems: the renderer does not need to understand the underlying physics, and the solver does not need to know how the pixels are drawn.

The framebuffer utilizes CUDA-OpenGL interoperability to register an OpenGL texture as a CUDA surface. Since the Navier-Stokes solver is accelerated by CUDA, the results already reside in GPU memory. By copying these results directly to the registered surface, the system completely avoids the need to transfer data to the host and back to the GPU, ensuring maximum throughput and minimum latency.

This architecture is realized through three core classes:

* **`Solver`**: This class performs the step-wise simulation of the Navier-Stokes equations. Its output is written into an allocated block of GPU memory containing the current state of the fluid.
* **`Renderer`**: This class takes an OpenGL texture handle and renders it as a full-screen quad. It treats the texture as a read-only resource and possesses no knowledge of CUDA kernels or the mathematical details of the simulation.
* **`Framebuffer`**: Acting as the bridge between the `Solver` and the `Renderer`, this class maps the GPU memory block allocated by the solver to an OpenGL texture via the aforementioned interoperability. It provides the texture handle required by the renderer.

#### GPU Implementation

Several optimizations were implemented to maximize the performance of the CUDA kernels:

* **Pitched Memory**: Pitched memory allocation is used for 2D grids to ensure optimal alignment and access patterns across memory rows.
* **Coalesced Access**: To increase contiguous memory access, multiple fields are packed into `float2` or `float4` vector types rather than stored as individual scalar fields. This is particularly effective for velocity and dye fields.
* **Non-coherent Loads**: Non-coherent data loads are used wherever possible for read-only data to reduce cache pressure.
* **Shared Memory**: Shared memory is employed to accelerate memory access for 5-point stencils in the diffusion kernels and during the iterative solving of the pressure equation. Care was taken to organize memory layouts to avoid bank conflicts.
* **Streams**: CUDA streams are used where sensible to overlap computation and data movement.

Finally, for performance reasons, the simulation grid size is set equal to the framebuffer size. While this couples the resolution of the physics to the window size, it provides a significant computational advantage: the spatial steps $\delta x$ and $\delta y$ both become equal to one. This eliminates numerous division operations across every kernel, which is critical as divisions are significantly more expensive than other arithmetic operations on the GPU.

## References

1. 'Numerical Simulation in Fluid Dynamics: A Practical Introduction' (1998) by Michael Griebel, Thomas Dornseifer and Tilman Neunhoeffer.
2. 'Stable fluids' (1999) by Jos Stam.
3. [https://en.wikipedia.org/wiki/Projection_method_(fluid_dynamics)](https://en.wikipedia.org/wiki/Projection_method_(fluid_dynamics))
4. [https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)
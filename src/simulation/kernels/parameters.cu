#include "simulation/kernels/parameters.cuh"

__constant__ f32 dt;
__constant__ f32 density;
__constant__ f32 viscosity;
__constant__ f32 viscosity_dye;
__constant__ f32 confinement;
__constant__ f32 jacobi_weight;
__constant__ f32 external_force_radius;
__constant__ f32 external_dye_radius;

__constant__ f32 dt_over_density;
__constant__ f32 density_over_dt;
__constant__ f32 density_inverse;
__constant__ f32 viscosity_times_dt;
__constant__ f32 viscosity_dye_times_dt;

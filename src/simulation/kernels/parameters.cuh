#pragma once

#include "platform/types.hpp"

// Actual simulation parameters
extern __constant__ f32 dt;
extern __constant__ f32 density;
extern __constant__ f32 viscosity;
extern __constant__ f32 viscosity_dye;
extern __constant__ f32 jacobi_weight;
extern __constant__ f32 external_force_radius;
extern __constant__ f32 external_dye_radius;

// Pre-calculated parameters for performance
extern __constant__ f32 dt_over_density;
extern __constant__ f32 density_over_dt;
extern __constant__ f32 density_inverse;
extern __constant__ f32 viscosity_times_dt;
extern __constant__ f32 viscosity_dye_times_dt;

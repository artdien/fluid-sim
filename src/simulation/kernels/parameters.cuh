#pragma once

#include "platform/types.hpp"

extern __constant__ f32 dt;
extern __constant__ f32 density;
extern __constant__ f32 density_inverse;
extern __constant__ f32 density_over_dt;
extern __constant__ f32 dt_over_density;
extern __constant__ f32 viscosity;
extern __constant__ f32 viscosity_dye;
extern __constant__ f32 jacobi_weight;

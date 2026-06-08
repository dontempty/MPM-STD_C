#pragma once

#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"
#include "core/mpi_topology.hpp"

namespace mpmstd::post {

// Validation diagnostics. Re_δ* = U_max·δ*/ν (Fig 7 DHVC); central temperature
// deviation Θ_c-Θ_m (Fig 9 NOB RBC). (rev.2 §9b) Global reductions ⇒ global
// normalization.
void compute_Re_delta_star_cpu(real_t& out, const core::CpuField& U, const core::Grid& grid, const core::Subdomain& sub);
void compute_Re_delta_star_gpu(real_t& out, const core::GpuField& U, const core::Grid& grid, const core::Subdomain& sub);

void compute_center_temp_cpu(real_t& out, const core::CpuField& T, const core::Grid& grid, const core::Subdomain& sub);
void compute_center_temp_gpu(real_t& out, const core::GpuField& T, const core::Grid& grid, const core::Subdomain& sub);

} // namespace mpmstd::post

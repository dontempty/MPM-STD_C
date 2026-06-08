#pragma once

#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"

namespace mpmstd::physics {

// LES subgrid-scale viscosity ν_t (rev.2 §9, P9). Composed in main BEFORE
// assemble; feeds μ_eff into assemble_momentum_var_visc. Omit ⇒ DNS.
void compute_sgs_viscosity_cpu(core::CpuField& nu_t,
                               const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                               const core::Grid& grid);
void compute_sgs_viscosity_gpu(core::GpuField& nu_t,
                               const core::GpuField& U, const core::GpuField& V, const core::GpuField& W,
                               const core::Grid& grid);

} // namespace mpmstd::physics

#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"
#include "core/boundary.hpp"

namespace mpmstd::equation {

// Assemble the 3-stage ADI banded system for a scalar transport equation (T):
// convection explicit + diffusion implicit (Crank-Nicolson). kappa is a field
// (NOB variable conductivity) or a constant field (OB). Momentum-independent.
// (rev.2 §5; bodies land in P1/P6/P7.)
void assemble_scalar_system_cpu(core::ScalarSystem& sys,
                                const core::CpuField& T,
                                const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                                const core::CpuField& kappa,
                                const core::Grid& grid, const core::Boundary& bc, real_t dt);

void assemble_scalar_system_gpu(core::ScalarSystem& sys,
                                const core::GpuField& T,
                                const core::GpuField& U, const core::GpuField& V, const core::GpuField& W,
                                const core::GpuField& kappa,
                                const core::Grid& grid, const core::Boundary& bc, real_t dt);

} // namespace mpmstd::equation

#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"

namespace mpmstd::equation {

// Build the pressure-Poisson RHS = div(U*) (+ NOB extra term later) and the
// per-axis transform/wavenumber metadata. (rev.2 §5; body P1/P3.)
void assemble_pressure_system_cpu(core::PressureSystem& poi,
                                  const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                                  const core::Grid& grid, real_t dt);
void assemble_pressure_system_gpu(core::PressureSystem& poi,
                                  const core::GpuField& U, const core::GpuField& V, const core::GpuField& W,
                                  const core::Grid& grid, real_t dt);

} // namespace mpmstd::equation

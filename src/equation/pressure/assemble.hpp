#pragma once

#include "equation/pressure/linear_system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"

namespace mpmstd::equation::pressure {

// Build the pressure-Poisson RHS = div(U*) (+ NOB extra term later) and the
// per-axis transform/wavenumber metadata. (rev.2 §5; body P1/P3.)
void assemble_pressure_system_cpu(equation::pressure::LinearSystem& poi,
                                  const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                                  const grid::Grid& grid, real_t dt);
void assemble_pressure_system_gpu(equation::pressure::LinearSystem& poi,
                                  const core::GpuField& U, const core::GpuField& V, const core::GpuField& W,
                                  const grid::Grid& grid, real_t dt);

} // namespace mpmstd::equation::pressure

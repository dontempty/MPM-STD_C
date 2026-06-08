#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/mpi_topology.hpp"

namespace mpmstd::equation {

// Solve the assembled scalar ADI system in place (T updated): directional sweeps
// + inter-rank communication, calling solve/ banded solvers. (rev.2 §5)
void solve_scalar_cpu(core::ScalarSystem& sys, core::CpuField& T, const core::Subdomain& sub);
void solve_scalar_gpu(core::ScalarSystem& sys, core::GpuField& T, const core::Subdomain& sub);

} // namespace mpmstd::equation

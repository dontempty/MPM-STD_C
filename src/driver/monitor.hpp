#pragma once

#include "core/field_cpu.hpp"
#include "core/mpi_topology.hpp"

namespace mpmstd::driver {

// Case-shared monitor line (step/time/dt + divergence check). Host-side
// orchestration (rev.2 M5). (body P1.)
void print_monitor_cpu(const core::MpiContext& mpi, int step, real_t time, real_t dt, real_t div_max);

} // namespace mpmstd::driver

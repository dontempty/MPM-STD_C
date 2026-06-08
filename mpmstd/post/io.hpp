#pragma once

#include "core/field_cpu.hpp"
#include "core/mpi_topology.hpp"

#include <string>

namespace mpmstd::post {

// Restart / field IO (rev.2 §10, U6: mandatory; target = existing global C-order
// double format). Host-side even in GPU runs (to_host() first) ⇒ CPU only, no
// _gpu twin. (body P1.)
void write_restart_cpu(const core::CpuField& f, const core::Subdomain& sub, const std::string& path);
void read_restart_cpu (core::CpuField& f,       const core::Subdomain& sub, const std::string& path);

} // namespace mpmstd::post

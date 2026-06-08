#include "post/io.hpp"

// P0 skeleton stubs (no-op). P1 ports MPI-IO restart (global C-order double;
// X-fastest↔Z-fastest transpose handled at the converter, see memory).
namespace mpmstd::post {
void write_restart_cpu(const core::CpuField&, const core::Subdomain&, const std::string&) { /* TODO(P1) */ }
void read_restart_cpu (core::CpuField&,       const core::Subdomain&, const std::string&) { /* TODO(P1) */ }
} // namespace mpmstd::post

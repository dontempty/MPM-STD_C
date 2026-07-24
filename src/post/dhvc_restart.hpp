#pragma once

#include "core/fields.hpp"    // core::CpuFields
#include "core/domain.hpp"    // grid::Grid, parallel::mpi::Subdomain

#include <mpi.h>
#include <array>
#include <string>
#include <utility>

namespace mpmstd::post {

// Parameters for the DHVC cross-mesh warm-start interpolation (all read from the
// case config in main; the source mesh is discovered from dir_in/meta.txt).
struct DhvcWarmStartConfig {
  double            Lx = 0.0, Ly = 0.0, Lz = 0.0, gamma3 = 0.0;  // this run's box + z-stretch
  std::string       io = "slab";                                 // "slab" (Tier 1) | "replicate"
  bool              dryrun = false;                              // dump checksums then exit
  std::array<int,3> target_dims{0, 0, 0};                       // this run's n_global (log line)
};

// DHVC host-side restart. Distinct from post::Restart (RBC): the LOAD is a
// cross-mesh warm start — read a previous run's restart (any mesh, discovered from
// meta.txt) and trilinearly interpolate (staggered, x/y periodic, z clamp) onto
// this grid. The SAVE is a run-end checkpoint: gathered .bin (decomposition-
// independent) + meta.txt (dims/step/time/Ra) for the next warm start. CPU-only;
// GPU cases stage device->host first (core::copy_all_to_host).
class DhvcRestart {
public:
  DhvcRestart(std::string dir_in, std::string dir_out)
    : dir_in_(std::move(dir_in)), dir_out_(std::move(dir_out)) {}

  bool warm() const { return !dir_in_.empty(); }   // dir_in given ⇒ warm-start run

  // Interpolating warm-start read into `fields` (U/V/W/T/P). Returns true if a
  // dry-run (checksum dump) was requested — the caller should then exit.
  bool load_warmstart(core::CpuFields& fields, const grid::Grid& grid,
                      const parallel::mpi::Subdomain& sub,
                      const DhvcWarmStartConfig& cfg, MPI_Comm comm) const;

  // Run-end checkpoint: gathered U/V/W/T/P .bin + meta.txt. No-op if dir_out is
  // empty. `fields` must already hold host data (core::copy_all_to_host).
  void save(const core::CpuFields& fields, const parallel::mpi::Subdomain& sub,
            int step, double time, std::array<int,3> n_global, double Ra,
            MPI_Comm comm) const;

private:
  std::string dir_in_, dir_out_;
};

} // namespace mpmstd::post

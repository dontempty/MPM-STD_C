#pragma once

#include "core/fields.hpp"        // core::CpuFields
#include "core/boundary.hpp"      // boundary::BoundaryCondition
#include "core/gpu_binding.hpp"   // parallel::mpi::Subdomain

#include <mpi.h>
#include <string>
#include <utility>
#include <vector>

namespace mpmstd::post {

// Non-field restart payload: step/time plus the case's statistics scalars.
// `extras` preserves registration/file order so the written stats.txt is
// byte-identical to the hand-rolled writer it replaces (RBC registers tc_sum,
// tc_n, wt_sum in that order).
struct RestartState {
  int    step = 0;
  double time = 0.0;
  std::vector<std::pair<std::string, double>> extras;
};

// Host-side field restart I/O. Consolidates the checkpoint trap-avoidance that
// used to sit inline in the RBC main:
//   - gathered write (root-gather + fwrite), NOT the MPI-IO collective write:
//     the matching MPI-IO read deadlocked on this cluster's NFS at np=2/256^3
//     (Report/34 addendum).
//   - P.bin existence decided on ONE rank + broadcast — a per-rank
//     filesystem::exists() that disagrees across ranks (NFS attr-cache skew)
//     makes some ranks enter the collective read while others skip it → deadlock.
// CPU-only by team layout (I/O is a host concern); GPU cases stage device→host
// first via core::copy_all_to_host.
class Restart {
public:
  Restart(std::string dir_in, std::string dir_out, int nrestart_write)
    : dir_in_(std::move(dir_in)), dir_out_(std::move(dir_out)),
      nrestart_write_(nrestart_write) {}

  bool warm() const { return !dir_in_.empty(); }   // dir_in given ⇒ resume run

  // Load U/V/W/T (+ optional P for backward compat) from dir_in, applying halo
  // exchange + BC to each, and return the stats.txt payload.
  RestartState load(core::CpuFields& fields, const boundary::BoundaryCondition& bc,
                    const parallel::mpi::Subdomain& sub, MPI_Comm comm) const;

  // Final restart write into dir_out (no-op when dir_out is empty).
  void save(const core::CpuFields& fields, const parallel::mpi::Subdomain& sub,
            const RestartState& state, MPI_Comm comm) const;

  // True when a periodic checkpoint is due at this step (dir_out set,
  // nrestart_write>0, step a multiple). The caller gates the expensive
  // device→host copy on this and skips the final step itself.
  bool checkpoint_due(int step) const {
    return !dir_out_.empty() && nrestart_write_ > 0 && step % nrestart_write_ == 0;
  }

  // Periodic checkpoint into alternating dir_out/ckpt_a | ckpt_b so a
  // half-written checkpoint never clobbers the last good one. No-op if not due.
  void checkpoint_if_due(int step, double time, const core::CpuFields& fields,
                         const parallel::mpi::Subdomain& sub,
                         const RestartState& state, MPI_Comm comm) const;

private:
  void write_dir_(const std::string& dir, const core::CpuFields& fields,
                  const parallel::mpi::Subdomain& sub, const RestartState& state,
                  MPI_Comm comm) const;

  std::string dir_in_, dir_out_;
  int nrestart_write_ = 0;
};

} // namespace mpmstd::post

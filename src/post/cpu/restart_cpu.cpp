#include "post/restart.hpp"
#include "post/io.hpp"             // read_restart_cpu / write_restart_gathered_cpu
#include "core/boundary_ops.hpp"   // core::sync_field_cpu

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>

namespace mpmstd::post {

using core::Var;

RestartState Restart::load(core::CpuFields& fields, const boundary::BoundaryCondition& bc,
                           const parallel::mpi::Subdomain& sub, MPI_Comm comm) const {
  const std::string& dir = dir_in_;
  read_restart_cpu(fields[Var::U], sub, dir + "/U.bin");
  read_restart_cpu(fields[Var::V], sub, dir + "/V.bin");
  read_restart_cpu(fields[Var::W], sub, dir + "/W.bin");
  read_restart_cpu(fields[Var::T], sub, dir + "/T.bin");
  // P.bin (accumulated pressure, van Kan) — optional for backward compat with
  // pre-Report/32 restarts; absent ⇒ P stays zero (self-starting projection).
  // existence must be decided on ONE rank and broadcast: read_restart_cpu /
  // sync_field_cpu below are collective, so a per-rank filesystem::exists() that
  // disagrees across ranks (NFS attr-cache skew) makes some ranks enter the
  // collective while others skip it — deadlock. See Report/34 "restart read hang".
  int has_p_bin = 0;
  {
    int rank; MPI_Comm_rank(comm, &rank);
    if (rank == 0) has_p_bin = std::filesystem::exists(dir + "/P.bin") ? 1 : 0;
    MPI_Bcast(&has_p_bin, 1, MPI_INT, 0, comm);
  }
  if (has_p_bin) {
    read_restart_cpu(fields[Var::P], sub, dir + "/P.bin");
    core::sync_field_cpu(fields, Var::P, bc, sub);
  }
  core::sync_field_cpu(fields, Var::U, bc, sub);
  core::sync_field_cpu(fields, Var::V, bc, sub);
  core::sync_field_cpu(fields, Var::W, bc, sub);
  core::sync_field_cpu(fields, Var::T, bc, sub);

  // stats.txt: root reads the whole file, broadcasts the text, and every rank
  // parses the same content (step/time are structural fields; every other key
  // → extras in file order). One broadcast, no per-scalar collective mismatch,
  // and the running sums land identically on all ranks (they must — each rank
  // adds the same globally-reduced increment every step).
  RestartState state;
  int rank; MPI_Comm_rank(comm, &rank);
  std::string content;
  if (rank == 0) {
    std::ifstream sf(dir + "/stats.txt");
    std::stringstream ss; ss << sf.rdbuf(); content = ss.str();
  }
  int len = static_cast<int>(content.size());
  MPI_Bcast(&len, 1, MPI_INT, 0, comm);
  content.resize(len);
  if (len > 0) MPI_Bcast(&content[0], len, MPI_CHAR, 0, comm);
  {
    std::istringstream is(content);
    std::string key; double val;
    while (is >> key >> val) {
      if      (key == "step") state.step = static_cast<int>(val);
      else if (key == "time") state.time = val;
      else state.extras.emplace_back(key, val);
    }
  }
  if (rank == 0)
    std::printf("[restart] loaded step=%d t=%.3g (%zu stats)\n",
                state.step, state.time, state.extras.size());
  return state;
}

void Restart::write_dir_(const std::string& dir, const core::CpuFields& fields,
                         const parallel::mpi::Subdomain& sub, const RestartState& state,
                         MPI_Comm comm) const {
  std::error_code ec; std::filesystem::create_directories(dir, ec);
  MPI_Barrier(comm);
  // gathered (root-gather + fwrite), NOT the MPI-IO collective write_restart_cpu:
  // the matching MPI-IO read deadlocked on this cluster's NFS at np=2/256^3
  // (Report/34 addendum) — DHVC's restart write uses the same gathered path.
  write_restart_gathered_cpu(fields[Var::U], sub, dir + "/U.bin");
  write_restart_gathered_cpu(fields[Var::V], sub, dir + "/V.bin");
  write_restart_gathered_cpu(fields[Var::W], sub, dir + "/W.bin");
  write_restart_gathered_cpu(fields[Var::T], sub, dir + "/T.bin");
  write_restart_gathered_cpu(fields[Var::P], sub, dir + "/P.bin");  // accumulated pressure (van Kan)
  int rank; MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    std::ofstream sf(dir + "/stats.txt");
    sf << std::setprecision(17)
       << "step " << state.step << "\n"
       << "time " << state.time << "\n";
    for (const auto& e : state.extras) sf << e.first << " " << e.second << "\n";
    std::printf("[restart] wrote to %s (step=%d)\n", dir.c_str(), state.step);
  }
}

void Restart::save(const core::CpuFields& fields, const parallel::mpi::Subdomain& sub,
                   const RestartState& state, MPI_Comm comm) const {
  if (dir_out_.empty()) return;
  write_dir_(dir_out_, fields, sub, state, comm);
}

void Restart::checkpoint_if_due(int step, double time, const core::CpuFields& fields,
                                const parallel::mpi::Subdomain& sub,
                                const RestartState& state, MPI_Comm comm) const {
  (void)time;
  if (!checkpoint_due(step)) return;
  const bool odd = ((step / nrestart_write_) % 2) != 0;
  write_dir_(dir_out_ + (odd ? "/ckpt_b" : "/ckpt_a"), fields, sub, state, comm);
}

} // namespace mpmstd::post

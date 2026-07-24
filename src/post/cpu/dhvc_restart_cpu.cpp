#include "post/dhvc_restart.hpp"
#include "post/io.hpp"   // SrcMesh, make_src_mesh, compute_src_z_box,
                         // read_source_slab, read_global_field,
                         // interp_global_to_field, write_restart_gathered_cpu

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace mpmstd::post {

using core::Var;

// Warm start — read the source restart (any mesh) + interpolate (staggered,
// trilinear, x/y periodic, z clamp) onto this run's grid. U,V,W,T,P(=P_acc).
// Body moved verbatim from apps/dhvc/main_gpu.cu.
bool DhvcRestart::load_warmstart(core::CpuFields& cpu_fields, const grid::Grid& grid,
                                 const parallel::mpi::Subdomain& sub,
                                 const DhvcWarmStartConfig& cfg, MPI_Comm comm) const {
  int rank; MPI_Comm_rank(comm, &rank);
  const bool is_root = (rank == 0);

  int sdim[3] = {0, 0, 0};
  if (is_root) {
    std::ifstream mf(dir_in_ + "/meta.txt"); std::string k; double v;
    while (mf >> k >> v) { if (k=="n1m") sdim[0]=(int)v; else if (k=="n2m") sdim[1]=(int)v; else if (k=="n3m") sdim[2]=(int)v; }
  }
  MPI_Bcast(sdim, 3, MPI_INT, 0, comm);
  if (sdim[0]<=0 || sdim[1]<=0 || sdim[2]<=0)
    throw std::runtime_error("warm start: bad/missing meta.txt dims in " + dir_in_);
  SrcMesh sm = make_src_mesh(sdim[0], sdim[1], sdim[2], cfg.Lx, cfg.Ly, cfg.Lz, cfg.gamma3);
  std::vector<real_t> buf;
  const Var   vars[5]  = {Var::U, Var::V, Var::W, Var::T, Var::P};
  const char* names[5] = {"U", "V", "W", "T", "P"};
  const int   stag[5]  = {0, 1, 2, -1, -1};   // U:x-face V:y-face W:z-face T,P:center

  // I/O path: "slab" (Tier 1, default) reads only this rank's source z-slab via
  // collective MPI-IO (per-rank memory ~ full_field/N_z, no Bcast); "replicate"
  // (legacy) reads the whole field on every rank + Bcast. Both feed the IDENTICAL
  // trilinear interpolation — only the source z indexing differs.
  const std::string& ws_io = cfg.io;
  const bool use_slab = (ws_io != "replicate");
  int kz0 = 0, slab_n3 = -1;
  if (use_slab) {
    const int n3t = static_cast<int>(sub.n_total()[2]), h = kHaloWidth;
    const auto& tzf = grid.xf(Direction::Z);
    const double z_min = static_cast<double>(tzf[h]);
    const double z_max = static_cast<double>(tzf[n3t - h]);
    compute_src_z_box(sm, z_min, z_max, /*margin=*/2, kz0, slab_n3);
  }
  for (int q = 0; q < 5; ++q) {
    const std::string fpath = dir_in_ + "/" + std::string(names[q]) + ".bin";
    if (use_slab) read_source_slab(buf, sdim[0],sdim[1],sdim[2], kz0, slab_n3, fpath, comm);
    else          read_global_field(buf, sdim[0],sdim[1],sdim[2], fpath, comm);
    interp_global_to_field(cpu_fields[vars[q]], grid, buf, sm, stag[q], kz0, slab_n3);
  }
  if (is_root)
    std::printf("[warm-start] interpolated %dx%dx%d -> %dx%dx%d from %s (io=%s)\n",
                sdim[0],sdim[1],sdim[2],
                cfg.target_dims[0],cfg.target_dims[1],cfg.target_dims[2],
                dir_in_.c_str(), use_slab ? "slab" : "replicate");

  // dry-run: dump per-var interior checksums (for slab-vs-replicate A/B), then exit.
  if (cfg.dryrun) {
    const auto nt=sub.n_total(); const int n1=nt[0],n2=nt[1],n3=nt[2],h=kHaloWidth;
    if (is_root) std::printf("[warm-start dryrun] var: sum  sumsq  min  max  (global interior, io=%s)\n", use_slab?"slab":"replicate");
    for (int q=0; q<5; ++q) {
      const real_t* d = cpu_fields[vars[q]].data();
      double s=0, s2=0, mn=1e300, mx=-1e300;
      for (int i=h;i<n1-h;++i) for (int j=h;j<n2-h;++j) for (int k=h;k<n3-h;++k) {
        const double v=(double)d[((std::size_t)i*n2+j)*n3+k];
        s+=v; s2+=v*v; mn=std::min(mn,v); mx=std::max(mx,v);
      }
      double gs,gs2,gmn,gmx;
      MPI_Reduce(&s,&gs,1,MPI_DOUBLE,MPI_SUM,0,comm);
      MPI_Reduce(&s2,&gs2,1,MPI_DOUBLE,MPI_SUM,0,comm);
      MPI_Reduce(&mn,&gmn,1,MPI_DOUBLE,MPI_MIN,0,comm);
      MPI_Reduce(&mx,&gmx,1,MPI_DOUBLE,MPI_MAX,0,comm);
      if (is_root) std::printf("[warm-start dryrun] %s: %.15e %.15e %.15e %.15e\n", names[q],gs,gs2,gmn,gmx);
    }
    MPI_Barrier(comm);
    return true;   // dryrun requested → caller should exit
  }
  return false;
}

// Run-end checkpoint. U,V,W,T + P (= accumulated pressure P_acc for the
// incremental scheme). Gathered C-order write (decomposition-independent).
// meta.txt carries step/time/mesh/Ra for continuation + cross-mesh warm-start.
void DhvcRestart::save(const core::CpuFields& cpu_fields, const parallel::mpi::Subdomain& sub,
                       int step, double time, std::array<int,3> n_global, double Ra,
                       MPI_Comm comm) const {
  if (dir_out_.empty()) return;
  std::error_code ec; std::filesystem::create_directories(dir_out_, ec);
  MPI_Barrier(comm);
  write_restart_gathered_cpu(cpu_fields[Var::U], sub, dir_out_ + "/U.bin");
  write_restart_gathered_cpu(cpu_fields[Var::V], sub, dir_out_ + "/V.bin");
  write_restart_gathered_cpu(cpu_fields[Var::W], sub, dir_out_ + "/W.bin");
  write_restart_gathered_cpu(cpu_fields[Var::T], sub, dir_out_ + "/T.bin");
  write_restart_gathered_cpu(cpu_fields[Var::P], sub, dir_out_ + "/P.bin");
  int rank; MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    std::ofstream mf(dir_out_ + "/meta.txt");
    mf << std::setprecision(17)
       << "step " << step << "\n" << "time " << time << "\n"
       << "n1m " << n_global[0] << "\n" << "n2m " << n_global[1] << "\n"
       << "n3m " << n_global[2] << "\n" << "Ra " << Ra << "\n";
    std::printf("[restart] wrote %s  (step=%d t=%.4f; U/V/W/T/P + meta)\n",
                dir_out_.c_str(), step, time);
  }
}

} // namespace mpmstd::post

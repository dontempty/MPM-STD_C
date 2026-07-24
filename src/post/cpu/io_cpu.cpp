#include "post/io.hpp"
#include "common/macros.hpp"   // kHaloWidth

#include <mpi.h>
#include <stdexcept>
#include <cstdio>
#include <vector>
#include <fstream>

// P1 — parallel binary restart IO (faithful port of post::write_scalar/
// read_scalar). File = global interior, row-major (axis 0 slowest), raw real_t,
// no header. Reads/writes the SAME format the old channel produced, so a frozen
// turbulent restart loads directly. CpuField is dumb data → Subdomain passed in.

namespace mpmstd::post {

namespace {

MPI_Datatype make_global_view(const parallel::mpi::Subdomain& sub) {
  const auto n_glb = sub.n_global();
  const auto n_int = sub.n_interior();
  const auto off   = sub.global_offset();
  int sizes[3]    = {n_glb[0], n_glb[1], n_glb[2]};
  int subsizes[3] = {n_int[0], n_int[1], n_int[2]};
  int starts[3]   = {off[0], off[1], off[2]};
  MPI_Datatype dt = MPI_DATATYPE_NULL;
  MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                           sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT, &dt);
  MPI_Type_commit(&dt);
  return dt;
}

MPI_Datatype make_local_view(const parallel::mpi::Subdomain& sub) {
  const auto n_tot = sub.n_total();
  const auto n_int = sub.n_interior();
  int sizes[3]    = {n_tot[0], n_tot[1], n_tot[2]};
  int subsizes[3] = {n_int[0], n_int[1], n_int[2]};
  int starts[3]   = {kHaloWidth, kHaloWidth, kHaloWidth};
  MPI_Datatype dt = MPI_DATATYPE_NULL;
  MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                           sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT, &dt);
  MPI_Type_commit(&dt);
  return dt;
}

} // anonymous namespace

void write_restart_cpu(const core::CpuField& f, const parallel::mpi::Subdomain& sub, const std::string& path) {
  MPI_Comm comm = sub.topology().cart_comm();
  MPI_Datatype filetype = make_global_view(sub);
  MPI_Datatype memtype  = make_local_view(sub);
  MPI_File fh = MPI_FILE_NULL;
  if (MPI_File_open(comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
    MPI_Type_free(&filetype); MPI_Type_free(&memtype);
    throw std::runtime_error("write_restart_cpu: cannot open '" + path + "'");
  }
  MPI_File_set_size(fh, 0);
  MPI_File_set_view(fh, 0, sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT,
                    filetype, "native", MPI_INFO_NULL);
  MPI_File_write_all(fh, const_cast<real_t*>(f.data()), 1, memtype, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);
  MPI_Type_free(&filetype); MPI_Type_free(&memtype);
}

// Restart reads use plain POSIX I/O (root fread + Bcast), NOT MPI-IO: collective
// AND independent MPI_File_read[_all] with a 3D subarray view both deadlock on
// this cluster's NFS mount at larger sizes (confirmed: 256^3/np=2 hangs inside
// the read call itself, after open/set_view both succeed — a ROMIO/NFS locking
// issue, not an application bug). Root reads the whole global field and
// broadcasts it; every rank then copies out its own interior block locally.
// Memory cost = one full global field per rank (same tradeoff as the existing
// "replicate" warm-start path) — acceptable at these problem sizes.
void read_restart_cpu(core::CpuField& f, const parallel::mpi::Subdomain& sub, const std::string& path) {
  MPI_Comm comm = sub.topology().cart_comm();
  int rank = 0; MPI_Comm_rank(comm, &rank);
  const auto n_glb = sub.n_global();
  const auto n_int = sub.n_interior();
  const auto n_tot = sub.n_total();
  const auto off   = sub.global_offset();
  const MPI_Datatype MT = sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;
  const long n_global_elems = (long)n_glb[0] * n_glb[1] * n_glb[2];

  std::vector<real_t> global_buf(n_global_elems);
  if (rank == 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("read_restart_cpu: cannot open '" + path + "'");
    in.read(reinterpret_cast<char*>(global_buf.data()),
            static_cast<std::streamsize>(n_global_elems * sizeof(real_t)));
    if (!in) throw std::runtime_error("read_restart_cpu: short read '" + path + "'");
  }
  MPI_Bcast(global_buf.data(), n_global_elems, MT, 0, comm);

  const int h = kHaloWidth;
  const int n2t = n_tot[1], n3t = n_tot[2];
  real_t* dst = f.data();
  for (int i = 0; i < n_int[0]; ++i)
    for (int j = 0; j < n_int[1]; ++j)
      for (int k = 0; k < n_int[2]; ++k) {
        const long gi = off[0] + i, gj = off[1] + j, gk = off[2] + k;
        const long gidx = (gi * n_glb[1] + gj) * n_glb[2] + gk;
        const long lidx = ((long)(i + h) * n2t + (j + h)) * n3t + (k + h);
        dst[lidx] = global_buf[gidx];
      }
}

void write_restart_gathered_cpu(const core::CpuField& f, const parallel::mpi::Subdomain& sub, const std::string& path) {
  MPI_Comm comm = sub.topology().cart_comm();
  int rank = 0, nproc = 1;
  MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &nproc);
  const auto ng = sub.n_global();
  const auto ni = sub.n_interior();
  const auto off = sub.global_offset();
  const auto nt = sub.n_total();
  const int  h  = kHaloWidth;
  const MPI_Datatype MT = (sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT;

  // pack this rank's interior (skip halo) into a contiguous C-order buffer
  const std::size_t nloc = (std::size_t)ni[0] * ni[1] * ni[2];
  std::vector<real_t> sendbuf(nloc);
  {
    const int n2t = nt[1], n3t = nt[2];
    const real_t* d = f.data();
    std::size_t p = 0;
    for (int i = 0; i < ni[0]; ++i)
      for (int j = 0; j < ni[1]; ++j)
        for (int k = 0; k < ni[2]; ++k)
          sendbuf[p++] = d[(((std::size_t)(i + h) * n2t) + (j + h)) * n3t + (k + h)];
  }
  // gather per-rank (offset[3], n_interior[3]) → root
  int meta[6] = {off[0], off[1], off[2], ni[0], ni[1], ni[2]};
  std::vector<int> allmeta(rank == 0 ? 6 * nproc : 0);
  MPI_Gather(meta, 6, MPI_INT, allmeta.data(), 6, MPI_INT, 0, comm);
  // gatherv the packed interiors → root
  std::vector<int> counts, displs;
  std::vector<real_t> gathered;
  long total = 0;
  if (rank == 0) {
    counts.resize(nproc); displs.resize(nproc);
    for (int r = 0; r < nproc; ++r) {
      counts[r] = allmeta[6*r+3] * allmeta[6*r+4] * allmeta[6*r+5];
      displs[r] = (int)total; total += counts[r];
    }
    gathered.resize((std::size_t)total);
  }
  MPI_Gatherv(sendbuf.data(), (int)nloc, MT,
              gathered.data(), counts.data(), displs.data(), MT, 0, comm);
  // root: scatter each rank's block into the global C-order array, then one fwrite
  if (rank == 0) {
    std::vector<real_t> glob((std::size_t)ng[0] * ng[1] * ng[2]);
    for (int r = 0; r < nproc; ++r) {
      const int o0=allmeta[6*r], o1=allmeta[6*r+1], o2=allmeta[6*r+2];
      const int m0=allmeta[6*r+3], m1=allmeta[6*r+4], m2=allmeta[6*r+5];
      const real_t* src = &gathered[(std::size_t)displs[r]];
      std::size_t p = 0;
      for (int i = 0; i < m0; ++i)
        for (int j = 0; j < m1; ++j)
          for (int k = 0; k < m2; ++k)
            glob[(((std::size_t)(o0 + i) * ng[1]) + (o1 + j)) * ng[2] + (o2 + k)] = src[p++];
    }
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("write_restart_gathered_cpu: cannot open '" + path + "'");
    std::fwrite(glob.data(), sizeof(real_t), glob.size(), fp);
    std::fclose(fp);
  }
  MPI_Barrier(comm);
}

} // namespace mpmstd::post

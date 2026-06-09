#include "post/io.hpp"
#include "common/macros.hpp"   // kHaloWidth

#include <mpi.h>
#include <stdexcept>

// P1 — parallel binary restart IO (faithful port of post::write_scalar/
// read_scalar). File = global interior, row-major (axis 0 slowest), raw real_t,
// no header. Reads/writes the SAME format the old channel produced, so a frozen
// turbulent restart loads directly. CpuField is dumb data → Subdomain passed in.

namespace mpmstd::post {

namespace {

MPI_Datatype make_global_view(const core::Subdomain& sub) {
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

MPI_Datatype make_local_view(const core::Subdomain& sub) {
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

void write_restart_cpu(const core::CpuField& f, const core::Subdomain& sub, const std::string& path) {
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

void read_restart_cpu(core::CpuField& f, const core::Subdomain& sub, const std::string& path) {
  MPI_Comm comm = sub.topology().cart_comm();
  MPI_Datatype filetype = make_global_view(sub);
  MPI_Datatype memtype  = make_local_view(sub);
  MPI_File fh = MPI_FILE_NULL;
  if (MPI_File_open(comm, path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
    MPI_Type_free(&filetype); MPI_Type_free(&memtype);
    throw std::runtime_error("read_restart_cpu: cannot open '" + path + "'");
  }
  MPI_File_set_view(fh, 0, sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT,
                    filetype, "native", MPI_INFO_NULL);
  MPI_File_read_all(fh, f.data(), 1, memtype, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);
  MPI_Type_free(&filetype); MPI_Type_free(&memtype);
}

} // namespace mpmstd::post

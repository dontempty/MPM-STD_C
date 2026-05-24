#include "post/restart_io.hpp"

#include "parallel/mpi/subdomain.hpp"

#include <stdexcept>
#include <vector>

namespace mpmstd::post {

namespace {

// Helper: build a derived datatype that describes this rank's interior slab
// inside the GLOBAL array (n1m_global, n2m_global, n3m_global).
// Returns (committed) datatype; caller owns and must MPI_Type_free.
MPI_Datatype make_global_view(const parallel::mpi::Subdomain& sub) {
  const auto n_glb = sub.n_global();
  const auto n_int = sub.n_interior();
  const auto off   = sub.global_offset();

  int sizes   [3] = { n_glb[0], n_glb[1], n_glb[2] };
  int subsizes[3] = { n_int[0], n_int[1], n_int[2] };
  int starts  [3] = { off[0],   off[1],   off[2]   };

  MPI_Datatype dt = MPI_DATATYPE_NULL;
  MPI_Type_create_subarray(3, sizes, subsizes, starts,
                            MPI_ORDER_C,
                            sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT,
                            &dt);
  MPI_Type_commit(&dt);
  return dt;
}

// Helper: build a derived datatype that describes this rank's interior slab
// inside the LOCAL halo'd array (n_total[d]). This is the "in-memory view"
// counterpart — together with the global-view above we can use MPI-IO without
// any intermediate copy.
MPI_Datatype make_local_view(const parallel::mpi::Subdomain& sub) {
  const auto n_tot = sub.n_total();
  const auto n_int = sub.n_interior();

  int sizes   [3] = { n_tot[0], n_tot[1], n_tot[2] };
  int subsizes[3] = { n_int[0], n_int[1], n_int[2] };
  // Interior starts at +kHaloWidth on every axis.
  int starts  [3] = { kHaloWidth, kHaloWidth, kHaloWidth };

  MPI_Datatype dt = MPI_DATATYPE_NULL;
  MPI_Type_create_subarray(3, sizes, subsizes, starts,
                            MPI_ORDER_C,
                            sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT,
                            &dt);
  MPI_Type_commit(&dt);
  return dt;
}

} // namespace

void write_scalar(const field::ScalarField& field, const std::string& path) {
  // Use the host buffer (caller is responsible for to_host() if device-only).
  const auto& sub = field.subdomain();
  MPI_Comm comm = sub.topology().cart_comm();

  MPI_Datatype filetype = make_global_view(sub);
  MPI_Datatype memtype  = make_local_view (sub);

  MPI_File fh = MPI_FILE_NULL;
  int rc = MPI_File_open(comm, path.c_str(),
                          MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  if (rc != MPI_SUCCESS) {
    MPI_Type_free(&filetype);
    MPI_Type_free(&memtype);
    throw std::runtime_error("write_scalar: cannot open '" + path + "' for write");
  }

  // Truncate (in case the file existed and was longer than the new payload).
  MPI_File_set_size(fh, 0);

  MPI_File_set_view(fh, /*disp=*/0,
                     sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT,
                     filetype, "native", MPI_INFO_NULL);

  // const_cast is safe: MPI_File_write_all does not modify the buffer.
  MPI_File_write_all(fh, const_cast<real_t*>(field.host_ptr()),
                      1, memtype, MPI_STATUS_IGNORE);

  MPI_File_close(&fh);
  MPI_Type_free(&filetype);
  MPI_Type_free(&memtype);
}

void read_scalar(field::ScalarField& field, const std::string& path) {
  const auto& sub = field.subdomain();
  MPI_Comm comm = sub.topology().cart_comm();

  MPI_Datatype filetype = make_global_view(sub);
  MPI_Datatype memtype  = make_local_view (sub);

  MPI_File fh = MPI_FILE_NULL;
  int rc = MPI_File_open(comm, path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if (rc != MPI_SUCCESS) {
    MPI_Type_free(&filetype);
    MPI_Type_free(&memtype);
    throw std::runtime_error("read_scalar: cannot open '" + path + "'");
  }

  MPI_File_set_view(fh, 0,
                     sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT,
                     filetype, "native", MPI_INFO_NULL);

  MPI_File_read_all(fh, field.host_ptr(), 1, memtype, MPI_STATUS_IGNORE);

  MPI_File_close(&fh);
  MPI_Type_free(&filetype);
  MPI_Type_free(&memtype);
}

} // namespace mpmstd::post

#include "post/io.hpp"
#include "grid/stretching.hpp"     // make_face_coordinates
#include "common/macros.hpp"       // kHaloWidth
#include "common/direction.hpp"    // Direction

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Cross-mesh warm start: read a whole global restart field at the SOURCE resolution
// and interpolate (staggered-aware, trilinear) onto the current run's interior.
// x,y periodic; z wall (clamp). Same domain + stretching law as the source assumed.

namespace mpmstd::post {

void read_global_field(std::vector<real_t>& buf, int n1, int n2, int n3,
                       const std::string& path, MPI_Comm comm) {
  const std::size_t n = (std::size_t)n1 * n2 * n3;
  buf.assign(n, real_t{0});
  int rank; MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("read_global_field: cannot open '" + path + "'");
    const std::size_t got = std::fread(buf.data(), sizeof(real_t), n, fp);
    std::fclose(fp);
    if (got != n) throw std::runtime_error("read_global_field: short read '" + path + "'");
  }
  MPI_Bcast(buf.data(), (int)n,
            sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT, 0, comm);
}

SrcMesh make_src_mesh(int n1, int n2, int n3, double Lx, double Ly, double Lz, double gamma_z) {
  SrcMesh sm; sm.n[0]=n1; sm.n[1]=n2; sm.n[2]=n3; sm.Lx=Lx; sm.Ly=Ly;
  const int    nd[3] = {n1, n2, n3};
  const double Ld[3] = {Lx, Ly, Lz};
  for (int d = 0; d < 3; ++d) {
    const grid::StretchKind kind = (d == 2 && gamma_z > 0.0)
                                 ? grid::StretchKind::Tanh : grid::StretchKind::Uniform;
    const real_t gam = (d == 2) ? static_cast<real_t>(gamma_z) : real_t{0};
    const std::vector<real_t> faces =
        grid::make_face_coordinates(kind, nd[d], static_cast<real_t>(Ld[d]), gam); // len nd[d]+1
    sm.xf[d].resize(nd[d]);
    sm.xc[d].resize(nd[d]);
    for (int i = 0; i < nd[d]; ++i) {
      sm.xf[d][i] = faces[i];                                  // staggered (velocity) location
      sm.xc[d][i] = real_t{0.5} * (faces[i] + faces[i + 1]);   // cell center
    }
  }
  return sm;
}

void compute_src_z_box(const SrcMesh& sm, double z_min, double z_max, int margin,
                       int& kz0, int& slab_n3) {
  const int n3 = sm.n[2];
  int lo = n3 - 1, hi = 0;
  // For a monotonically-increasing node array, bracket [z_min, z_max]:
  //   lo = largest index with node <= z_min  (clamp 0 when z_min below first node)
  //   hi = smallest index with node >= z_max (clamp n3-1 when z_max above last node)
  // locate() near z_min/z_max may return {lo, lo+1} / {hi-1, hi}; margin (>=1)
  // guarantees both bracket members are inside the slab. Union over centers + faces
  // covers every staggered variable (T,P,U,V use centers along z; W uses faces).
  auto bracket = [&](const std::vector<real_t>& a) {
    int i_lo = 0;
    for (int i = 0; i < n3; ++i) { if ((double)a[i] <= z_min) i_lo = i; else break; }
    int i_hi = n3 - 1;
    for (int i = n3 - 1; i >= 0; --i) { if ((double)a[i] >= z_max) i_hi = i; else break; }
    lo = std::min(lo, i_lo);
    hi = std::max(hi, i_hi);
  };
  bracket(sm.xc[2]);
  bracket(sm.xf[2]);
  kz0 = std::max(0, lo - margin);
  const int kz1 = std::min(n3 - 1, hi + margin);
  slab_n3 = kz1 - kz0 + 1;
}

void read_source_slab(std::vector<real_t>& buf, int n1, int n2, int n3,
                      int kz0, int slab_n3, const std::string& path, MPI_Comm comm) {
  const std::size_t n = (std::size_t)n1 * n2 * slab_n3;
  buf.assign(n, real_t{0});
  const MPI_Datatype etype = (sizeof(real_t) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT;

  // ROMIO collective-buffering hints. Without these the strided z-subarray read
  // degrades into per-element independent I/O — the same trap that made the old
  // MPI-IO restart write run at ~1.8 MB/min. Collective aggregation fixes it.
  MPI_Info info; MPI_Info_create(&info);
  MPI_Info_set(info, "romio_cb_read",  "enable");
  MPI_Info_set(info, "cb_buffer_size", "16777216");

  MPI_File fh;
  if (MPI_File_open(comm, path.c_str(), MPI_MODE_RDONLY, info, &fh) != MPI_SUCCESS) {
    MPI_Info_free(&info);
    throw std::runtime_error("read_source_slab: cannot open '" + path + "'");
  }

  // This rank's block: full (x,y), z-slab [kz0, kz0+slab_n3) of the global (n1,n2,n3)
  // C-order array. Each rank sets its OWN view (different kz0/slab_n3 are fine);
  // adjacent ranks' margin overlap is harmless for reads.
  const int gsizes[3]   = {n1, n2, n3};
  const int subsizes[3] = {n1, n2, slab_n3};
  const int starts[3]   = {0,  0,  kz0};
  MPI_Datatype filetype;
  MPI_Type_create_subarray(3, gsizes, subsizes, starts, MPI_ORDER_C, etype, &filetype);
  MPI_Type_commit(&filetype);

  MPI_File_set_view(fh, 0, etype, filetype, const_cast<char*>("native"), info);
  MPI_Status st;
  MPI_File_read_all(fh, buf.data(), (int)n, etype, &st);

  MPI_File_close(&fh);
  MPI_Type_free(&filetype);
  MPI_Info_free(&info);
}

namespace {
// Locate ct in monotonically-increasing cs[0..ns-1]; returns (i0,i1,w) with
// value = (1-w)*v[i0] + w*v[i1]. periodic → wrap with period L>0; else clamp.
inline void locate(const real_t* cs, int ns, double ct, double L, bool periodic,
                   int& i0, int& i1, double& w) {
  if (periodic) {
    const double lo = cs[0];
    double x = ct - lo;
    x -= std::floor(x / L) * L;                 // x in [0, L)
    const double cx = lo + x;
    if (cx >= cs[ns - 1]) {                     // wrap segment [cs[ns-1], cs[0]+L)
      i0 = ns - 1; i1 = 0;
      const double seg = (cs[0] + L) - cs[ns - 1];
      w = seg > 0.0 ? (cx - cs[ns - 1]) / seg : 0.0;
      return;
    }
    int a = 0, b = ns - 1;
    while (b - a > 1) { const int m = (a + b) / 2; if (cs[m] <= cx) a = m; else b = m; }
    i0 = a; i1 = a + 1; w = (cx - cs[a]) / (cs[a + 1] - cs[a]);
  } else {
    if (ct <= cs[0])      { i0 = 0;      i1 = 0;      w = 0.0; return; }
    if (ct >= cs[ns - 1]) { i0 = ns - 1; i1 = ns - 1; w = 0.0; return; }
    int a = 0, b = ns - 1;
    while (b - a > 1) { const int m = (a + b) / 2; if (cs[m] <= ct) a = m; else b = m; }
    i0 = a; i1 = a + 1; w = (ct - cs[a]) / (cs[a + 1] - cs[a]);
  }
}
} // anonymous namespace

void interp_global_to_field(core::CpuField& tgt, const core::Grid& tgt_grid,
                            const std::vector<real_t>& src, const SrcMesh& sm, int stagger_axis,
                            int kz0, int slab_n3) {
  const int h = kHaloWidth;
  const auto nt = tgt.n_total();
  const int n1t = nt[0], n2t = nt[1], n3t = nt[2];
  const int sn1 = sm.n[0], sn2 = sm.n[1], sn3 = sm.n[2];
  // z-slab layout: replicate path (slab_n3<0) ⇒ whole field, kz0=0, stride=sn3.
  // slab path ⇒ src holds only z∈[kz0, kz0+slab_n3); global k maps to (k-kz0).
  const int skn = (slab_n3 < 0) ? sn3 : slab_n3;

  // source interior coords (face along the staggered axis, else center)
  const real_t* sx = (stagger_axis == 0 ? sm.xf[0].data() : sm.xc[0].data());
  const real_t* sy = (stagger_axis == 1 ? sm.xf[1].data() : sm.xc[1].data());
  const real_t* sz = (stagger_axis == 2 ? sm.xf[2].data() : sm.xc[2].data());
  // target coords (this rank, local index → coordinate)
  const real_t* tx = (stagger_axis == 0 ? tgt_grid.xf(Direction::X).data() : tgt_grid.xc(Direction::X).data());
  const real_t* ty = (stagger_axis == 1 ? tgt_grid.xf(Direction::Y).data() : tgt_grid.xc(Direction::Y).data());
  const real_t* tz = (stagger_axis == 2 ? tgt_grid.xf(Direction::Z).data() : tgt_grid.xc(Direction::Z).data());

  real_t* out = tgt.data();
  auto S = [&](int i, int j, int k) -> double {
    return static_cast<double>(src[((std::size_t)i * sn2 + j) * skn + (k - kz0)]);
  };

  for (int li = h; li < n1t - h; ++li) {
    int ix0, ix1; double wx; locate(sx, sn1, (double)tx[li], sm.Lx, true,  ix0, ix1, wx);
    for (int lj = h; lj < n2t - h; ++lj) {
      int iy0, iy1; double wy; locate(sy, sn2, (double)ty[lj], sm.Ly, true,  iy0, iy1, wy);
      for (int lk = h; lk < n3t - h; ++lk) {
        int iz0, iz1; double wz; locate(sz, sn3, (double)tz[lk], 0.0,   false, iz0, iz1, wz);
        const double c00 = (1 - wx) * S(ix0, iy0, iz0) + wx * S(ix1, iy0, iz0);
        const double c01 = (1 - wx) * S(ix0, iy0, iz1) + wx * S(ix1, iy0, iz1);
        const double c10 = (1 - wx) * S(ix0, iy1, iz0) + wx * S(ix1, iy1, iz0);
        const double c11 = (1 - wx) * S(ix0, iy1, iz1) + wx * S(ix1, iy1, iz1);
        const double c0 = (1 - wy) * c00 + wy * c10;
        const double c1 = (1 - wy) * c01 + wy * c11;
        out[((std::size_t)li * n2t + lj) * n3t + lk] = static_cast<real_t>((1 - wz) * c0 + wz * c1);
      }
    }
  }
}

} // namespace mpmstd::post

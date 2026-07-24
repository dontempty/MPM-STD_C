#pragma once

#include "core/field_cpu.hpp"
#include "core/gpu_binding.hpp"
#include "core/grid.hpp"          // grid::Grid (alias of grid::Grid) — xc/xf accessors

#include <string>
#include <vector>

namespace mpmstd::post {

// Restart / field IO (rev.2 §10, U6: mandatory; target = existing global C-order
// double format). Host-side even in GPU runs (to_host() first) ⇒ CPU only, no
// _gpu twin. (body P1.)
void write_restart_cpu(const core::CpuField& f, const parallel::mpi::Subdomain& sub, const std::string& path);
void read_restart_cpu (core::CpuField& f,       const parallel::mpi::Subdomain& sub, const std::string& path);

// Faster restart write for strided decompositions (z-split → the MPI-IO subarray
// pattern is pathologically slow): gather the global interior to root (MPI_Gatherv)
// and write it with one contiguous fwrite. Produces the IDENTICAL global C-order
// double file that write_restart_cpu / read_global_field expect.
void write_restart_gathered_cpu(const core::CpuField& f, const parallel::mpi::Subdomain& sub, const std::string& path);

// ── cross-mesh warm start ───────────────────────────────────────────────────────
// Read the WHOLE global interior field (.bin = n1*n2*n3 raw real_t, C-order, the
// same format write_restart_cpu produces) into `buf` on every rank (root reads +
// Bcast). Used when the source mesh differs from the current run's mesh.
// "replicate" warm-start path — per-rank memory = full source field (does NOT
// scale with rank count). Prefer read_source_slab for large problems.
void read_global_field(std::vector<real_t>& buf, int n1, int n2, int n3,
                       const std::string& path, MPI_Comm comm);

// Tier 1 distributed warm-start read: each rank reads ONLY the source z-slab
// [kz0, kz0+slab_n3) over the full x,y range, via collective MPI-IO
// (MPI_File_read_all + a 3D subarray view). Per-rank memory ~ full_field / N_z;
// no Bcast; parallel I/O. `buf` is sized n1*n2*slab_n3 (slab-local, C-order with
// the same (i*n2+j)*slab_n3 + (k-kz0) layout interp_global_to_field expects).
void read_source_slab(std::vector<real_t>& buf, int n1, int n2, int n3,
                      int kz0, int slab_n3, const std::string& path, MPI_Comm comm);

// Source mesh coordinates (global, no MPI decomposition), reconstructed from the
// source dims + the SAME stretching law via make_face_coordinates. Per axis we hold
// cell-center xc[d] (len n[d]) and the staggered face location xf[d] (len n[d],
// = the first n faces; the (n+1)-th face is the periodic image / domain end).
struct SrcMesh {
  int                 n[3]  = {0,0,0};
  double              Lx = 0.0, Ly = 0.0;   // periodic lengths (x,y); z is wall (clamp)
  std::vector<real_t> xc[3];
  std::vector<real_t> xf[3];
};

// Build a SrcMesh from source dims + per-axis stretching (uniform x,y; tanh z γ).
SrcMesh make_src_mesh(int n1, int n2, int n3, double Lx, double Ly, double Lz, double gamma_z);

// Compute the source z-index box [kz0, kz0+slab_n3) that brackets a target rank's
// physical z-extent [z_min, z_max] for trilinear interpolation. Considers BOTH the
// source cell-centers and faces, pads by `margin`, and clamps to [0, n3) (z is the
// wall = clamp axis, never periodic). Used to size read_source_slab.
void compute_src_z_box(const SrcMesh& sm, double z_min, double z_max, int margin,
                       int& kz0, int& slab_n3);

// Interpolate a source field onto this rank's interior of `tgt`, trilinear
// tensor-product. x,y periodic; z clamped (wall). stagger_axis selects the
// staggered location to evaluate at:
//   0 = x-face (U), 1 = y-face (V), 2 = z-face (W), -1 = cell center (P,T).
// `src` may be either the WHOLE global field (kz0=0, slab_n3<0 ⇒ full sm.n[2]) or
// a z-slab from read_source_slab (kz0,slab_n3 = that slab's z-origin/extent). The
// trilinear math is identical in both cases — only the source z indexing shifts.
void interp_global_to_field(core::CpuField& tgt, const grid::Grid& tgt_grid,
                            const std::vector<real_t>& src, const SrcMesh& sm, int stagger_axis,
                            int kz0 = 0, int slab_n3 = -1);

} // namespace mpmstd::post

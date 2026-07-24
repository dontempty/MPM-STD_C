#include "equation/scalar/kernels.hpp"
#include "equation/scalar/scalar_var_stencil.hpp"
#include "common/macros.hpp"

// =============================================================================
//  Variable-κ̂ (NOB) ENERGY kernels — FAITHFUL line-by-line port of
//  PaScaL_TCS module_solve_thermal.f90 (mpi_thermal_solver). See kernels.hpp
//  for the scheme overview. This is the reference's delta-form 3-factor ADI.
//
//  The per-cell math (face stencils, axis bands, explicit residual R, delta-
//  form RHS pack) lives in scalar_var_stencil.hpp (build_A_rhs_cell /
//  energy_rhs_cell, MPMSTD_HD — shared with the gpu/ backend); this file only
//  provides the CPU (sys,row) dispatch loops.
//
//  Reference axis x2 = wall-normal (velocity V) → my Z (velocity W); ref x3 → my
//  Y (velocity V); ref x1 → my X (velocity U). Wall: validated mpmstd wiring
//  (antisymmetric θ ghost + modify_tdma_row_cpu fold), δθ_wall=0.
// =============================================================================

namespace mpmstd::equation::scalar::kernels {

// Build the sweep-axis ADI bands (A,B,C) AND pack the delta-form RHS into D
// (fused build_A_RHS). Stage 1 computes the explicit residual R inline inside
// build_A_rhs_cell — `src` is read only on stages 2/3 (the transposed running
// increment δ from the previous sweep).
// z_owns_lower/upper gate the jem/jep dkPP masks to ranks that own the PHYSICAL
// wall — without the guard, np3>1 fired these masks at interior rank seams
// (mid-height for np3=2, exactly the θ_c plane): audit D5, Report/RBC.
void build_A_rhs_nob(Direction      d,
                     real_t*        A, real_t* B, real_t* C, real_t* D,
                     const real_t*  src,
                     const real_t*  T,
                     const real_t*  U, const real_t* V, const real_t* W,
                     const real_t*  KPP, const real_t* dKPP,
                     const real_t*  invRhoCp, const real_t* dinvRhoCp,
                     const real_t*  dx1, const real_t* dmx1,
                     const real_t*  dx2, const real_t* dmx2,
                     const real_t*  dx3, const real_t* dmx3,
                     int            n1_tot, int n2_tot, int n3_tot,
                     int            n1_int, int n2_int, int n3_int,
                     real_t         Ct, real_t dt,
                     bool           is_stage1,
                     bool           z_owns_lower, bool z_owns_upper) {
  auto build_cell = [&](int i, int j, int k, int p, bool wall_lo, bool wall_hi) {
    real_t am, ac, ap, Dv;
    build_A_rhs_cell(d, i, j, k, n1_tot, n2_tot, n3_tot,
                              src, T, U, V, W, KPP, dKPP, invRhoCp, dinvRhoCp,
                              dx1, dmx1, dx2, dmx2, dx3, dmx3,
                              Ct, dt, is_stage1, wall_lo, wall_hi, am, ac, ap, Dv);
    A[p] = am; B[p] = ac; C[p] = ap; D[p] = Dv;
  };

  // dispatch over the sweep axis (same (sys,row) packing as the const path)
  if (d == Direction::X) {
    const int n_sys = n2_int * n3_int;
    for (int ii = 0; ii < n1_int; ++ii) {
      const int i = ii + kHaloWidth;
      for (int jj = 0; jj < n2_int; ++jj)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int j = jj + kHaloWidth, k = kk + kHaloWidth;
          const int p = ii * n_sys + (jj * n3_int + kk);
          build_cell(i, j, k, p, z_owns_lower && kk==0, z_owns_upper && kk==n3_int-1);
        }
    }
  } else if (d == Direction::Y) {
    const int n_sys = n1_int * n3_int;
    for (int jj = 0; jj < n2_int; ++jj) {
      const int j = jj + kHaloWidth;
      for (int ii = 0; ii < n1_int; ++ii)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int i = ii + kHaloWidth, k = kk + kHaloWidth;
          const int p = jj * n_sys + (ii * n3_int + kk);
          build_cell(i, j, k, p, z_owns_lower && kk==0, z_owns_upper && kk==n3_int-1);
        }
    }
  } else { // Direction::Z  (wall-normal) — PaScaL Stage 3 with jem/jep dkPP mask
    const int n_sys = n1_int * n2_int;
    for (int kk = 0; kk < n3_int; ++kk) {
      const int k = kk + kHaloWidth;
      const bool wall_lo = z_owns_lower && (kk == 0);
      const bool wall_hi = z_owns_upper && (kk == n3_int - 1);
      for (int ii = 0; ii < n1_int; ++ii)
        for (int jj = 0; jj < n2_int; ++jj) {
          const int i = ii + kHaloWidth, j = jj + kHaloWidth;
          const int p = kk * n_sys + (ii * n2_int + jj);
          build_cell(i, j, k, p, wall_lo, wall_hi);
        }
    }
  }
}

} // namespace mpmstd::equation::scalar::kernels

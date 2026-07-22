#pragma once

#include "common/types.hpp"
#include "common/direction.hpp"

namespace mpmstd::equation::momentum::kernels {

// ── MPM-STD BW-ADI explicit RHS ──────────────────────────────────────────────
// rhs[c] = dt * nu * (Lx + Ly + Lz)(q)[c]
//        - conv_factor * dt * div(u*q)[c]
//
// Standard Laplacian only (no own-direction doubling).
// Backward-Euler for own direction is handled by visc_factor=1.0 in build_adi_bands.
// Convection uses 0.25 coefficient (semi-implicit: rest goes into ADI bands).
// Interior cells only; halos must be filled before this call.
// conv_factor controls the convection contribution: 0.25 normally, 0 to disable.
void compute_mpmstd_rhs(real_t*       rhs,
                         const real_t* q,
                         const real_t* U, const real_t* V, const real_t* W,
                         const real_t* dx1, const real_t* dmx1,
                         const real_t* dx2, const real_t* dmx2,
                         const real_t* dx3, const real_t* dmx3,
                         int n1_tot, int n2_tot, int n3_tot,
                         real_t viscosity, real_t dt,
                         Direction own_dir,
                         bool z_staggered = false,
                         real_t conv_factor = real_t{0.25});

// ── Variable-viscosity (NOB) FAITHFUL RHS ────────────────────────────────────
// Folds the ENTIRE reference stage-1 explicit residual (PaScaL_TCS
// module_solve_momentum.f90: solvedU 341-413 → own=X; solvedW → own=Y via ref
// x3; solvedV-minus-buoyancy → own=Z(wall) via ref x2) into one delta-form
// scalar D = dt·(R − M·qⁿ − cross-bookkeeping). Per component the OWN direction
// (own_dir) carries the ×2 own-velocity diffusion + own 0.25 convection; the two
// CROSS directions carry the (cancelled→implicit) own-velocity diffusion + the
// weight-2 cross-STRESS + the 0.25 advective cross-convection. PRESSURE +
// BUOYANCY are SKIPPED (composed in main). NO jum/jup wall masks — wall via
// zero-ghost + modify_tdma_row_cpu fold. Output already ×dt. Do NOT also call
// add_transpose_stress_* on this path (cross-stress is already weight-2 here).
// AXIS SWAP: ref x1(stream)→X, x2(WALL)→Z, x3(span)→Y. Cmu=√(Pr/Ra).
void compute_mpmstd_rhs_var(real_t*       rhs,
                            const real_t* q,
                            const real_t* U, const real_t* V, const real_t* W,
                            const real_t* mu, const real_t* invrho,
                            const real_t* dx1, const real_t* dmx1,
                            const real_t* dx2, const real_t* dmx2,
                            const real_t* dx3, const real_t* dmx3,
                            int n1_tot, int n2_tot, int n3_tot,
                            real_t Cmu, real_t dt,
                            Direction own_dir,
                            bool z_staggered = false);

// Variable-viscosity FAITHFUL ADI bands for sweep direction d (component own =
// own_dir). d==own_dir → ×2 diffusion (cell μ; single for own-X, two-cell for
// own-Y/own-Z) + own 0.25 convection on the dx (cell-width) metric + the 0.5·∂q
// own term. d≠own_dir → ×1 diffusion with the FAITHFUL doubly volume-weighted
// cross-face μ (interp through own_dir to the velocity face, then through d —
// matches the RHS faces mu1..mu6, ref 675-685; a no-op at μ≡1) + 0.25 convection
// on the dmx metric. invρ is own-face interpolated (ref invRhoc). q = own
// component field (for the 0.5·∂q own conv), q_adv = advecting velocity along d.
// dx_along/dmx_along = swept-dir metric; dx_own/dmx_own = own_dir metric (for the
// face interps). z_staggered = wall-normal own-Z metric.
void build_adi_bands_var(Direction      d,
                         Direction      own_dir,
                         real_t*        A, real_t* B, real_t* C, real_t* D,
                         const real_t*  src,
                         const real_t*  q,
                         const real_t*  q_adv,
                         const real_t*  mu, const real_t* invrho,
                         const real_t*  dx_along, const real_t* dmx_along,
                         const real_t*  dx_own,   const real_t* dmx_own,
                         int            n1_tot, int n2_tot, int n3_tot,
                         int            n1_int,  int n2_int,  int n3_int,
                         real_t         Cmu, real_t dt,
                         bool           z_staggered = false);

// ── Explicit source accumulation ─────────────────────────────────────────────
// rhs[c] += dt * src[c]  (interior only)
void add_source_rhs(real_t*       rhs,
                     const real_t* src,
                     int n1_tot, int n2_tot, int n3_tot,
                     real_t dt);

// ── Uniform body force ───────────────────────────────────────────────────────
// rhs[c] += val  (interior only)  — used for mean pressure gradient in channel flow
void add_constant_rhs(real_t* rhs,
                       int n1_tot, int n2_tot, int n3_tot,
                       real_t val);

// ── ADI band assembly (MPM-STD BW-ADI) ──────────────────────────────────────
// Build (A, B, C) bands and pack src into D for the tridiagonal system
//   (I − visc_factor·dt·ν·L_d − 0.25·dt·conv_d) δ = src
// along direction d.  Layout: D[row * n_sys + sys].
//
// visc_factor = 1.0 for the "own" direction (backward Euler)
// visc_factor = 0.5 for cross directions   (Crank-Nicolson)
//
// q      : current field values (for convective terms in bands)
// q_adv  : advecting velocity in direction d (U for X, V for Y, W for Z)
void build_adi_bands(Direction      d,
                      real_t*        A, real_t* B, real_t* C, real_t* D,
                      const real_t*  src,
                      const real_t*  q,
                      const real_t*  q_adv,
                      const real_t*  dx_along, const real_t* dmx_along,
                      int            n1_tot, int n2_tot, int n3_tot,
                      int            n1_int,  int n2_int,  int n3_int,
                      real_t         viscosity, real_t dt,
                      real_t         visc_factor = real_t{0.5},
                      bool           z_staggered = false);

// ── Scatter TDMA solution back to 3-D buffer ─────────────────────────────────
void scatter_from_tdma(Direction      d,
                        real_t*        dst,
                        const real_t*  D_solution,
                        int n1_tot, int n2_tot, int n3_tot,
                        int n1_int,  int n2_int,  int n3_int);

// ── q += delta (interior cells) ──────────────────────────────────────────────
void add_increment(real_t*       q,
                    const real_t* delta,
                    int n1_tot, int n2_tot, int n3_tot);

// ── MPM-STD block-coupling correction (constant viscosity, convection part) ──
//
// The fully-coupled MPM-STD momentum solve treats the off-diagonal velocity
// blocks (how δV, δW feed back into the U-equation, and δW into the V-equation)
// with a block lower-triangular Gauss-Seidel step applied AFTER the three
// diagonal ADI predictors and BEFORE the velocity update:
//
//   dV ← dV − dt·0.25·( dW·∂V/∂z )                      (blockLdV: M23·dW)
//   dU ← dU − dt·0.25·( dV·∂U/∂y + dW·∂U/∂z )           (blockLdU: M12·dV + M13·dW)
//
// This cross-velocity convection of the increments is the streak↔vortex coupling
// that sustains near-wall turbulence; omitting it lets the channel relaminarize.
// The variable-viscosity cross-stress terms (∂/∂y(μ ∂δV/∂x), …) are dropped here
// because for constant viscosity they are a continuity correction that the
// pressure projection already enforces.
//
// dU, dV, dW are the increment buffers (must have valid halos for the stencil).
// U, V are the current-step velocity fields (u^n).  Interior cells only.

// dV −= dt·0.25·(dW·∂V/∂z) interpolated to V (y-face) locations.
void block_couple_dV(real_t*       dV,
                     const real_t* dW,
                     const real_t* V,
                     const real_t* dx2, const real_t* dmx2,
                     const real_t* dx3, const real_t* dmx3,
                     int n1_tot, int n2_tot, int n3_tot,
                     real_t dt);

// dU −= dt·0.25·(dV·∂U/∂y + dW·∂U/∂z) interpolated to U (x-face) locations.
void block_couple_dU(real_t*       dU,
                     const real_t* dV,
                     const real_t* dW,
                     const real_t* U,
                     const real_t* dx1, const real_t* dmx1,
                     const real_t* dmx2,
                     const real_t* dmx3,
                     int n1_tot, int n2_tot, int n3_tot,
                     real_t dt);

// ── FAITHFUL variable-μ block coupling — MPM-STD_main arrangement ─────────────
// (ref MPM-STD core_momentum.f90, driver submodule.f90:202-227; Report/RBC
// wall-reignition analysis.) The reference applies all six off-diagonal velocity
// couplings with the solve order U → V → W (wall-normal LAST): three LOWER
// (M21/M31/M32) via the dd* terms injected into the next component's RHS before
// its implicit ADI solve, three UPPER (M23/M12/M13) as explicit post-hoc
// corrections to the HORIZONTAL increments dV/dU only. The wall-normal dW never
// receives an unfiltered explicit correction — the previous PaScaL-transposed
// arrangement did exactly that (block dW ← dV) and it was the measured cold-wall
// reignition pump (Report/30 E9, Report/36).
//
// block_couple_dV_var ≡ ref blockLdV 1612-1688 (M23dWm at 1682): corrects the
//   horizontal dV using the solved wall-normal increment dW.
//     dV -= dt·( 0.25·(ddw3·∂V/∂z_lo + ddw4·∂V/∂z_hi)
//              − 0.5·Cmu·invρ̂/dx3·(μz_hi·∂y(dW)_hi − μz_lo·∂y(dW)_lo) )
//   (ddw = dW interp to the V node via dx2; μz = z-face μ, ref mu5/mu6.)
void block_couple_dV_var(real_t*       dV,
                         const real_t* dW,
                         const real_t* V,
                         const real_t* mu, const real_t* invrho,
                         const real_t* dx2, const real_t* dmx2,
                         const real_t* dx3, const real_t* dmx3,
                         int n1_tot, int n2_tot, int n3_tot,
                         real_t Cmu, real_t dt);

// block_couple_dU_var ≡ ref blockLdU 1690-1785: corrects dU using the corrected
//   span increment (dV, M12 via ∂U/∂y) and the wall-normal increment (dW, M13
//   via ∂U/∂z), full viscous cross-stress of the increments. Run AFTER
//   block_couple_dV_var + dV halo (the reference uses the corrected dv).
void block_couple_dU_var(real_t*       dU,
                         const real_t* dV,
                         const real_t* dW,
                         const real_t* U,
                         const real_t* mu, const real_t* invrho,
                         const real_t* dx1, const real_t* dmx1,
                         const real_t* dx2, const real_t* dmx2,
                         const real_t* dx3, const real_t* dmx3,
                         int n1_tot, int n2_tot, int n3_tot,
                         real_t Cmu, real_t dt);

// ── FAITHFUL dd* forward-substitution coupling (block LOWER-triangular) ───────
// The three M21/M31/M32 off-diagonals applied to the RHS using the already-
// solved preceding increments, BEFORE the component's own ADI solve. Reference
// solve order solvedU→solvedV→solvedW = my dU→dV→dW; the forward sub is:
//   dV (ref solvedV, M21):     rhs_v -= dt·(...dU...)                (couple_dd_dV_var)
//   dW (ref solvedW, M31+M32): rhs_w -= dt·(...dU...) -= dt·(...dV...) (couple_dd_dW_var)
// These do NOT vanish at μ≡1 (increment×velocity-gradient convection). Every
// coupling into the stiff wall-normal dW passes through its implicit z-ADI —
// the stabiliser for strong viscosity contrast (glycerol). The preceding
// increments must be synced before each call.

// couple_dd_dV_var ≡ ref solvedV 1211-1218 (M21ddU): dU increment → dV RHS.
void couple_dd_dV_var(real_t*       rhs_v,
                      const real_t* dU,
                      const real_t* V,
                      const real_t* mu, const real_t* invrho,
                      const real_t* dx1, const real_t* dmx1,
                      const real_t* dx2, const real_t* dmx2,
                      int n1_tot, int n2_tot, int n3_tot,
                      real_t Cmu, real_t dt);

// couple_dd_dW_var ≡ ref solvedW 1524-1540 (M31ddU + M32ddV): dU and dV
// increments → wall-normal (dW) RHS. Run AFTER the dV solve + sync.
void couple_dd_dW_var(real_t*       rhs_w,
                      const real_t* dU,
                      const real_t* dV,
                      const real_t* W,
                      const real_t* mu, const real_t* invrho,
                      const real_t* dx1, const real_t* dmx1,
                      const real_t* dx2, const real_t* dmx2,
                      const real_t* dx3, const real_t* dmx3,
                      int n1_tot, int n2_tot, int n3_tot,
                      real_t Cmu, real_t dt);

// ── NOB transpose viscous stress (cross-stress), component i ─────────────────
//   T_i = Σ_j ∂_j(μ ∂_i u_j)  — the ∇u^T half of ∇·(μ(∇u+∇u^T)). Added to the
// component RHS as dt·Cmu·invRho·T_i. At μ=const it is ∂_i(∇·u) ≈ 0 (so it does
// not perturb the μ≡1/OB validation), and combined with the diagonal own term
// it doubles the own direction (matching the reference 2× factor). This is the
// piece the diagonal-only var-μ RHS dropped; it is required for the strong
// glycerol viscosity contrast (NOB_RBC_SPEC.md §3). μ is volume-weighted to the
// edge between the component's face and the cross face; z = wall-normal staggered.
void add_transpose_stress_u_cpu(real_t* rhs_u, const real_t* U, const real_t* V, const real_t* W,
    const real_t* mu, const real_t* invrho,
    const real_t* dx1, const real_t* dmx1, const real_t* dx2, const real_t* dmx2,
    const real_t* dx3, const real_t* dmx3, int n1, int n2, int n3, real_t Cmu, real_t dt);
void add_transpose_stress_v_cpu(real_t* rhs_v, const real_t* U, const real_t* V, const real_t* W,
    const real_t* mu, const real_t* invrho,
    const real_t* dx1, const real_t* dmx1, const real_t* dx2, const real_t* dmx2,
    const real_t* dx3, const real_t* dmx3, int n1, int n2, int n3, real_t Cmu, real_t dt);
void add_transpose_stress_w_cpu(real_t* rhs_w, const real_t* U, const real_t* V, const real_t* W,
    const real_t* mu, const real_t* invrho,
    const real_t* dx1, const real_t* dmx1, const real_t* dx2, const real_t* dmx2,
    const real_t* dx3, const real_t* dmx3, int n1, int n2, int n3, real_t Cmu, real_t dt);

} // namespace mpmstd::equation::momentum::kernels

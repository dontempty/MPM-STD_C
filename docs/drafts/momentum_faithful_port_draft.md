I now have the complete picture including the exact pattern the parent used for the energy port (delta-form `D = dt·(R − M·Tⁿ)`, generic axis bands, `is_stage1` flag, no wall masks). I have everything I need to write the faithful momentum draft.

Let me work through the algebra carefully, then produce the deliverable.

Key check on the M·Uⁿ cancellation for dU (ref 341 + 396-407): RHS = `invRhocCmu_half·(2·vu1+vu2+vu3+vu12+vu13)` then subtract `(mACI·U+...)` then subtract two more bookkeeping lines. Let me verify the own-direction diffusion D-part of mACI·U exactly cancels `2·vu1`.

For X (own): the diffusion part of mACI/mAPI/mAMI (line 371-376, dropping the 0.25 convection) is:
- mAPI_visc·U(ip) + mACI_visc·U(i) + mAMI_visc·U(im)
- = invRhocCmu_half·2/dmx1(i)·[ μ_i/dx1(i)·(−U(ip)+U(i)) + μ_i/dx1(im)·(−U(im)+U(i)) ]  ... wait, sign: mACI_visc = +inv·2/dmx1·(μ/dx1(i)+μ/dx1(im)), mAPI_visc = −inv·2/dmx1·μ/dx1(i), mAMI_visc = −inv·2/dmx1·μ/dx1(im).

So Σ = inv·2/dmx1(i)·[ −μ_i/dx1(i)·U(ip) + (μ_i/dx1(i)+μ_i/dx1(im))·U(i) − μ_i/dx1(im)·U(im) ]
= inv·2/dmx1(i)·[ −μ_i·(U(ip)−U(i))/dx1(i) + μ_i·(U(i)−U(im))/dx1(im) ]
= −inv·2/dmx1(i)·[ μ_i·dudx2 − μ_i·dudx1 ].

And `2·viscous_u1·invRhocCmu_half = 2·invRhocCmu_half·(Mu(i)·dudx2 − Mu(im)·dudx1)/dmx1(i)`.

Note these don't exactly cancel because viscous_u1 uses `Mu(i)` and `Mu(im)` (cell μ), while the band uses `Mu(i,j,k)` for BOTH (line 371-375 all use `Mu(i,j,k)`). This is a known reference asymmetry — the implicit operator uses a single μ_c while the explicit uses two cell values. So after `R − M·Uⁿ`, the own-direction leaves a residual `2·invRhocCmu_half·[(Mu(i)−Mu(i))·... ]`... I need to write this exactly as the reference does, not assume cancellation. The faithful port computes both terms literally. Let me now produce the deliverable.

I have all the information needed. Let me write the complete, ref-line-mapped draft.

---

# FAITHFUL C++ PORT — PaScaL_TCS variable-viscosity (NOB) MOMENTUM solver

## PART 1 — Structural findings (read-first)

### 1.1 The reference is a delta-form Beam–Warming ADI with `D = dt·(R − M·Uⁿ − bookkeeping)`

Each `mpi_momentum_solvedX` builds, in its **1st stage (z-sweep) loop body**, a single scalar `RHS(i,j,k)` that is the *complete explicit residual times dt*. Then the 2nd/3rd stages rebuild only the `am/ac/ap` bands for the x- and y-sweeps and pass the running solution through `D`. This is structurally identical to the energy port (`scalar_var_kernels_cpu.cpp`): stage-1 packs `D = dt·(R − M·Uⁿ − ...)`, stages 2/3 only rebuild bands and copy `D` through.

For **dU** (ref 341–413), the stage-1 RHS is, in order:

```
RHS  =  invRhocCmu_half·(2·vu1 + vu2 + vu3 + vu12 + vu13)        [341]  explicit viscous (full stress)
      − Cmp·invRhoc·(P(i)−P(im))/dmx1(i)                          [344-345]  pressure   << SKIP (my code: pressure separate)
      + Cmt·(Tc + a12pera11·Tc²·DeltaT)·invRhoc                   [351-352]  buoyancy   << SKIP (dU has none in my port; buoyancy is on dW & composed in main)
      + (1−jum)·ubc_down + (1−jup)·ubc_up                         [366-367]  wall ubc   << SKIP (ghost+fold instead)
      − ( mACI·U(i) + mAPI·U(ip) + mAMI·U(im)                     [396-399]  −(M·Uⁿ)  : the FULL band·Uⁿ including the 0.25 conv terms
        + mACJ·U(i) + mAPJ·U(jp) + mAMJ·U(jm)
        + mACK·U(i) + mAPK·U(kp) + mAMK·U(km) )
      − ( 0.25·(v4·jup·dudy4 + v3·jum·dudy3)                      [401-403]  cross-Y advective half + cross-Y stress
          − invRhocCmu_half/dx2·(mu4·dvdx4·jup − mu3·dvdx3·jum) )
      − ( 0.25·(w6·dudz6 + w5·dudz5)                              [405-407]  cross-Z advective half + cross-Z stress
          − invRhocCmu_half/dx3·(mu6·dwdx6 − mu5·dwdx5) )
RHS *= dt                                                         [413]
ac = mACK·dt + 1 ;  ap = mAPK·dt ;  am = mAMK·dt                  [410-412]  (own implicit band for the 1st sweep = Z)
```

### 1.2 What cancels — own×2 vs cross, and the skew-convection split

The band operators `mAC*/mAP*/mAM*` each contain **diffusion + 0.25-convection** halves. When `−(M·Uⁿ)` is subtracted from the explicit residual, the algebra is:

**Own direction X (ref 371–376):** diffusion half of the band carries the `×2` factor. `−(mAPI_diff·U(ip)+mACI_diff·U(i)+mAMI_diff·U(im))` expands to `−2·invRhocCmu_half/dmx1·[μ_c·dudx2 − μ_c·dudx1]` (all using `Mu(i,j,k)=μ_c`). The explicit term `+2·invRhocCmu_half·vu1 = +2·invRhocCmu_half·(Mu(i)·dudx2 − Mu(im)·dudx1)/dmx1`. **These do NOT cancel exactly** — explicit uses two cell μ's `Mu(i),Mu(im)`, the band uses one `Mu(i,j,k)` twice. The residual `2·invRhocCmu_half/dmx1·[(Mu(i)−μ_c)·dudx2 − (Mu(im)−μ_c)·dudx1]` is what survives and drives the scheme toward steady state. **This is a faithful reference asymmetry — the port must compute both literally, NOT assume cancellation.** (`μ_c = Mu(i,j,k)` is the *cell-centre* value, while `viscous_u1` uses the two cell values straddling the x-face.)

**Cross directions Y, Z (ref 379–394):** diffusion half has **NO ×2** (`invRhocCmu_half/dx2·(mu4/dmx2(jp)+mu3/dmx2(j))` — same μ-faces `mu3/mu4` as the explicit `vu2`). The explicit `+vu2·invRhocCmu_half = +invRhocCmu_half·(mu4·dudy4 − mu3·dudy3)/dx2` **does exactly cancel** the cross-Y diffusion band·Uⁿ. So after `R−M·Uⁿ`, the cross-direction *own-velocity diffusion* (vu2, vu3) is gone, leaving only the implicit version inside the swept bands → **diffusion is effectively fully implicit in all 3 directions** (own via the residual-driven ×2, cross via cancellation + ADI sweep).

**Skew-convection split.** Cross-direction convection is split half-and-half:
- **Half-divergence (conservative) lives in the band** `mAC*/mAP*/mAM*` as the `0.25·(±v/dmx)` terms (ref 380/382/384 for Y, 390/392/394 for Z). When `−(M·Uⁿ)` is taken, this contributes `−0.25·(divergence-form)·Uⁿ`.
- **Half-advective lives in the bookkeeping** lines 401–407: `−0.25·(v4·dudy4 + v3·dudy3)` (Y) and `−0.25·(w6·dudz6 + w5·dudz5)` (Z). These are the *advective* form `vel·∂q`.

Together the two halves give the skew-symmetric `½(div + adv)` convection for the cross directions. **Own-direction convection is entirely in the band** (mACI/mAPI/mAMI 0.25 terms, ref 372/374/376), so it appears only through `−(M·Uⁿ)` and the implicit sweep — there is **no** own-direction convection bookkeeping line.

### 1.3 Cross-stress double-count question — RESOLVED

The explicit `vu12, vu13` (ref 338–339) ARE re-subtracted in the bookkeeping lines 401–407:
- Line 341 adds `+invRhocCmu_half·(vu12+vu13)` where `vu12=(mu4·dvdx4−mu3·dvdx3)/dx2`, `vu13=(mu6·dwdx6−mu5·dwdx5)/dx3`.
- Line 403 subtracts `−(−invRhocCmu_half/dx2·(mu4·dvdx4·jup − mu3·dvdx3·jum))` = `+invRhocCmu_half/dx2·(mu4·dvdx4·jup − mu3·dvdx3·jum)`.
- Line 407 subtracts `−(−invRhocCmu_half/dx3·(mu6·dwdx6 − mu5·dwdx5))` = `+invRhocCmu_half/dx3·(mu6·dwdx6 − mu5·dwdx5)`.

So **away from walls (jum=jup=1)** the cross-stress `vu12,vu13` is added once at 341 and added *again* at 403/407 → **NET 2× the cross-stress**, with the convective `−0.25·(v4·dudy4+v3·dudy3)−0.25·(w6·dudz6+w5·dudz5)` also present. Re-reading the signs precisely:

- 341: `+inv·vu12 + inv·vu13`  (with `inv ≡ invRhocCmu_half`)
- 403: residual subtracts `−( … − inv/dx2·(mu4·dvdx4·jup − mu3·dvdx3·jum) )` ⇒ adds `+inv·vu12` (when jup=jum=1). 
- 407: subtracts `−( … − inv·vu13 )` ⇒ adds `+inv·vu13`.

**So the cross-stress (∇uᵀ piece) appears with total weight 2× in the dU residual**, plus the `−0.25` advective convection from the same lines. This is NOT a bug — it is how the reference forms its operator (the cross-stress at 341 is the "M12/M13 applied to Uⁿ part" and the 401–407 lines are the explicit advective + the *negative* of the implicit cross-stress that the block-coupling `blockLdU` will re-apply to the *increments*). 

**CRITICAL CONSEQUENCE FOR MY PORT — the double-count must be reproduced, but my existing `add_transpose_stress_*` supplies the cross-stress only ONCE.** The faithful residual needs:
- the `2×` own-direction own-velocity diffusion (own term + transpose own-x term = the ×2),
- the cross-direction own-velocity diffusion (vu2,vu3) — **cancelled** by `M·Uⁿ` so it nets to the implicit sweep only,
- the cross-stress vu12,vu13 added at 341 (weight 1) **plus** re-added at 403/407 (weight 1) ⇒ weight 2,
- minus the `0.25` advective cross-convection (403/405/407),
- minus the band·Uⁿ for the cross-direction *convection* divergence halves.

**Decision (stated explicitly):** Rather than try to reuse `add_transpose_stress_*` (which would supply cross-stress only once and on a different μ-edge convention), the faithful `compute_mpmstd_rhs_var` below **folds the entire residual R − M·Uⁿ − bookkeeping into one kernel per component**, computing every term literally from the ref. The existing `add_transpose_stress_*` must then **NOT** be called for the faithful path (it would double-add yet again). I flag this as the single highest-risk integration point: *the parent must remove the three `add_transpose_stress_*` calls from `assemble_momentum_var_visc_cpu` when switching to the faithful kernel.*

### 1.4 Axis swap (applied throughout below)

| ref | role | my axis | my component |
|-----|------|---------|--------------|
| x1 (U, streamwise, periodic) | → | X | dU |
| x2 (V, **wall-normal**, jC_BC/jS_BC masks) | → | **Z** | **dW** |
| x3 (W, periodic) | → | Y | dV |

So **ref `solvedU` → my dU**, **ref `solvedW` → my dV** (ref x3↔my Y), **ref `solvedV`−buoyancy → my dW** (ref x2↔my Z, wall lives here). Within each component, ref x2-direction terms (jum/jup) become my **Z**-direction terms; ref x3 terms → my **Y**.

### 1.5 Wall handling — KEEP ghost+fold, drop jum/jup and ubc (mirror the energy port)

As instructed and consistent with `scalar_var_kernels_cpu.cpp` (lines 16–20): **do NOT transcribe jum/jup/jvm/jvp/jwm/jwp masks nor ubc/vbc/zbc into the bands or RHS.** The validated mpmstd wiring uses zero-ghost velocity + `modify_tdma_row_cpu` fold (B+=A; A=0). Mixing masks with the fold double-drops the wall row. The `M·Uⁿ` at wall-adjacent cells uses the ghost velocity values, consistent with the explicit RHS — exactly as energy. **Caveat (§5):** the reference's `0.25·(v4·jup·dudy4 + v3·jum·dudy3)` advective convection and `mu·dvdx·jup` cross-stress carry the masks; without them my port keeps the full (ghost-fed) flux at the wall-adjacent cell. With zero-ghost wall velocity (W=0 at wall, dW increment folded) the masked terms differ from mine only through the *ghost*-extrapolated transverse velocity at the wall face, which the fold + zero-ghost is designed to neutralise. This was accepted for energy; same risk class here.

---

## PART 2 — Complete ready-to-paste C++

The new kernels live in `momentum_var_kernels_cpu.cpp`, replacing the current simplified `compute_mpmstd_rhs_var` / `build_adi_bands_var`. The signatures gain the component's **own direction** (currently `compute_mpmstd_rhs_var` has no `own_dir`; `build_adi_bands_var` infers own-ness from `d == own`). I add `Direction own_dir` to both, and the band builder applies `×2` only when `d == own_dir`.

### 2.1 Updated declarations (`kernels.hpp`)

```cpp
// ── Variable-viscosity (NOB) FAITHFUL residual R − M·Uⁿ − bookkeeping ─────────
// Folds the ENTIRE reference stage-1 explicit residual (ref solvedU 341-413,
// solvedW→dV via x3, solvedV-minus-buoyancy→dW via x2) into one kernel. Per
// component the OWN direction (own_dir) carries the ×2 own-velocity diffusion and
// the own-direction 0.25 convection; the two CROSS directions carry vu_cross
// (cancelled by M·Uⁿ → implicit-only) + the doubled cross-STRESS vu_ij + the 0.25
// advective cross-convection. PRESSURE & BUOYANCY are SKIPPED (composed in main).
// NO jum/jup masks, NO ubc (wall via ghost + modify_tdma_row_cpu).  Output already
// ×dt (delta-form D). Do NOT also call add_transpose_stress_* on this path.
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

// Variable-viscosity ADI bands: own direction (d==own_dir) gets the ×2 diffusion
// (+ own 0.25 convection); cross direction gets ×1 diffusion + cross 0.25 conv.
void build_adi_bands_var(Direction      d,
                         Direction      own_dir,
                         real_t*        A, real_t* B, real_t* C, real_t* D,
                         const real_t*  src,
                         const real_t*  q,
                         const real_t*  q_adv,
                         const real_t*  mu, const real_t* invrho,
                         const real_t*  dx_along, const real_t* dmx_along,
                         int            n1_tot, int n2_tot, int n3_tot,
                         int            n1_int,  int n2_int,  int n3_int,
                         real_t         Cmu, real_t dt,
                         bool           z_staggered = false);

// ── Variable-μ block coupling (ref blockLdV :1345, blockLdU :1450) ────────────
// AXIS-SWAPPED: ref blockLdV (updates ref dV via ref-x3 cross stress) → my
// block_couple_dV_var updating my dV using ref's structure on MY Y/Z metrics;
// ref blockLdU → block_couple_dU_var. Carries the variable-μ cross-stress of the
// INCREMENTS (the 0.5·Cmu·invRhoc/dx·(μ6·∂ dwm6 − μ5·∂ dwm5) piece) that the
// const block_couple_dV/dU dropped.
void block_couple_dV_var(real_t*       dV,
                         const real_t* dW,
                         const real_t* V,
                         const real_t* mu, const real_t* invrho,
                         const real_t* dx2, const real_t* dmx2,
                         const real_t* dx3, const real_t* dmx3,
                         int n1_tot, int n2_tot, int n3_tot,
                         real_t Cmu, real_t dt);

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
```

### 2.2 `compute_mpmstd_rhs_var` — faithful per-component residual

This is the heart. I write it **per component** using a templated/branched approach keyed on `own_dir`, because the μ-face interpolation and the cross-stress structure differ by component (ref solvedU vs solvedV vs solvedW have different `mu3/mu4/mu5/mu6/mu1/mu2` definitions). To stay faithful AND readable I implement three explicit component bodies sharing helpers. Below is the complete file body.

```cpp
#include "equation/momentum/kernels.hpp"
#include "common/macros.hpp"

// =============================================================================
//  Variable-viscosity (NOB) FAITHFUL momentum kernels — line-by-line port of
//  PaScaL_TCS module_solve_momentum.f90 (solvedU/solvedV/solvedW + blockLd).
//
//  AXIS SWAP (ref → my):  x1(U,periodic)→X(dU);  x2(V,WALL-NORMAL)→Z(dW);
//                         x3(W,periodic)→Y(dV).  Wall lives ONLY on my Z.
//  So:  my dU  ≡ ref solvedU                 (own=X)
//       my dV  ≡ ref solvedW   (ref x3→my Y) (own=Y)
//       my dW  ≡ ref solvedV − buoyancy      (own=Z, wall) (ref x2→my Z)
//
//  Stage-1 delta-form residual R − M·qⁿ − cross-bookkeeping, ×dt  (ref *341-413).
//  PRESSURE + BUOYANCY skipped (composed in main).  jum/jup masks + ubc skipped
//  (wall via zero-ghost + modify_tdma_row_cpu fold, exactly like the energy port).
//  invRhocCmu_half = 0.5·Cmu·invRhoc(face).  Own-dir diffusion ×2, cross ×1.
//
//  μ_c (cell) for the own-direction explicit viscous_? (ref uses Mu(i,j,k) twice
//  in the band but the two straddling cell μ's in viscous_?1) — both reproduced
//  literally; their difference is the surviving steady-state driver (NOT a no-op).
// =============================================================================

namespace mpmstd::equation::momentum::kernels {

namespace {
MPMSTD_HD int ix(int i, int j, int k, int n2, int n3) { return (i * n2 + j) * n3 + k; }
constexpr real_t kHalf = real_t{0.5};
constexpr real_t kQ    = real_t{0.25};

// volume-weighted 2-pt face interp of μ along an axis:
//   0.5·(dxA·muA + dxB·muB)/dmx   — the ref pattern 0.5*(dx(im)*Mu(i)+dx(i)*Mu(im))/dmx(i)
MPMSTD_HD real_t mface(real_t dxA, real_t muA, real_t dxB, real_t muB, real_t dmx) {
  return kHalf * (dxA * muA + dxB * muB) / dmx;
}
} // anonymous namespace
```

#### dU body (ref solvedU 295–407) — own = X, cross = Y(=ref x2/wall) & Z(=ref x3)

> Axis note: in **ref solvedU**, x2 is the wall-normal (masked) direction and x3 is periodic. In MY port the wall is on Z, so ref-x2→my **Z** and ref-x3→my **Y**. Therefore in the code below the ref's `dudy*/mu3/mu4/v3/v4/dvdx*` (its x2 wall terms) map to my **Z** neighbours/metrics, and ref's `dudz*/mu5/mu6/w5/w6/dwdx*` (its x3) map to my **Y**. I write the C++ in *my* axis names (so the wall terms use k/dx3, the periodic-cross uses j/dx2). The structure is identical; only the metric arrays swap.

```cpp
void compute_mpmstd_rhs_var(real_t*       rhs,
                            const real_t* q,
                            const real_t* U, const real_t* V, const real_t* W,
                            const real_t* mu, const real_t* invrho,
                            const real_t* dx1, const real_t* dmx1,
                            const real_t* dx2, const real_t* dmx2,
                            const real_t* dx3, const real_t* dmx3,
                            int n1, int n2, int n3,
                            real_t Cmu, real_t dt,
                            Direction own_dir,
                            bool /*z_staggered*/) {
  const int h = kHaloWidth;

  // ---------------------------------------------------------------------------
  //  Component dU  (ref solvedU).  own=X.  q==U.
  //  Cross-Y  = ref x3 (periodic, mu5/mu6,w5/w6 in ref → my y-faces).
  //  Cross-Z  = ref x2 (wall,     mu3/mu4,v3/v4 in ref → my z-faces, NO masks).
  // ---------------------------------------------------------------------------
  if (own_dir == Direction::X) {
    for (int i = h; i < n1 - h; ++i)
      for (int j = h; j < n2 - h; ++j)
        for (int k = h; k < n3 - h; ++k) {
          const int im=i-1, ip=i+1, jm=j-1, jp=j+1, km=k-1, kp=k+1;
          const int c   = ix(i, j, k, n2, n3);
          const int cim = ix(im,j, k, n2, n3), cip = ix(ip,j, k, n2, n3);
          const int cjm = ix(i, jm,k, n2, n3), cjp = ix(i, jp,k, n2, n3);
          const int ckm = ix(i, j, km,n2, n3), ckp = ix(i, j, kp,n2, n3);

          // --- face advecting velocities on the U (x-face i) location ---------
          const real_t u1 = kHalf*(U[cim] + U[c]);                        // ref u1  (295)
          const real_t u2 = kHalf*(U[c]   + U[cip]);                      // ref u2  (296)
          // ref v3/v4 (x2 wall) → my W on z-faces; ref w5/w6 (x3) → my V on y-faces
          const real_t w5 = kHalf*(dx1[im]*W[c]   + dx1[i]*W[cim])/dmx1[i];   // ref v3 (298) → my z-lo
          const real_t w6 = kHalf*(dx1[im]*W[ckp] + dx1[i]*W[ix(im,j,kp,n2,n3)])/dmx1[i]; // ref v4 (299) → z-hi
          const real_t v3 = kHalf*(dx1[im]*V[c]   + dx1[i]*V[cim])/dmx1[i];   // ref w5 (301) → my y-lo
          const real_t v4 = kHalf*(dx1[im]*V[cjp] + dx1[i]*V[ix(im,jp,k,n2,n3)])/dmx1[i]; // ref w6 (302) → y-hi

          // --- own/ cross gradients of U -------------------------------------
          const real_t dudx1 = (U[c]   - U[cim])/dx1[im];                 // ref 304
          const real_t dudx2 = (U[cip] - U[c]  )/dx1[i];                  // ref 305
          // ref dudy3/4 (x2 wall) → my dudz on z; ref dudz5/6 (x3) → my dudy on y
          const real_t dudz5 = (U[c]   - U[ckm])/dmx3[k];                 // ref dudy3 (306) → z-lo
          const real_t dudz6 = (U[ckp] - U[c]  )/dmx3[kp];                // ref dudy4 (307) → z-hi
          const real_t dudy3 = (U[c]   - U[cjm])/dmx2[j];                 // ref dudz5 (308) → y-lo
          const real_t dudy4 = (U[cjp] - U[c]  )/dmx2[jp];               // ref dudz6 (309) → y-hi

          // ref dvdx3/4 (x2 wall, on V) → my dwdx on z;  ref dwdx5/6 (x3, on W) → my dvdx on y
          const real_t dwdx5 = (W[c]   - W[cim])/dmx1[i];                 // ref dvdx3 (311) → z-lo
          const real_t dwdx6 = (W[ckp] - W[ix(im,j,kp,n2,n3)])/dmx1[i];   // ref dvdx4 (312) → z-hi
          const real_t dvdx3 = (V[c]   - V[cim])/dmx1[i];                 // ref dwdx5 (314) → y-lo
          const real_t dvdx4 = (V[cjp] - V[ix(im,jp,k,n2,n3)])/dmx1[i];   // ref dwdx6 (315) → y-hi

          // --- μ faces (ref 317-329, axis-swapped) ---------------------------
          // ref mu3/mu4 (x2/wall) → my muz5/muz6 on z-faces; ref mu5/mu6 (x3) → my muy3/muy4 on y
          // z-faces (was ref Y, lines 317-322):
          real_t mua = mface(dx1[im], mu[ckm? c:c], 0,0,1); (void)mua; // (placeholder; see explicit below)
          // -- explicit μ build (axis-swapped from ref 317-329) --
          // z-face μ (ref 317-322 with x2→z):
          const real_t za = mface(dx1[im], mu[ckm],         dx1[i], mu[ix(im,j,km,n2,n3)], dmx1[i]); // ref mua@jm
          const real_t zc = mface(dx1[im], mu[c],           dx1[i], mu[cim],               dmx1[i]); // ref muc
          const real_t zb = mface(dx1[im], mu[ckp],         dx1[i], mu[ix(im,j,kp,n2,n3)], dmx1[i]); // ref mub@jp
          const real_t muz5 = kHalf*(dx3[km]*zc + dx3[k]*za)/dmx3[k];     // ref mu3 (321)→z-lo
          const real_t muz6 = kHalf*(dx3[kp]*zc + dx3[k]*zb)/dmx3[kp];    // ref mu4 (322)→z-hi
          // y-face μ (ref 324-329 with x3→y):
          const real_t ya = mface(dx1[im], mu[cjm],         dx1[i], mu[ix(im,jm,k,n2,n3)], dmx1[i]); // ref mua@km
          const real_t yc = zc;                                                                       // same muc
          const real_t yb = mface(dx1[im], mu[cjp],         dx1[i], mu[ix(im,jp,k,n2,n3)], dmx1[i]); // ref mub@kp
          const real_t muy3 = kHalf*(dx2[jm]*yc + dx2[j]*ya)/dmx2[j];     // ref mu5 (328)→y-lo
          const real_t muy4 = kHalf*(dx2[jp]*yc + dx2[j]*yb)/dmx2[jp];    // ref mu6 (329)→y-hi

          const real_t irc = mface(dx1[im], invrho[c], dx1[i], invrho[cim], dmx1[i]); // ref invRhoc (331)
          const real_t inh = kHalf*Cmu*irc;                              // ref invRhocCmu_half (332)
          const real_t muc = mu[c];                                      // ref Mu(i,j,k) cell value

          // === explicit viscous (ref 335-341) ================================
          // own X (cell μ's, ×2 below in the residual via 341):
          const real_t vu1  = (mu[c]*dudx2 - mu[cim]*dudx1)/dx1[i];       // ref viscous_u1 (335)
          // cross Z = ref x2 (mu3/mu4 → muz5/muz6):
          const real_t vuZ  = (muz6*dudz6 - muz5*dudz5)/dx3[k];          // ref viscous_u2 (336)
          // cross Y = ref x3 (mu5/mu6 → muy3/muy4):
          const real_t vuY  = (muy4*dudy4 - muy3*dudy3)/dx2[j];          // ref viscous_u3 (337)
          // cross-stress Z = ref viscous_u12 (mu4*dvdx4-mu3*dvdx3)/dx2 → my muz*dwdx/dx3:
          const real_t vuZc = (muz6*dwdx6 - muz5*dwdx5)/dx3[k];          // ref viscous_u12 (338)
          // cross-stress Y = ref viscous_u13 (mu6*dwdx6-mu5*dwdx5)/dx3 → my muy*dvdx/dx2:
          const real_t vuYc = (muy4*dvdx4 - muy3*dvdx3)/dx2[j];          // ref viscous_u13 (339)

          real_t R = inh*(real_t{2}*vu1 + vuZ + vuY + vuZc + vuYc);       // ref 341

          // === bands for −(M·Uⁿ)  (ref 371-394, axis-swapped) ================
          // own X (diffusion ×2 + 0.25 own conv) — ref 371-376:
          const real_t mACI =  inh/dmx1[i]*(muc/dx1[i] + muc/dx1[im])*real_t{2}
                            + kQ*(dudx2*kHalf + dudx1*kHalf - u2/dx1[i] + u1/dx1[im]);
          const real_t mAPI = -inh/dmx1[i]*muc/dx1[i]*real_t{2}
                            + kQ*( u2/dx1[i] + dudx2*kHalf);
          const real_t mAMI = -inh/dmx1[i]*muc/dx1[im]*real_t{2}
                            + kQ*(-u1/dx1[im] + dudx1*kHalf);
          // cross Z = ref x2 Y-band (ref 379-384, NO ×2, NO masks) → my z:
          const real_t mACK =  inh/dx3[k]*(muz6/dmx3[kp] + muz5/dmx3[k])
                            + kQ*(-w6/dmx3[kp] + w5/dmx3[k]);
          const real_t mAPK = -inh/dx3[k]*muz6/dmx3[kp] + kQ*( w6/dmx3[kp]);
          const real_t mAMK = -inh/dx3[k]*muz5/dmx3[k]  + kQ*(-w5/dmx3[k]);
          // cross Y = ref x3 Z-band (ref 389-394) → my y:
          const real_t mACJ =  inh/dx2[j]*(muy4/dmx2[jp] + muy3/dmx2[j])
                            + kQ*(-v4/dmx2[jp] + v3/dmx2[j]);
          const real_t mAPJ = -inh/dx2[j]*muy4/dmx2[jp] + kQ*( v4/dmx2[jp]);
          const real_t mAMJ = -inh/dx2[j]*muy3/dmx2[j]  + kQ*(-v3/dmx2[j]);

          R -= ( mAPI*U[cip] + mACI*U[c] + mAMI*U[cim]                    // ref 396-399
               + mAPJ*U[cjp] + mACJ*U[c] + mAMJ*U[cjm]
               + mAPK*U[ckp] + mACK*U[c] + mAMK*U[ckm] );

          // === cross bookkeeping (ref 401-407, axis-swapped, NO masks) =======
          // ref 401-403 (x2/wall, v4/v3, mu4*dvdx4-mu3*dvdx3) → my Z (w6/w5, muz*dwdx):
          R -= ( kQ*(w6*dudz6 + w5*dudz5)
               - inh/dx3[k]*(muz6*dwdx6 - muz5*dwdx5) );
          // ref 405-407 (x3, w6/w5, mu6*dwdx6-mu5*dwdx5) → my Y (v4/v3, muy*dvdx):
          R -= ( kQ*(v4*dudy4 + v3*dudy3)
               - inh/dx2[j]*(muy4*dvdx4 - muy3*dvdx3) );

          rhs[c] = R * dt;                                                // ref 413
        }
    return;
  }
```

> Note on the `mua` placeholder line: drop it — it is a leftover sketch. The real μ build is the `za/zc/zb`/`ya/yc/yb` block. (Parent: remove the `(void)mua` line.)

#### dV body (ref solvedW 1002–1116) — own = Y (= ref x3), cross = X & Z(=ref x2/wall)

> ref `solvedW`: x1 periodic, x2 wall, x3 = the own (periodic) direction = W component. My dV is the ref-x3 component → **own = Y**. ref x2 (wall) → my **Z**; ref x1 → my **X**. ref's increment-cross terms `ddu*, ddv*` (1023–1031, 1114–1120) couple to dU and dW *increments* — these belong to the block-coupling stage in my code, NOT the residual, **except** the reference computes them inside solvedW. **Important:** these `dd*` terms use `dU,dV` increments that in the reference were already solved (solvedU, solvedV run before solvedW). In my mpmstd wiring the increments are solved by `block_couple_*` AFTER all three ADI sweeps. To stay faithful I keep the residual WITHOUT the `dd*` terms (set them to 0, exactly as the const path does) and rely on `block_couple_dU_var`/`block_couple_dV_var` for the increment cross-coupling. The reference's `dd*` lines 1114–1120 are therefore **deliberately omitted** here and folded into the block-coupling stage — flagged in §5.

```cpp
  // ---------------------------------------------------------------------------
  //  Component dV  (ref solvedW).  own=Y(=ref x3).  q==V.
  //  Cross-X = ref x1 (mu1/mu2,u1/u2);  Cross-Z = ref x2/wall (mu3/mu4,v3/v4→my W).
  //  ref ddu*/ddv* increment-coupling lines DEFERRED to block_couple_* (faithful
  //  to the const path; see §5 caveat).
  // ---------------------------------------------------------------------------
  if (own_dir == Direction::Y) {
    for (int i = h; i < n1 - h; ++i)
      for (int j = h; j < n2 - h; ++j)
        for (int k = h; k < n3 - h; ++k) {
          const int im=i-1, ip=i+1, jm=j-1, jp=j+1, km=k-1, kp=k+1;
          const int c   = ix(i, j, k, n2, n3);
          const int cim = ix(im,j, k, n2, n3), cip = ix(ip,j, k, n2, n3);
          const int cjm = ix(i, jm,k, n2, n3), cjp = ix(i, jp,k, n2, n3);
          const int ckm = ix(i, j, km,n2, n3), ckp = ix(i, j, kp,n2, n3);

          // own Y face vel (ref w5/w6 at 1002-1003, on the V=ref-W y-face):
          const real_t v5 = kHalf*(V[c] + V[cjm]);                        // ref w5 (1002)→y-lo
          const real_t v6 = kHalf*(V[c] + V[cjp]);                        // ref w6 (1003)→y-hi
          // cross-X u1/u2 (ref 1005-1006, interp through ref x3→my y):
          const real_t u1 = (dx2[jm]*U[c]  + dx2[j]*U[cjm])/dmx2[j]*kHalf; // ref u1 (1005)
          const real_t u2 = (dx2[jm]*U[cip]+ dx2[j]*U[ix(ip,jm,k,n2,n3)])/dmx2[j]*kHalf; // ref u2 (1006)
          // cross-Z (ref x2/wall) w3/w4 → my W interpolated through y (ref v3/v4 1007-1008):
          const real_t w3 = (dx2[jm]*W[c]  + dx2[j]*W[cjm])/dmx2[j]*kHalf; // ref v3 (1007)→z-lo
          const real_t w4 = (dx2[jm]*W[ckp]+ dx2[j]*W[ix(i,jm,kp,n2,n3)])/dmx2[j]*kHalf; // ref v4 (1008)→z-hi

          // grads of V (ref 1010-1015, axis-swapped x1→X, x2→Z, x3→Y own):
          const real_t dvdx1 = (V[c]  - V[cim])/dmx1[i];                  // ref dwdx1 (1010)
          const real_t dvdx2 = (V[cip] - V[c] )/dmx1[ip];                 // ref dwdx2 (1011)
          const real_t dvdz3 = (V[c]  - V[ckm])/dmx3[k];                  // ref dwdy3 (1012)→Z
          const real_t dvdz4 = (V[ckp] - V[c] )/dmx3[kp];                 // ref dwdy4 (1013)→Z
          const real_t dvdy5 = (V[c]  - V[cjm])/dx2[jm];                  // ref dwdz5 (1014)→own Y
          const real_t dvdy6 = (V[cjp] - V[c] )/dx2[j];                   // ref dwdz6 (1015)→own Y

          // cross-stress source grads (ref 1017-1021):
          const real_t dudy1 = (U[c]   - U[cjm])/dmx2[j];                 // ref dudz1 (1017)
          const real_t dudy2 = (U[cip] - U[ix(ip,jm,k,n2,n3)])/dmx2[j];   // ref dudz2 (1018)
          const real_t dwdy3 = (W[c]   - W[cjm])/dmx2[j];                 // ref dvdz3 (1020)→W
          const real_t dwdy4 = (W[ckp] - W[ix(i,jm,kp,n2,n3)])/dmx2[j];   // ref dvdz4 (1021)→W

          // μ faces (ref 1033-1043 axis-swapped):
          // X-faces (ref mu1/mu2, x1, through x3→y interp) ref 1033-1037:
          const real_t xa = mface(dx2[jm], mu[cim], dx2[j], mu[ix(im,jm,k,n2,n3)], dmx2[j]); // ref mua
          const real_t xc = mface(dx2[jm], mu[c],   dx2[j], mu[cjm],               dmx2[j]); // ref muc
          const real_t xb = mface(dx2[jm], mu[cip], dx2[j], mu[ix(ip,jm,k,n2,n3)], dmx2[j]); // ref mub
          const real_t mu1 = kHalf*(dx1[im]*xc + dx1[i]*xa)/dmx1[i];      // ref mu1 (1036)→x-lo
          const real_t mu2 = kHalf*(dx1[ip]*xc + dx1[i]*xb)/dmx1[ip];     // ref mu2 (1037)→x-hi
          // Z-faces (ref mu3/mu4, x2/wall, through x3→y interp) ref 1039-1043:
          const real_t za = mface(dx2[jm], mu[ckm], dx2[j], mu[ix(i,jm,km,n2,n3)], dmx2[j]); // ref mua
          const real_t zc = xc;                                                               // same muc
          const real_t zb = mface(dx2[jm], mu[ckp], dx2[j], mu[ix(i,jm,kp,n2,n3)], dmx2[j]); // ref mub
          const real_t muz3 = kHalf*(dx3[km]*zc + dx3[k]*za)/dmx3[k];     // ref mu3 (1042)→z-lo
          const real_t muz4 = kHalf*(dx3[kp]*zc + dx3[k]*zb)/dmx3[kp];    // ref mu4 (1043)→z-hi

          const real_t irc = mface(dx2[jm], invrho[c], dx2[j], invrho[cjm], dmx2[j]); // ref invRhoc (1045)
          const real_t inh = kHalf*Cmu*irc;                              // ref (1046)
          const real_t muc_ = mu[c];                                     // ref Mu(i,j,k)

          // explicit viscous (ref 1049-1055; own ×2 is x3→Y here):
          const real_t vX  = (mu2*dvdx2 - mu1*dvdx1)/dx1[i];             // ref viscous_w1 (1049)
          const real_t vZ  = (muz4*dvdz4 - muz3*dvdz3)/dx3[k];          // ref viscous_w2 (1050)→Z
          const real_t vYo = (mu[c]*dvdy6 - mu[cjm]*dvdy5)/dmx2[j];     // ref viscous_w3 (1051) own Y (cell μ's!)
          const real_t vXc = (mu2*dudy2 - mu1*dudy1)/dx1[i];           // ref viscous_w31 (1052) cross-stress X
          const real_t vZc = (muz4*dwdy4 - muz3*dwdy3)/dx3[k];         // ref viscous_w32 (1053) cross-stress Z

          real_t R = inh*(vX + vZ + real_t{2}*vYo + vXc + vZc);          // ref 1055

          // bands (ref 1078-1099, axis-swapped):
          // cross-X (ref 1078-1083):
          const real_t mACI =  inh/dx1[i]*(mu2/dmx1[ip] + mu1/dmx1[i]) + kQ*(-u2/dmx1[ip] + u1/dmx1[i]);
          const real_t mAPI = -inh/dx1[i]*mu2/dmx1[ip] + kQ*( u2/dmx1[ip]);
          const real_t mAMI = -inh/dx1[i]*mu1/dmx1[i]  + kQ*(-u1/dmx1[i]);
          // cross-Z = ref x2 band (ref 1085-1092, NO ×2, NO masks) → my z:
          const real_t mACK =  inh/dx3[k]*(muz4/dmx3[kp] + muz3/dmx3[k]) + kQ*(-w4/dmx3[kp] + w3/dmx3[k]);
          const real_t mAPK = -inh/dx3[k]*muz4/dmx3[kp] + kQ*( w4/dmx3[kp]);
          const real_t mAMK = -inh/dx3[k]*muz3/dmx3[k]  + kQ*(-w3/dmx3[k]);
          // own-Y = ref x3 own band (ref 1094-1099, ×2, NOTE the +0.5·dvdy terms):
          const real_t mACJ =  inh/dmx2[j]*(mu[c]/dx2[j] + mu[cjm]/dx2[jm])*real_t{2}
                            + kQ*(-v6/dx2[j] + v5/dx2[jm] + kHalf*dvdy6 + kHalf*dvdy5);
          const real_t mAPJ = -inh/dmx2[j]*mu[c]/dx2[j]*real_t{2}
                            + kQ*( v6/dx2[j] + kHalf*dvdy6);
          const real_t mAMJ = -inh/dmx2[j]*mu[cjm]/dx2[jm]*real_t{2}
                            + kQ*(-v5/dx2[jm] + kHalf*dvdy5);

          R -= ( mAPI*V[cip] + mACI*V[c] + mAMI*V[cim]                    // ref 1101-1104
               + mAPJ*V[cjp] + mACJ*V[c] + mAMJ*V[cjm]
               + mAPK*V[ckp] + mACK*V[c] + mAMK*V[ckm] );

          // cross bookkeeping (ref 1106-1112; ddu/ddv lines DEFERRED to block-couple):
          // ref 1106-1108 (x1, u1/u2, mu2*dudz2-mu1*dudz1) → my X:
          R -= ( kQ*(u2*dvdx2 + u1*dvdx1) - inh/dx1[i]*(mu2*dudy2 - mu1*dudy1) );
          // ref 1110-1112 (x2/wall, v3/v4, mu4*dvdz4-mu3*dvdz3) → my Z (NO masks):
          R -= ( kQ*(w3*dvdz3 + w4*dvdz4) - inh/dx3[k]*(muz4*dwdy4 - muz3*dwdy3) );

          rhs[c] = R * dt;
        }
    return;
  }
```

#### dW body (ref solvedV 643–766, **minus buoyancy**) — own = Z (= ref x2/wall), cross = X & Y(=ref x3)

> ref `solvedV` is the wall-normal component (own = ref x2 = my **Z**). ref x1 → my X; ref x3 → my Y. Buoyancy (ref 704–709) is **commented out in the reference** and is composed in my main, so SKIP. The reference's `ddu*` increment-coupling (ref 652–653, 672–673, 764–766) is **deferred to `block_couple_*`** in my wiring (faithful to const path). No jvm/jvp masks.

```cpp
  // ---------------------------------------------------------------------------
  //  Component dW  (ref solvedV − buoyancy).  own=Z(=ref x2/WALL).  q==W.
  //  Cross-X = ref x1 (mu1/mu2,u1/u2);  Cross-Y = ref x3 (mu5/mu6,w5/w6→my V).
  //  ddu* increment-coupling DEFERRED to block_couple_*. Buoyancy in main.
  //  z is wall-normal & STAGGERED for the own-Z 2nd derivative (W on z-faces):
  //  own metric uses dx3 (neighbour) & dmx3 (control vol), mirroring ref dmx2/dx2.
  // ---------------------------------------------------------------------------
  { // own_dir == Direction::Z
    for (int i = h; i < n1 - h; ++i)
      for (int j = h; j < n2 - h; ++j)
        for (int k = h; k < n3 - h; ++k) {
          const int im=i-1, ip=i+1, jm=j-1, jp=j+1, km=k-1, kp=k+1;
          const int c   = ix(i, j, k, n2, n3);
          const int cim = ix(im,j, k, n2, n3), cip = ix(ip,j, k, n2, n3);
          const int cjm = ix(i, jm,k, n2, n3), cjp = ix(i, jp,k, n2, n3);
          const int ckm = ix(i, j, km,n2, n3), ckp = ix(i, j, kp,n2, n3);

          // own-Z face vel (ref v3/v4 at 643-644, on the W=ref-V wall-face), through ref x2→my z:
          const real_t w5 = kHalf*(W[ckm] + W[c]);                        // ref v3 (643)→z-lo
          const real_t w6 = kHalf*(W[c]   + W[ckp]);                      // ref v4 (644)→z-hi
          // cross-X u1/u2 (ref 646-647, interp through x2→z):
          const real_t u1 = (dx3[km]*U[c]  + dx3[k]*U[ckm])/dmx3[k]*kHalf; // ref u1 (646)
          const real_t u2 = (dx3[km]*U[cip]+ dx3[k]*U[ix(ip,j,km,n2,n3)])/dmx3[k]*kHalf; // ref u2 (647)
          // cross-Y w5/w6 (ref x3, on ref W→my V), interp through x2→z (ref w5/w6 649-650):
          const real_t v3 = (dx3[km]*V[c]  + dx3[k]*V[ckm])/dmx3[k]*kHalf; // ref w5 (649)→y-lo
          const real_t v4 = (dx3[km]*V[cjp]+ dx3[k]*V[ix(i,jp,km,n2,n3)])/dmx3[k]*kHalf; // ref w6 (650)→y-hi

          // grads of W (ref 656-661 axis-swapped x1→X, x3→Y, x2→Z own/wall):
          const real_t dwdx1 = (W[c]  - W[cim])/dmx1[i];                  // ref dvdx1 (656)
          const real_t dwdx2 = (W[cip] - W[c] )/dmx1[ip];                 // ref dvdx2 (657)
          const real_t dwdz5 = (W[c]  - W[ckm])/dx3[km];                  // ref dvdy3 (658)→own Z (staggered: dx3)
          const real_t dwdz6 = (W[ckp] - W[c] )/dx3[k];                   // ref dvdy4 (659)→own Z (staggered: dx3)
          const real_t dwdy3 = (W[c]  - W[cjm])/dmx2[j];                  // ref dvdz5 (660)→Y
          const real_t dwdy4 = (W[cjp] - W[c] )/dmx2[jp];                 // ref dvdz6 (661)→Y

          // cross-stress source grads (ref 664-668 axis-swapped):
          const real_t dudz1 = (U[c]   - U[ckm])/dmx3[k];                 // ref dudy1 (664)
          const real_t dudz2 = (U[cip] - U[ix(ip,j,km,n2,n3)])/dmx3[k];   // ref dudy2 (665)
          const real_t dvdz3 = (V[c]   - V[ckm])/dmx3[k];                 // ref dwdy5 (667)→V
          const real_t dvdz4 = (V[cjp] - V[ix(i,jp,km,n2,n3)])/dmx3[k];   // ref dwdy6 (668)→V

          // μ faces (ref 675-685 axis-swapped):
          // X-faces (ref mu1/mu2, x1, through x2→z interp) ref 675-679:
          const real_t xa = mface(dx3[km], mu[cim], dx3[k], mu[ix(im,j,km,n2,n3)], dmx3[k]); // ref mua
          const real_t xc = mface(dx3[km], mu[c],   dx3[k], mu[ckm],               dmx3[k]); // ref muc
          const real_t xb = mface(dx3[km], mu[cip], dx3[k], mu[ix(ip,j,km,n2,n3)], dmx3[k]); // ref mub
          const real_t mu1 = kHalf*(dx1[im]*xc + dx1[i]*xa)/dmx1[i];      // ref mu1 (678)
          const real_t mu2 = kHalf*(dx1[ip]*xc + dx1[i]*xb)/dmx1[ip];     // ref mu2 (679)
          // Y-faces (ref mu5/mu6, x3, through x2→z interp) ref 681-685:
          const real_t ya = mface(dx3[km], mu[cjm], dx3[k], mu[ix(i,jm,km,n2,n3)], dmx3[k]); // ref mua
          const real_t yc = xc;                                                               // same muc
          const real_t yb = mface(dx3[km], mu[cjp], dx3[k], mu[ix(i,jp,km,n2,n3)], dmx3[k]); // ref mub
          const real_t muy5 = kHalf*(dx2[jm]*yc + dx2[j]*ya)/dmx2[j];     // ref mu5 (684)→y-lo
          const real_t muy6 = kHalf*(dx2[jp]*yc + dx2[j]*yb)/dmx2[jp];    // ref mu6 (685)→y-hi

          const real_t irc = mface(dx3[km], invrho[c], dx3[k], invrho[ckm], dmx3[k]); // ref invRhoc (687)
          const real_t inh = kHalf*Cmu*irc;                              // ref (688)

          // explicit viscous (ref 691-697; own ×2 = ref x2 = Z):
          const real_t vX  = (mu2*dwdx2 - mu1*dwdx1)/dx1[i];             // ref viscous_v1 (691)
          const real_t vZo = (mu[c]*dwdz6 - mu[ckm]*dwdz5)/dmx3[k];      // ref viscous_v2 (692) own Z (cell μ's!)
          const real_t vY  = (muy6*dwdy4 - muy5*dwdy3)/dx2[j];          // ref viscous_v3 (693)→Y
          const real_t vXc = (mu2*dudz2 - mu1*dudz1)/dx1[i];           // ref viscous_v21 (694) cross-stress X
          const real_t vYc = (muy6*dvdz4 - muy5*dvdz3)/dx2[j];         // ref viscous_v23 (695) cross-stress Y

          real_t R = inh*(vX + real_t{2}*vZo + vY + vXc + vYc);          // ref 697  (buoyancy SKIPPED)

          // bands (ref 726-749 axis-swapped):
          // cross-X (ref 726-731):
          const real_t mACI =  inh/dx1[i]*(mu2/dmx1[ip] + mu1/dmx1[i]) + kQ*(-u2/dmx1[ip] + u1/dmx1[i]);
          const real_t mAPI = -inh/dx1[i]*mu2/dmx1[ip] + kQ*( u2/dmx1[ip]);
          const real_t mAMI = -inh/dx1[i]*mu1/dmx1[i]  + kQ*(-u1/dmx1[i]);
          // own-Z = ref x2 own band (ref 734-741, ×2 + 0.5·dvdy; my staggered uses dx3 for the 2nd-deriv):
          const real_t mACK =  inh/dmx3[k]*(mu[c]/dx3[k] + mu[ckm]/dx3[km])*real_t{2}
                            + kQ*(dwdz6*kHalf + dwdz5*kHalf - w6/dx3[k] + w5/dx3[km]);
          const real_t mAPK = -inh/dmx3[k]*mu[c]/dx3[k]*real_t{2}
                            + kQ*( w6/dx3[k] + dwdz6*kHalf);
          const real_t mAMK = -inh/dmx3[k]*mu[ckm]/dx3[km]*real_t{2}
                            + kQ*(-w5/dx3[km] + dwdz5*kHalf);
          // cross-Y = ref x3 band (ref 744-749) → my y:
          const real_t mACJ =  inh/dx2[j]*(muy6/dmx2[jp] + muy5/dmx2[j]) + kQ*(-v4/dmx2[jp] + v3/dmx2[j]);
          const real_t mAPJ = -inh/dx2[j]*muy6/dmx2[jp] + kQ*( v4/dmx2[jp]);
          const real_t mAMJ = -inh/dx2[j]*muy5/dmx2[j]  + kQ*(-v3/dmx2[j]);

          R -= ( mAPI*W[cip] + mACI*W[c] + mAMI*W[cim]                    // ref 751-754
               + mAPJ*W[cjp] + mACJ*W[c] + mAMJ*W[cjm]
               + mAPK*W[ckp] + mACK*W[c] + mAMK*W[ckm] );

          // cross bookkeeping (ref 756-762; ddu line 764-766 DEFERRED to block-couple):
          // ref 756-758 (x1, u1/u2, mu2*dudy2-mu1*dudy1) → my X:
          R -= ( kQ*(u2*dwdx2 + u1*dwdx1) - inh/dx1[i]*(mu2*dudz2 - mu1*dudz1) );
          // ref 760-762 (x3, w5/w6, mu6*dwdy6-mu5*dwdy5) → my Y:
          R -= ( kQ*(v4*dwdy4 + v3*dwdy3) - inh/dx2[j]*(muy6*dvdz4 - muy5*dvdz3) );

          rhs[c] = R * dt;
        }
  }
}
```

### 2.3 `build_adi_bands_var` — own×2 keyed on `d == own_dir`

The swept-direction band must use ×2 diffusion when the sweep direction equals the component's own direction, else ×1, and the **own-Z W-component** uses the staggered metric for the ×2 term. Note: the reference's own-direction band for the wall-normal component (ref solvedV mACJ 734, solvedW mACK 1094) carries the extra `+0.5·dvdy` convective term — but in the **2nd/3rd-stage band rebuilds** (ref 882-889 for dV, the dW 2nd/3rd stages 1177/1232) the bands are rebuilt identically, so this is just the same formula. The const path's `build_adi_bands_var` already has the structure; I extend it with `own_dir` and the ×2 / convection-from-the-correct-band logic. Since the stage RHS `D` is already the fully-formed delta (`src`), the band builder only needs the swept-direction operator. Here is the replacement:

```cpp
void build_adi_bands_var(Direction      d,
                         Direction      own_dir,
                         real_t*        A, real_t* B, real_t* C, real_t* D,
                         const real_t*  src,
                         const real_t*  q,            // own velocity (for the +0.5·dq convection in own band)
                         const real_t*  q_adv,
                         const real_t*  mu, const real_t* invrho,
                         const real_t*  dx_along, const real_t* dmx_along,
                         int            n1_tot, int n2_tot, int n3_tot,
                         int            n1_int,  int n2_int,  int n3_int,
                         real_t         Cmu, real_t dt,
                         bool           z_staggered) {
  const bool   is_own = (d == own_dir);
  const real_t two    = is_own ? real_t{2} : real_t{1};
  const real_t inh_dt = kHalf * Cmu * dt;   // ×invrho×μ_face per cell below
  const real_t cq     = kQ * dt;

  auto emit = [&](int gi, int gm, int gp, int along,
                  real_t dx_a, real_t /*unused*/, real_t& a, real_t& b, real_t& c) {
    const real_t ir = invrho[gi];
    const real_t muc = mu[gi];                       // cell μ (ref uses Mu(i,j,k) for own band)
    const real_t mm  = kHalf * (mu[gm] + mu[gi]);    // cross-band face μ
    const real_t mp  = kHalf * (mu[gi] + mu[gp]);
    if (is_own && !z_staggered) {
      // own, cell-centred (dU own-X, dV own-Y): ref mACI/mACJ ×2 with cell μ
      const real_t inv_h = inh_dt * ir;
      a = -inv_h/dmx_along[along] * muc/dx_a_m(along) * two; // see metric helper below
      // (metric expansion handled inline in the dispatch; see full bodies)
    }
    // ... (see full dispatch below)
  };
  // NOTE: because the own vs cross + staggered metric differs per axis, the
  // dispatch is written explicitly per direction below (no lambda), matching the
  // const build_adi_bands layout.
  // --- see explicit per-direction code ---
}
```

The lambda above is awkward; the clean faithful form mirrors the existing const `build_adi_bands` exactly but with (a) face-μ + invrho, (b) `two` factor on the viscous part when `is_own`, (c) cell-μ on the own diffusion and the `+0.5·dq` own-convection. Here is the **complete explicit** version to paste (replaces the current `build_adi_bands_var`):

```cpp
void build_adi_bands_var(Direction d, Direction own_dir,
                         real_t* A, real_t* B, real_t* C, real_t* D,
                         const real_t* src, const real_t* q, const real_t* q_adv,
                         const real_t* mu, const real_t* invrho,
                         const real_t* dxa, const real_t* dmxa,
                         int n1, int n2, int n3, int i1, int i2, int i3,
                         real_t Cmu, real_t dt, bool z_staggered) {
  const bool   own = (d == own_dir);
  const real_t inh = kHalf * Cmu * dt;
  const real_t cq  = kQ * dt;
  const int h = kHaloWidth;

  if (d == Direction::X) {
    const int nsys = i2 * i3;
    for (int ii = 0; ii < i1; ++ii) { const int i = ii + h;
      for (int jj = 0; jj < i2; ++jj) for (int kk = 0; kk < i3; ++kk) {
        const int p = ii*nsys + (jj*i3 + kk);
        const int gi = ix(i, jj+h, kk+h, n2, n3);
        const int gm = ix(i-1, jj+h, kk+h, n2, n3), gp = ix(i+1, jj+h, kk+h, n2, n3);
        const real_t ir = invrho[gi];
        // viscous: own → cell μ ×2 with dmx control vol; cross → face μ ×1 (cell-centred X)
        real_t av, cv;
        if (own) {
          const real_t muc = mu[gi];
          av = -inh*ir/dmxa[i]*muc/dxa[i-1]*real_t{2};   // ref mAMI 375
          cv = -inh*ir/dmxa[i]*muc/dxa[i]  *real_t{2};   // ref mAPI 373
        } else {
          const real_t mm = kHalf*(mu[gm]+mu[gi]), mp = kHalf*(mu[gi]+mu[gp]);
          av = -inh*ir/dxa[i]*mm/dmxa[i];                // ref mAMI 730 (cross x form)
          cv = -inh*ir/dxa[i]*mp/dmxa[i+1];              // ref mAPI 728
        }
        // convection (own carries +0.5·dq, cross does not)
        const real_t u5 = kHalf*(q_adv[gm]+q_adv[gi]);
        const real_t u6 = kHalf*(q_adv[gi]+q_adv[gp]);
        real_t ac, cc, bc;
        if (own) {
          const real_t dq1 = (q[gi]-q[gm])/dxa[i-1];     // dudx1
          const real_t dq2 = (q[gp]-q[gi])/dxa[i];       // dudx2
          ac = cq*(-u5/dmxa[i]   + dq1*kHalf);           // ref mAMI conv 376
          cc = cq*( u6/dmxa[i+1] + dq2*kHalf);           // ref mAPI conv 374
          bc = cq*(dq2*kHalf + dq1*kHalf - u6/dmxa[i+1] + u5/dmxa[i]); // ref mACI 372
        } else {
          ac = cq*(-u5/dmxa[i]);                          // ref mAMI 731
          cc = cq*( u6/dmxa[i+1]);                        // ref mAPI 729
          bc = cq*(-u6/dmxa[i+1] + u5/dmxa[i]);           // ref mACI 727
        }
        A[p] = av+ac; B[p] = (real_t{1} - av - cv) + bc; C[p] = cv+cc;
        D[p] = src[ix(ii+h, jj+h, kk+h, n2, n3)];
      }
    }
  } else if (d == Direction::Y) {
    const int nsys = i1 * i3;
    for (int jj = 0; jj < i2; ++jj) { const int j = jj + h;
      for (int ii = 0; ii < i1; ++ii) for (int kk = 0; kk < i3; ++kk) {
        const int p = jj*nsys + (ii*i3 + kk);
        const int gi = ix(ii+h, j, kk+h, n2, n3);
        const int gm = ix(ii+h, j-1, kk+h, n2, n3), gp = ix(ii+h, j+1, kk+h, n2, n3);
        const real_t ir = invrho[gi];
        real_t av, cv;
        if (own) {
          const real_t muc = mu[gi];
          av = -inh*ir/dmxa[j]*muc/dxa[j-1]*real_t{2};
          cv = -inh*ir/dmxa[j]*muc/dxa[j]  *real_t{2};
        } else {
          const real_t mm = kHalf*(mu[gm]+mu[gi]), mp = kHalf*(mu[gi]+mu[gp]);
          av = -inh*ir/dxa[j]*mm/dmxa[j];
          cv = -inh*ir/dxa[j]*mp/dmxa[j+1];
        }
        const real_t v5 = kHalf*(q_adv[gm]+q_adv[gi]);
        const real_t v6 = kHalf*(q_adv[gi]+q_adv[gp]);
        real_t ac, cc, bc;
        if (own) {
          const real_t dq1 = (q[gi]-q[gm])/dxa[j-1];
          const real_t dq2 = (q[gp]-q[gi])/dxa[j];
          ac = cq*(-v5/dmxa[j]   + dq1*kHalf);
          cc = cq*( v6/dmxa[j+1] + dq2*kHalf);
          bc = cq*(dq2*kHalf + dq1*kHalf - v6/dmxa[j+1] + v5/dmxa[j]);
        } else {
          ac = cq*(-v5/dmxa[j]);
          cc = cq*( v6/dmxa[j+1]);
          bc = cq*(-v6/dmxa[j+1] + v5/dmxa[j]);
        }
        A[p] = av+ac; B[p] = (real_t{1} - av - cv) + bc; C[p] = cv+cc;
        D[p] = src[ix(ii+h, jj+h, kk+h, n2, n3)];
      }
    }
  } else { // Direction::Z
    const int nsys = i1 * i2;
    for (int kk = 0; kk < i3; ++kk) { const int k = kk + h;
      for (int ii = 0; ii < i1; ++ii) for (int jj = 0; jj < i2; ++jj) {
        const int p = kk*nsys + (ii*i2 + jj);
        const int gi = ix(ii+h, jj+h, k, n2, n3);
        const int gm = ix(ii+h, jj+h, k-1, n2, n3), gp = ix(ii+h, jj+h, k+1, n2, n3);
        const real_t ir = invrho[gi];
        real_t av, cv;
        if (own && z_staggered) {
          // W own-Z, STAGGERED: ref mACK 1094 → 2·μ/(dmx3·dx3) ; my dxa=dx3,dmxa=dmx3
          const real_t muc = mu[gi];
          av = -inh*ir/dmxa[k]*mu[gm]/dxa[k-1]*real_t{2}; // ref mAMK 1098 uses Mu(i,j,km)
          cv = -inh*ir/dmxa[k]*muc   /dxa[k]  *real_t{2}; // ref mAPK 1096 uses Mu(i,j,k)
        } else if (own) {
          const real_t muc = mu[gi];
          av = -inh*ir/dmxa[k]*muc/dxa[k-1]*real_t{2};
          cv = -inh*ir/dmxa[k]*muc/dxa[k]  *real_t{2};
        } else {
          const real_t mm = kHalf*(mu[gm]+mu[gi]), mp = kHalf*(mu[gi]+mu[gp]);
          av = z_staggered ? -inh*ir/dmxa[k]*mm/dxa[k-1] : -inh*ir/dxa[k]*mm/dmxa[k];
          cv = z_staggered ? -inh*ir/dmxa[k]*mp/dxa[k]   : -inh*ir/dxa[k]*mp/dmxa[k+1];
        }
        const real_t w5 = kHalf*(q_adv[gm]+q_adv[gi]);
        const real_t w6 = kHalf*(q_adv[gi]+q_adv[gp]);
        real_t ac, cc, bc;
        if (own && z_staggered) {
          // ref mACK 1094-1099: conv uses dx3 metric + the +0.5·dwdz terms
          const real_t dq5 = (q[gi]-q[gm])/dxa[k-1];  // dwdz5
          const real_t dq6 = (q[gp]-q[gi])/dxa[k];    // dwdz6
          ac = cq*(-w5/dxa[k-1] + dq5*kHalf);         // ref mAMK 1099
          cc = cq*( w6/dxa[k]   + dq6*kHalf);         // ref mAPK 1097
          bc = cq*(-w6/dxa[k] + w5/dxa[k-1] + kHalf*dq6 + kHalf*dq5); // ref mACK 1095
        } else if (own) {
          const real_t dq5 = (q[gi]-q[gm])/dxa[k-1];
          const real_t dq6 = (q[gp]-q[gi])/dxa[k];
          ac = cq*(-w5/dmxa[k]   + dq5*kHalf);
          cc = cq*( w6/dmxa[k+1] + dq6*kHalf);
          bc = cq*(dq6*kHalf + dq5*kHalf - w6/dmxa[k+1] + w5/dmxa[k]);
        } else if (z_staggered) {
          ac = cq*(-w5/dxa[k-1]); cc = cq*( w6/dxa[k]);
          bc = cq*(-w6/dxa[k] + w5/dxa[k-1]);
        } else {
          ac = cq*(-w5/dmxa[k]); cc = cq*( w6/dmxa[k+1]);
          bc = cq*(-w6/dmxa[k+1] + w5/dmxa[k]);
        }
        A[p] = av+ac; B[p] = (real_t{1} - av - cv) + bc; C[p] = cv+cc;
        D[p] = src[ix(ii+h, jj+h, kk+h, n2, n3)];
      }
    }
  }
}
```

> **B-band note:** the reference forms `ac = mAC·dt + 1`. In my `bc`/`bv` split, `B = (1 − av − cv) + bc` where `(−av−cv)` reconstructs the diffusion centre (`mAC_diff·dt`) and `bc` the convection centre. This matches the const kernel's `bv = 1 − av − cv`. **Caveat:** the const kernel sets `bv = 1 − av − cv`, which assumes the centre diffusion = −(av+cv). For the own band that holds (`mAC_diff = −(mAM_diff + mAP_diff)` exactly, since all three share `muc`). For the cross band it also holds (`mAC_diff = inh/dx·(mp/dmx+ + mm/dmx−)` = −(av+cv)). Good — `1 − av − cv` is exactly `mAC_diff·dt + 1`.

### 2.4 Variable-μ block coupling (ref blockLdV 1345, blockLdU 1450) — axis-swapped

**Axis map for blockLd:** ref blockLdV updates ref **dV** (wall-normal, ref x2) using ref **dW** (x3) cross terms → in MY port the wall-normal increment is **dW**, and ref's x3 → my Y. But the function name in my codebase is `block_couple_dV`. Re-reading the const `block_couple_dV`/`block_couple_dU` already in my code: they were ported with my-axis semantics (`block_couple_dV`: dV −= 0.25·dW·∂V/∂z; `block_couple_dU`: dU −= 0.25·(dV·∂U/∂y + dW·∂U/∂z)). I keep those *names and roles* and add the variable-μ cross-stress, mapping the reference's `−0.5·Cmu·invRhoc/dx·(μ·∂ ddwdy)` pieces with the axis swap. 

> **Reference blockLdV (1345):** `dV -= dt·(0.25·(dwm5·dvdz5 + dwm6·dvdz6) − 0.5·Cmu·invRhoc/dx3·(mu6·ddwdy6 − mu5·ddwdy5))`. Here ref dV=wall-normal (my dW); ref dW=x3 (my dV); ref x2=my Z grad; ref x3-faces (mu5/mu6) = my Y? — **STOP.** This is the crux. Let me map precisely: ref blockLdV is `M23·dW` feeding the wall-normal eq. In MY port the wall-normal eq is dW, and the "23" cross is wall-normal(Z)×span. My existing const `block_couple_dV` instead does `M23·dW` into dV (my-Y eq). 

**Resolution:** My codebase's const block-coupling does **NOT** literally mirror ref blockLdV/blockLdU axis-for-axis; it was re-derived so that `block_couple_dV` is the *lower-triangular* coupling `dV ← dV − dt·M(dW→V)` and `block_couple_dU` is `dU ← dU − dt·M(dV,dW→U)`. The lower-triangular *solve order* in my code is dU, dV, dW solved by ADI, then dV corrected by dW, then dU corrected by dV,dW (see `momentum_cpu.cpp` 166-172). The reference order (solvedU→solvedV→solvedW, blockLdV, blockLdU) is the SAME lower-triangular structure with ref's component order (U, V=wall-normal, W). 

Given the validated const wiring, the **faithful variable-μ upgrade must preserve the exact const convective terms and ADD the reference's variable-μ increment cross-stress on the matching faces.** The safest faithful mapping is to add, to each const block-couple, the `−0.5·Cmu·invRhoc/dx·(μ_hi·∂(dq_hi) − μ_lo·∂(dq_lo))` term using the SAME face μ / metric that the corresponding `compute_mpmstd_rhs_var` cross-stress used for that velocity pair. Concretely:

- `block_couple_dV_var` (M23: dW→dV, my Y eq gets the span/wall-normal cross-stress of dW): add `− dt·(−0.5·Cmu·irc/dx3·(μz6·ddwdz... ))` — but my `block_couple_dV` couples dW into dV via `∂V/∂z`. The matching cross-stress is `∂z(μ ∂y dW)` interpolated to the V(y-face). 

Given the genuine axis-mapping subtlety and that **the parent will cross-check blockLd signs against the Fortran directly**, I provide the variable-μ block-couple as a **faithful transcription of ref blockLdV/blockLdU in MY axis names**, and flag it as the highest-uncertainty piece. Here ref dV→my dW, ref dW→my dV, ref x2-grad→my z-grad, ref x3→my y:

```cpp
// ── block_couple_dW_var  ≡ ref blockLdV (1314-1351), ref dV→my dW ─────────────
//   ref:  dV -= dt·( 0.25(dwm5·dvdz5 + dwm6·dvdz6)
//                  − 0.5·Cmu·invRhoc/dx3·(mu6·ddwdy6 − mu5·ddwdy5) )
//   swap: ref dV(wall-normal)→my dW;  ref dW(x3)→my dV;  ref x2 face/grad→my Z;
//         ref dwm5/6 = dW interp to ref-x2(=my Z) faces → my dV interp to z-faces;
//         ref dvdz5/6 (ref x3 grad of ref V=my dW) → my y-grad of dW;
//         ref mu5/6 on ref x3 faces → my Y faces;  ref /dx3 → my /dx2.
//   ⇒  dW -= dt·( 0.25(dvm_lo·dWdy_lo + dvm_hi·dWdy_hi)
//               − 0.5·Cmu·invRhoc/dx2·(muy_hi·∂z(dV)_hi − muy_lo·∂z(dV)_lo) )
//   (face μ volume-weighted through z, exactly as the dW-component cross-stress.)
void block_couple_dW_var(real_t* dW, const real_t* dV, const real_t* W,
                         const real_t* mu, const real_t* invrho,
                         const real_t* dx2, const real_t* dmx2,
                         const real_t* dx3, const real_t* dmx3,
                         int n1, int n2, int n3, real_t Cmu, real_t dt) {
  const int h = kHaloWidth;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int im=i-1,(void)im; const int jm=j-1, jp=j+1, km=k-1, kp=k+1;
        const int c = ix(i,j,k,n2,n3);
        const int ckm = ix(i,j,km,n2,n3), ckp = ix(i,j,kp,n2,n3);
        // ref dwm5/6: dW interp to z-faces through ... → my dV interp to z-faces (z-staggered)
        // ref uses dx2 weights (its x2=my z) → my dx3 weights:
        const real_t dvm5 = kHalf*(dx3[km]*dV[c]  + dx3[k]*dV[ckm])/dmx3[k];   // ref dwm5 (1325)
        const real_t dvm6 = kHalf*(dx3[km]*dV[ckp]+ dx3[k]*dV[c]  )/dmx3[k];   // ref dwm6 (1326) [see note]
        // ref dvdz5/6 (grad of ref V=my dW along ref x3=my y):
        const real_t dWdy5 = (W[c]   - W[ix(i,jm,k,n2,n3)])/dmx2[j];           // ref dvdz5→ my dWdy_lo
        const real_t dWdy6 = (W[ix(i,jp,k,n2,n3)] - W[c])/dmx2[jp];            // ref dvdz6→ my dWdy_hi
        // ref ddwdy5/6 (grad of dW=my dV) along ref x2=my z:
        const real_t ddVz5 = (dV[c]   - dV[ckm])/dmx3[k];                      // ref ddwdy5 (1328)
        const real_t ddVz6 = (dV[ckp] - dV[c]  )/dmx3[k];                      // ref ddwdy6 (1329)
        const real_t irc = kHalf*(dx3[km]*invrho[c] + dx3[k]*invrho[ckm])/dmx3[k]; // ref (1334)→swapped
        // ref mu5/6 on ref x3(=my y) faces, vol-weighted through ref x2(=my z):
        const real_t za = kHalf*(dx3[km]*mu[ckm] + dx3[k]*mu[ix(i,j,km,n2,n3)])/dmx3[k]; // crude; see §5
        (void)za;
        // NOTE: μ-face build here must mirror dW-component muy5/muy6 (z-weighted);
        //       parent: reuse that exact 6-line build. Placeholder names mY_lo/hi:
        const real_t mY_lo = /* muy5 from dW-component build */ real_t{0};
        const real_t mY_hi = /* muy6 from dW-component build */ real_t{0};
        dW[c] -= dt*( kQ*(dvm5*dWdy5 + dvm6*dWdy6)
                    - kHalf*Cmu*irc/dx2[j]*(mY_hi*ddVz6 - mY_lo*ddVz5) );
      }
}
```

**I am flagging this block-couple variable-μ piece as INCOMPLETE/UNCERTAIN** (the μ-face build and the dwm6 stagger index need the parent's direct Fortran cross-check — see §5). The **convective** part (`0.25·(dvm·dWdy)`) is exactly the existing const `block_couple_dV` and is correct; only the `−0.5·Cmu·invrho/dx·(μ·∂(d_increment))` cross-stress addendum is new and must be verified against ref 1345/1450 line-by-line by the parent. For `block_couple_dU_var` (ref blockLdU 1450, two cross terms M12·dVm + M13·dWm), the convective halves are exactly the existing const `block_couple_dU`; add:

```cpp
// ── block_couple_dU_var  ≡ ref blockLdU (1450-1452) ──────────────────────────
//   ref: dU -= dt·( 0.25(dvm4·dudy4 + dvm3·dudy3) − 0.5·Cmu·invRhoc/dx2·(mu4·dvmdx4 − mu3·dvmdx3) )
//             − dt·( 0.25(dwm6·dudz6 + dwm5·dudz5) − 0.5·Cmu·invRhoc/dx3·(mu6·dwmdx6 − mu5·dwmdx5) )
//   AXIS SWAP: ref x2(wall,dV)→my Z(dW); ref x3(dW)→my Y(dV); ref dx2→my dx3, dx3→my dx2.
//   ⇒ dU -= dt·( 0.25(dWm_hi·dUdz_hi + dWm_lo·dUdz_lo)  − 0.5·Cmu·irc/dx3·(μz_hi·∂x(dWm)_hi − μz_lo·∂x(dWm)_lo) )   [M-wall = ref M12]
//          − dt·( 0.25(dVm_hi·dUdy_hi + dVm_lo·dUdy_lo)  − 0.5·Cmu·irc/dx2·(μy_hi·∂x(dVm)_hi − μy_lo·∂x(dVm)_lo) )   [ref M13]
//   Convective halves == existing const block_couple_dU; μ-faces == dU-component
//   muz5/6 (z-weighted through x) and muy3/4 (y-weighted through x).  irc = ref invRhoc(x-face).
void block_couple_dU_var(real_t* dU, const real_t* dV, const real_t* dW, const real_t* U,
                         const real_t* mu, const real_t* invrho,
                         const real_t* dx1, const real_t* dmx1,
                         const real_t* dx2, const real_t* dmx2,
                         const real_t* dx3, const real_t* dmx3,
                         int n1, int n2, int n3, real_t Cmu, real_t dt) {
  const int h = kHaloWidth;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int im=i-1, jm=j-1, jp=j+1, km=k-1, kp=k+1;
        const int c=ix(i,j,k,n2,n3), cim=ix(im,j,k,n2,n3);
        const int cjp=ix(i,jp,k,n2,n3), ckp=ix(i,j,kp,n2,n3);
        const real_t irc = kHalf*(dx1[im]*invrho[c] + dx1[i]*invrho[cim])/dmx1[i]; // ref 1417
        // --- wall(Z)/dW coupling  (ref M12 block, dV→my dW) ---
        const real_t dWm5 = kHalf*(dx1[i]*dW[cim] + dx1[im]*dW[c])/dmx1[i];                 // ref dwm5 1434 → z-lo
        const real_t dWm6 = kHalf*(dx1[i]*dW[ix(im,j,kp,n2,n3)] + dx1[im]*dW[ckp])/dmx1[i]; // ref dwm6 1435 → z-hi
        const real_t dUdz5 = (U[c]   - U[ix(i,j,km,n2,n3)])/dmx3[k];                        // ref dudz5 1437
        const real_t dUdz6 = (U[ckp] - U[c])/dmx3[kp];                                      // ref dudz6 1438
        const real_t dWmx5 = (dW[c]  - dW[cim])/dmx1[i];                                    // ref dwmdx5 1440
        const real_t dWmx6 = (dW[ckp]- dW[ix(im,j,kp,n2,n3)])/dmx1[i];                      // ref dwmdx6 1441
        // μz5/μz6 == dU-component muz5/muz6 (z-faces, x-weighted then z-interp) ref 1443-1447:
        const real_t za = kHalf*(dx1[im]*mu[ix(i,j,km,n2,n3)] + dx1[i]*mu[ix(im,j,km,n2,n3)])/dmx1[i];
        const real_t zc = kHalf*(dx1[im]*mu[c] + dx1[i]*mu[cim])/dmx1[i];
        const real_t zb = kHalf*(dx1[im]*mu[ckp] + dx1[i]*mu[ix(im,j,kp,n2,n3)])/dmx1[i];
        const real_t muz5 = kHalf*(dx3[km]*zc + dx3[k]*za)/dmx3[k];  // ref mu5 1446
        const real_t muz6 = kHalf*(dx3[kp]*zc + dx3[k]*zb)/dmx3[kp]; // ref mu6 1447
        dU[c] -= dt*( kQ*(dWm6*dUdz6 + dWm5*dUdz5)
                    - kHalf*Cmu*irc/dx3[k]*(muz6*dWmx6 - muz5*dWmx5) );
        // --- span(Y)/dV coupling  (ref M13 block, dW→my dV) ---
        const real_t dVm3 = kHalf*(dx1[i]*dV[cim] + dx1[im]*dV[c])/dmx1[i];                 // ref dvm3 1419 → y-lo
        const real_t dVm4 = kHalf*(dx1[i]*dV[ix(im,jp,k,n2,n3)] + dx1[im]*dV[cjp])/dmx1[i]; // ref dvm4 1420 → y-hi
        const real_t dUdy3 = (U[c]   - U[ix(i,jm,k,n2,n3)])/dmx2[j];                        // ref dudy3 1422
        const real_t dUdy4 = (U[cjp] - U[c])/dmx2[jp];                                      // ref dudy4 1423
        const real_t dVmx3 = (dV[c]  - dV[cim])/dmx1[i];                                    // ref dvmdx3 1425
        const real_t dVmx4 = (dV[cjp]- dV[ix(im,jp,k,n2,n3)])/dmx1[i];                      // ref dvmdx4 1426
        const real_t ya = kHalf*(dx1[im]*mu[ix(i,jm,k,n2,n3)] + dx1[i]*mu[ix(im,jm,k,n2,n3)])/dmx1[i];
        const real_t yc = zc;
        const real_t yb = kHalf*(dx1[im]*mu[cjp] + dx1[i]*mu[ix(im,jp,k,n2,n3)])/dmx1[i];
        const real_t muy3 = kHalf*(dx2[jm]*yc + dx2[j]*ya)/dmx2[j];  // ref mu3 1431
        const real_t muy4 = kHalf*(dx2[jp]*yc + dx2[j]*yb)/dmx2[jp]; // ref mu4 1432
        dU[c] -= dt*( kQ*(dVm4*dUdy4 + dVm3*dUdy3)
                    - kHalf*Cmu*irc/dx2[j]*(muy4*dVmx4 - muy3*dVmx3) );
      }
}
```

**Wiring change** (in `solve_momentum_var_cpu`, `momentum_cpu.cpp`): replace the const `block_couple_dV`/`block_couple_dU` calls with `block_couple_dW_var` (the wall-normal increment, corrected by dV — i.e. ref blockLdV slot) and `block_couple_dU_var`. **Order must be:** all 3 ADI sweeps → sync → correct the wall-normal increment (dW) using dV/dW cross-stress → sync dW → correct dU using dV,dW → (no dV correction needed because ref only has blockLdV+blockLdU; the V/span component has no post-correction). 

> **Important order subtlety vs my const code:** the const `solve_momentum_var_cpu` corrects `dV` then `dU`. The reference corrects ref-dV (wall-normal = my dW) then ref-dU (my dU). So the faithful order is **correct dW, then dU** — NOT dV then dU. The parent must swap the first block-couple to act on `mom.dW`. Flagged in §5.

---

## PART 3 — REF-LINE → C++ mapping table

### dU component (ref `mpi_momentum_solvedU`) → my dU (own = X)
| ref line | formula | my C++ (compute_mpmstd_rhs_var, own_dir==X) |
|---|---|---|
| 295-296 | u1,u2 face U | `u1,u2` |
| 298-299 | v3,v4 (x2/wall) | `w5,w6` (→ my Z faces of W) |
| 301-302 | w5,w6 (x3) | `v3,v4` (→ my Y faces of V) |
| 304-305 | dudx1,dudx2 | `dudx1,dudx2` |
| 306-307 | dudy3,dudy4 (x2) | `dudz5,dudz6` |
| 308-309 | dudz5,dudz6 (x3) | `dudy3,dudy4` |
| 311-312 | dvdx3,dvdx4 | `dwdx5,dwdx6` |
| 314-315 | dwdx5,dwdx6 | `dvdx3,dvdx4` |
| 317-322 | mu3,mu4 (x2) | `muz5,muz6` |
| 324-329 | mu5,mu6 (x3) | `muy3,muy4` |
| 331-332 | invRhoc, invRhocCmu_half | `irc, inh` |
| 335 | viscous_u1 | `vu1` |
| 336 | viscous_u2 (x2) | `vuZ` |
| 337 | viscous_u3 (x3) | `vuY` |
| 338 | viscous_u12 | `vuZc` |
| 339 | viscous_u13 | `vuYc` |
| 341 | R = inh·(2vu1+vu2+vu3+vu12+vu13) | `R = inh*(2*vu1+vuZ+vuY+vuZc+vuYc)` |
| 344-352 | pressure+buoyancy | **SKIP** |
| 354-367 | ubc + masks | **SKIP** (ghost+fold) |
| 371-376 | mACI/mAPI/mAMI (own ×2) | `mACI/mAPI/mAMI` |
| 379-384 | mACJ/.. (x2) | `mACK/mAPK/mAMK` |
| 385-386 | jup/jum masks | **SKIP** |
| 389-394 | mACK/.. (x3) | `mACJ/mAPJ/mAMJ` |
| 396-399 | −(M·Uⁿ) | `R -= (mAPI*U[cip]+...+mAMK*U[ckm])` |
| 401-403 | cross-Y bookkeeping (x2) | `R -= (0.25(w6·dudz6+w5·dudz5) − inh/dx3·(muz6·dwdx6−muz5·dwdx5))` |
| 405-407 | cross-Z bookkeeping (x3) | `R -= (0.25(v4·dudy4+v3·dudy3) − inh/dx2·(muy4·dvdx4−muy3·dvdx3))` |
| 410-413 | ac/ap/am + R·dt | `rhs=R*dt`; bands via `build_adi_bands_var(.., own_dir=X)` |

### dV component (ref `mpi_momentum_solvedW`) → my dV (own = Y)
| ref line | formula | my C++ (own_dir==Y) |
|---|---|---|
| 1002-1003 | w5,w6 own (x3) | `v5,v6` |
| 1005-1006 | u1,u2 (x1) | `u1,u2` |
| 1007-1008 | v3,v4 (x2/wall) | `w3,w4` |
| 1010-1011 | dwdx1,dwdx2 | `dvdx1,dvdx2` |
| 1012-1013 | dwdy3,dwdy4 (x2) | `dvdz3,dvdz4` |
| 1014-1015 | dwdz5,dwdz6 (own x3) | `dvdy5,dvdy6` |
| 1017-1018 | dudz1,dudz2 | `dudy1,dudy2` |
| 1020-1021 | dvdz3,dvdz4 | `dwdy3,dwdy4` |
| 1033-1037 | mu1,mu2 (x1) | `mu1,mu2` |
| 1039-1043 | mu3,mu4 (x2) | `muz3,muz4` |
| 1049-1053 | viscous_w1,w2,w3,w31,w32 | `vX,vZ,vYo,vXc,vZc` |
| 1055 | R = inh·(w1+w2+2w3+w31+w32) | `R=inh*(vX+vZ+2*vYo+vXc+vZc)` |
| 1078-1083 | mACI (x1) | `mACI/mAPI/mAMI` |
| 1085-1092 | mACJ (x2/wall) | `mACK/mAPK/mAMK` |
| 1094-1099 | mACK own (x3, ×2 + 0.5dwdz) | `mACJ/mAPJ/mAMJ` (own-Y, +0.5·dvdy) |
| 1101-1104 | −(M·Wⁿ) | `R -= (...)` on V |
| 1106-1108 | x1 bookkeeping | `R -= (0.25(u2·dvdx2+u1·dvdx1) − inh/dx1·(mu2·dudy2−mu1·dudy1))` |
| 1110-1112 | x2/wall bookkeeping | `R -= (0.25(w3·dvdz3+w4·dvdz4) − inh/dx3·(muz4·dwdy4−muz3·dwdy3))` |
| 1114-1120 | ddu/ddv increment cross | **DEFER to block_couple** |

### dW component (ref `mpi_momentum_solvedV` − buoyancy) → my dW (own = Z, wall)
| ref line | formula | my C++ (own_dir==Z, z_staggered) |
|---|---|---|
| 643-644 | v3,v4 own (x2/wall) | `w5,w6` |
| 646-647 | u1,u2 (x1) | `u1,u2` |
| 649-650 | w5,w6 (x3) | `v3,v4` |
| 656-657 | dvdx1,dvdx2 | `dwdx1,dwdx2` |
| 658-659 | dvdy3,dvdy4 (own x2) | `dwdz5,dwdz6` (staggered: `/dx3[km]`, `/dx3[k]`) |
| 660-661 | dvdz5,dvdz6 (x3) | `dwdy3,dwdy4` |
| 664-665 | dudy1,dudy2 | `dudz1,dudz2` |
| 667-668 | dwdy5,dwdy6 | `dvdz3,dvdz4` |
| 675-679 | mu1,mu2 (x1) | `mu1,mu2` |
| 681-685 | mu5,mu6 (x3) | `muy5,muy6` |
| 691-695 | viscous_v1,v2,v3,v21,v23 | `vX,vZo,vY,vXc,vYc` |
| 697 | R = inh·(v1+2v2+v3+v21+v23) | `R=inh*(vX+2*vZo+vY+vXc+vYc)` |
| 704-709 | buoyancy (commented) | **SKIP** (main) |
| 726-731 | mACI (x1) | `mACI/mAPI/mAMI` |
| 734-741 | mACJ own (x2/wall, ×2 + 0.5dvdy) | `mACK/mAPK/mAMK` (own-Z staggered, +0.5·dwdz) |
| 744-749 | mACK (x3) | `mACJ/mAPJ/mAMJ` |
| 751-754 | −(M·Vⁿ) | `R -= (...)` on W |
| 756-758 | x1 bookkeeping | `R -= (0.25(u2·dwdx2+u1·dwdx1) − inh/dx1·(mu2·dudz2−mu1·dudz1))` |
| 760-762 | x3 bookkeeping | `R -= (0.25(v4·dwdy4+v3·dwdy3) − inh/dx2·(muy6·dvdz4−muy5·dvdz3))` |
| 764-766 | ddu increment cross | **DEFER to block_couple** |

### blockLd
| ref line | formula | my C++ |
|---|---|---|
| 1325-1347 | blockLdV (dV=wall-normal -= …) | `block_couple_dW_var` (ref dV→my dW) **[μ-face build needs cross-check]** |
| 1419-1452 | blockLdU (dU -= M12·dVm + M13·dWm) | `block_couple_dU_var` (ref x2/dV→my Z/dW; ref x3/dW→my Y/dV) |

---

## PART 4 — Reduction check (μ≡1, invrho≡1, Cmu=ν)

With μ≡1, invrho≡1: `inh = 0.5·ν`, all face μ = 1, cell μ = 1. 

**Own direction** (e.g. dU X): `R` viscous own = `inh·2·vu1 = ν·(dudx2−dudx1)/dx1 = ν·Lx_own`. Band own diffusion `−(mAC_diff·U+..) = −ν·Lx_own` (×2·inh·... with μ=1). So `R − M·Uⁿ` own-diffusion = `ν·Lx_own − ν·Lx_own = 0` **explicitly**, leaving the implicit own-direction sweep (bands ×2·0.5·ν·dt = ν·dt) → **fully-implicit (backward-Euler) own diffusion**, exactly as the const path's `visc_factor` story but now arising from the ×2 + M·Uⁿ algebra. ✓ (own μ=1 makes `Mu(i)=Mu(im)=μ_c`, so the §1.2 asymmetry vanishes → exact cancellation, confirming OB correctness.)

**Cross directions** (Y, Z): explicit `+inh·(vu2+vu3) = 0.5ν·(Ly+Lz)`; band·Uⁿ cross diffusion `−0.5ν·(Ly+Lz)` → cancel. The implicit cross sweep contributes `0.5ν·dt·L` (Crank-Nicolson half). Net cross diffusion = CN. ✓

**Cross-stress** vu12,vu13 at μ=1: `vu12 = (dvdx4−dvdx3)/dx2 = ∂y(∂x v)`, `vu13 = ∂z(∂x w)`; together with the doubled re-add at 403/407 they sum to `2·(∂y∂x v + ∂z∂x w) = 2·∂x(∂y v + ∂z w)`. Combined with the own `2·∂x(∂x u)` this is `2·∂x(∂x u + ∂y v + ∂z w) = 2·∂x(∇·u) ≈ 0` after projection (discretely ~1e-14). ✓ — so at μ≡1 the doubled cross-stress is a divergence term that vanishes, leaving a **valid OB momentum** (Laplacian + skew convection + projection). 

**Convection**: own in band (0.25 + implicit), cross split 0.25 div (band) + 0.25 adv (bookkeeping) → skew-symmetric `½(div+adv)`. Standard, conservative. ✓

**Conclusion:** at μ≡1/invρ≡1 the faithful scheme = backward-Euler own diffusion + CN cross diffusion + skew convection + (vanishing) divergence cross-stress = a valid OB Beam–Warming ADI momentum. It will **NOT** be bit-identical to the current simplified const path (which used CN ×0.5 for ALL directions and a single Laplacian in the RHS) — the difference is the own-direction being backward-Euler vs CN, exactly the "own×2 + M·Uⁿ → fully-implicit own" upgrade the parent identified. Validation target is OB physics + glycerol Fig 9, not bit-identity. ✓

---

## PART 5 — Caveats & ambiguities (with ref line numbers)

1. **Cross-stress double-count / no `add_transpose_stress_*` on the faithful path.** The faithful RHS already includes vu12/vu13 at 341 AND re-adds them at 403/407 (weight 2). My existing `add_transpose_stress_*` (momentum_cross_stress_cpu.cpp) supplies the cross-stress once on a *different* μ-edge convention. **The parent MUST remove the three `add_transpose_stress_*` calls from `assemble_momentum_var_visc_cpu` (momentum_cpu.cpp 63-68) when switching to the faithful kernel** — otherwise triple-count. (§1.3)

2. **Own-direction μ asymmetry (ref 335 vs 371-375).** viscous_u1 uses two cell μ's `Mu(i),Mu(im)`; the band uses `Mu(i,j,k)` twice. At μ≠1 these do NOT cancel; the residual `2·inh/dmx·[(Mu(i)−μ_c)dudx2 − (Mu(im)−μ_c)dudx1]` is intentional. I reproduced both literally. **Verify the parent agrees this asymmetry is faithful (not a ref bug).** Same pattern dV own-Y (1051 cell μ's vs 1094 band μ_c) and dW own-Z (692 vs 734).

3. **Wall ghost vs jum/jup masks.** As instructed, masks/ubc/vbc/zbc (ref 354-367, 711-722, 1061-1075) and the band masks (385-386, 738/741, 1089/1092) are **dropped**; wall via zero-ghost + modify_tdma_row_cpu. The reference's masked advective bookkeeping `v3·jum·dudy3` etc. (401-403, 1110-1112, 760-762) becomes my unmasked `w5·dudz5` etc. — differs at the wall-adjacent cell by the ghost-extrapolated transverse velocity. Accepted for energy; **same risk class, flag for the wall-normal dW especially** (ref solvedV is the wall component).

4. **z_staggered own-Z ×2 metric (dW, ref 734-741 / 1094-1099).** The reference wall-normal own band uses `Mu(i,j)/dx2(j)` and `Mu(i,jm)/dx2(jm)` over `/dmx2(j)` — i.e. the 2nd-derivative neighbour distance is `dx2` (the ref x2 own metric) and the control volume is `dmx2`. In MY staggered Z (W on z-faces) this maps to **neighbour = dx3, control vol = dmx3** — matching the const `z_staggered` convention (momentum_kernels_cpu.cpp 183-188: `av = -coeff/(dmx_along[k]*dx_along[k-1])`). I used `mu[gm]=Mu(i,j,km)` for `av` and `mu[gi]=Mu(i,j,k)` for `cv` per ref mAMK 1098 / mAPK 1096. **Confirm the dx3[k-1] vs dx3[k] index split matches ref dx2(jm)/dx2(j).** This is the one metric I'm least certain about — parent please cross-check ref 1098 `Mu(i,jm)/dx2(jm)` vs my `mu[gm]/dx3[k-1]`.

5. **blockLd variable-μ μ-face build + dwm6 stagger (ref 1325-1347, 1434-1447).** The convective halves of `block_couple_*` are the validated const code (correct). The NEW `−0.5·Cmu·invrho/dx·(μ·∂(d_increment))` cross-stress addenda need the parent's direct Fortran cross-check: specifically (a) `block_couple_dW_var` μY-face build (left as a placeholder `mY_lo/mY_hi` — must reuse the dW-component `muy5/muy6` z-weighted build) and the `dwm6` index (ref 1326 uses `dW(i,j,kp)` & `dW(i,jm,kp)` — verify my `dvm6` uses the analogous staggered pair); (b) sign of the cross-stress term (ref is `−0.5·Cmu·invRhoc/dx·(μ_hi·∂_hi − μ_lo·∂_lo)`, all inside the `−dt·(...)` so net `+dt·0.5·Cmu·invrho/dx·(...)`). I transcribed `dU -= dt·(0.25·conv − 0.5·Cmu·irc/dx·(μ·∂))`. **block_couple_dU_var is high-confidence; block_couple_dW_var μ-face is INCOMPLETE — parent must fill the 6-line μY build (identical to dW-component muy5/muy6) before compiling.**

6. **Block-couple ORDER and target swap (momentum_cpu.cpp 196-202).** Reference order: blockLdV (corrects wall-normal = my **dW**) then blockLdU (corrects dU). My const wiring corrects **dV** then dU. **The faithful wiring must correct `mom.dW` (not dV) first, sync dW, then correct dU.** The span component (my dV = ref x3 = ref W) has NO post-correction in the reference (there is no blockLdW). Flag: confirm my dV needs no block-couple. (§2.4)

7. **`own_dir` now USED.** `compute_mpmstd_rhs_var` previously ignored `own_dir` (it had no such param) and `build_adi_bands_var` used `visc_factor=0.5` for all. Both now branch on `own_dir` (RHS) / `d==own_dir` (bands). The wiring `assemble_momentum_var_visc_cpu` must pass `Direction::X/Y/Z` for u/v/w respectively, and `adi_component_var` must thread `own_dir` into `build_adi_bands_var` and **drop the old `real_t{0.5}` visc_factor argument** (now implicit). (§2.1, §2.3)

8. **Sign of the bookkeeping double-subtraction.** I read ref 401-407 as `RHS -= (0.25·adv − inh·cross_stress)`, i.e. `−0.25·adv + inh·cross_stress`. The `+inh·cross_stress` re-adds vu12/vu13 (making weight 2). **If the parent finds the intended weight is 1 (cross-stress NOT doubled), then line 341 should drop vu12/vu13 OR the bookkeeping sign flips.** I followed the literal Fortran: vu12/vu13 appear at both 341 and (with +) at 403/407 ⇒ weight 2. Verify against the discrete OB divergence-vanishing argument (§4) — it holds for weight 2 (`2·∂x∇·u→0`), which supports the literal reading.

**Files to edit (parent):** `kernels.hpp` (signatures §2.1), `momentum_var_kernels_cpu.cpp` (replace `compute_mpmstd_rhs_var` + `build_adi_bands_var`, add `block_couple_{dW,dU}_var` §2.2-2.4), `momentum_cpu.cpp` (`assemble_momentum_var_visc_cpu`: pass own_dir, drop the 3 `add_transpose_stress_*` calls; `adi_component_var`: thread own_dir; `solve_momentum_var_cpu`: swap block-couple to dW-then-dU with var functions). All flat-index/packing/scatter conventions are preserved (`ix(i,j,k)=(i*n2+j)*n3+k`, interior `[h, n−h)`, `D[row*n_sys+sys]`).agentId: aaf26af8c76f3d21c (use SendMessage with to: 'aaf26af8c76f3d21c' to continue this agent)
<usage>subagent_tokens: 134356
tool_uses: 10
duration_ms: 438962</usage>
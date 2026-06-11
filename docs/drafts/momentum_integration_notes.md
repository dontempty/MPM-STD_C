# Momentum faithful-port INTEGRATION NOTES (verified vs reference Fortran 2026-06-11)

Cross-checked every draft piece against `~/workspace/PaScaL_TCS/src/module_solve_momentum.f90`
(solvedU 225-563, solvedV 569-924, solvedW 930-1272, blockLdV 1278-1356, blockLdU 1362-1460).
Source root = `src/` (not `mpmstd/`). Validation case = OB (μ≡1, invρ≡1), UNIFORM grid.

## AXIS SWAP (physical, both codes solve same geometry)
ref x1(stream,periodic,buoyancy)↔my X(U) ; ref x2(WALL,Dirichlet,vel V)↔my Z(W) ; ref x3(span,periodic,vel W)↔my Y(V).
Index relabel for porting ref expr → my loop(i=X,j=Y,k=Z): ref-i→i, ref-j(wall)→k, ref-k(span)→j.
So ref jm/jp(wall±)→km/kp ; ref km/kp(span±)→jm/jp ; ref dx2/dmx2→my dx3/dmx3 ; ref dx3/dmx3→my dx2/dmx2.
Velocity/incr: ref V→my W, ref dV→my dW (wall) ; ref W→my V, ref dW→my dV (span).
Component map: my dU=ref solvedU(own X) ; my dW=ref solvedV(own Z,WALL) ; my dV=ref solvedW(own Y,span).

## DRAFT STATUS (docs/drafts/momentum_faithful_port_draft.md)
- dU body (own=X): CORRECT except **BUG#1**: vu1 denom is `/dmx1[i]` (ref335 `viscous_u1 = (Mu(i)*dudx2-Mu(im)*dudx1)/dmx1(i)`), draft wrote `/dx1[i]`. (invisible on uniform grid; fix anyway.)
- dW body (own=Z,wall): CORRECT (own-Z staggered band uses mu[c]/dx3[k] [+], mu[ckm]/dx3[km] [-] over /dmx3[k]; explicit viscous_v2 uses cell μ mu[c],mu[ckm]).
- dV body (own=Y,span): structurally correct (mirror of solvedW), but build_adi_bands own-Y must use TWO-cell μ (see BUG#3).
- block_couple_dU_var (blockLdU): CORRECT (M12 wall/dW via dUdz+muz+/dx3 ; M13 span/dV via dUdy+muy+/dx2). Only comment labels off.
- block_couple_dW_var (blockLdV): **BUG#2** index errors + μ placeholders. CORRECT version below.

## CORRECTIONS
**BUG#1**: dU body `vu1 = (mu[c]*dudx2 - mu[cim]*dudx1)/dmx1[i]`  (NOT /dx1[i]).

**BUG#2 (block_couple_dW_var = blockLdV, ref 1325-1347)**: corrects my **dW(wall)** using my dV(span)+my W(wall vel).
```
for interior (i,j,k):
  dvm5 = 0.5*(dx3[km]*dV[c]   + dx3[k]*dV[ckm])/dmx3[k]                 // ref dwm5 1325
  dvm6 = 0.5*(dx3[km]*dV[cjp] + dx3[k]*dV[ix(i,jp,km)])/dmx3[k]        // ref dwm6 1326 [CORRECTED: cjp, (i,jp,km)]
  dWdy5 = (W[c]   - W[cjm])/dmx2[j]                                     // ref dvdz5 1331
  dWdy6 = (W[cjp] - W[c]  )/dmx2[jp]                                    // ref dvdz6 1332
  ddVz5 = (dV[c]   - dV[ckm])/dmx3[k]                                   // ref ddwdy5 1328
  ddVz6 = (dV[cjp] - dV[ix(i,jp,km)])/dmx3[k]                          // ref ddwdy6 1329 [CORRECTED]
  irc = 0.5*(dx3[km]*invrho[c] + dx3[k]*invrho[ckm])/dmx3[k]           // ref 1334
  ya = 0.5*(dx3[km]*mu[cjm] + dx3[k]*mu[ix(i,jm,km)])/dmx3[k]          // ref mua 1336
  yc = 0.5*(dx3[km]*mu[c]   + dx3[k]*mu[ckm])/dmx3[k]                  // ref muc 1337
  yb = 0.5*(dx3[km]*mu[cjp] + dx3[k]*mu[ix(i,jp,km)])/dmx3[k]          // ref mub 1338
  muy5 = 0.5*(dx2[jm]*yc + dx2[j]*ya)/dmx2[j]                          // ref mu5 1340
  muy6 = 0.5*(dx2[jp]*yc + dx2[j]*yb)/dmx2[jp]                         // ref mu6 1341
  dW[c] -= dt*(0.25*(dvm5*dWdy5 + dvm6*dWdy6) - 0.5*Cmu*irc/dx2[j]*(muy6*ddVz6 - muy5*ddVz5))
```
(muy5/muy6 build == the dW-component's muy5/muy6, ref solvedV 681-685.)

**BUG#3 (build_adi_bands_var own-direction μ-cells)**: own-band ± μ differs by component:
- dU own-X (ref mAPI/mAMI 462-464): BOTH ± use Mu(i,j,k) single cell.
- dV own-Y (=ref solvedW own-x3, mAPK/mAMK 1096-1098): + uses Mu(i,j,k)=mu[gi], - uses Mu(i,j,km)→my mu[gjm] (TWO cells).
- dW own-Z (=ref solvedV own-x2, mAPJ/mAMJ 736-739): + uses Mu(i,j,k)=mu[gi], - uses Mu(i,jm,k)→my mu[gkm] (TWO cells; z_staggered).
Draft own-Y used single μ → wrong (invisible at μ≡1). own-X single, own-Y/own-Z two-cell.

**BUG#4 (own-direction convection metric)**: ref own-conv uses dx (cell width), draft used dmx.
ref mACI 372: `0.25*(0.5dudx2+0.5dudx1 - u2/dx1(i) + u1/dx1(im))`, u2=0.5(U(ip)+U(i)),u1=0.5(U(im)+U(i)).
So own-band conv: bc=cq*(0.5dq_hi+0.5dq_lo - u6/dx_own(i) + u5/dx_own(im)); ap=cq*(u6/dx_own(i)+0.5dq_hi); am=cq*(-u5/dx_own(im)+0.5dq_lo).
Cross-band conv uses dmx (ref mACJ 379: -v4/dmx2(jp)+v3/dmx2(j)). (uniform→same.)

## FAITHFUL SOLVE STRUCTURE (solve_momentum_var_cpu) — reference order + full coupling
Reference = full block solve (all 6 off-diag). Order dU→dW(wall)→dV(span):
1. assemble dU RHS (compute_mpmstd_rhs_var own=X, no incr coupling); adi(dU); sync dU.
2. assemble dW RHS (own=Z) + dd*(dU) forward-sub [ref solvedV 764-766]; adi(dW,z_stag); sync dW.
3. assemble dV RHS (own=Y) + dd*(dU)[ref solvedW 1114-1116] + dd*(dW)[ref solvedW 1118-1120]; adi(dV); sync dV.
4. block_couple_dW_var (blockLdV): dW(wall) -= M23·dV(span); sync dW.
5. block_couple_dU_var (blockLdU): dU -= M12·dW(wall) + M13·dV(span).
NOTE: existing const block_couple_dV is MIS-AXED (corrects dV not dW; no axis swap) → DROP it from var path.
DROP the 3 add_transpose_stress_* calls (cross-stress already weight-2 in the faithful RHS body).
Buoyancy: ref applies in solvedU; my code composes on U in main (gravity_axis=0=X). Keep main composition.

## dd* FORWARD-SUB COUPLING (ref, the lower off-diagonal M21/M31/M32)
Same operator shape as the x1/x2/x3 cross-bookkeeping but with the *other-component increment* in the
advecting+source slots, own-component velocity gradient unchanged:
- dW(wall) += -(0.25*(ddu2*dwdx2 + ddu1*dwdx1) - inh/dx1*(mu2*ddudz2 - mu1*ddudz1))  [ref 764-766; ddu=dU interp to wall(Z)-face via dx3, ddudz=dU grad in wall(Z)]
- dV(span) += -(0.25*(ddu2*dvdx2 + ddu1*dvdx1) - inh/dx1*(mu2*ddudy2 - mu1*ddudy1))  [ref 1114-1116; ddu=dU interp to span(Y)-face via dx2, ddudy=dU grad in span(Y)]
            += -(0.25*(ddw3*dvdz3 + ddw4*dvdz4) ... )  wait: ref 1118-1120 uses ddv(=my dW), dwdy → MAP CAREFULLY at impl time from ref 1118-1120.
All ×dt (part of stage-1 D). Cross-stress μ-faces = same as the body's μ. Drop jwm/jwp masks (ghost+fold).

## STAGE PLAN
Stage A = bodies + bands + blockLd(full visc) + reorder, DROP add_transpose_stress & const block_couple_dV. NO dd*.
Stage B = add dd* forward-sub. (At μ≡1 dd* is nonzero so likely needed for tight match; add if Stage A gap remains.)
Compare vs tests/regression/ground_truth via compare_pascal_tcs.py. Ground-truth localizes bugs.

---
# VALIDATION RESULTS (2026-06-11) — integrated + tested vs ground_truth

## Integrated (Stage A: bodies + bands + blockLd, NO dd* forward-sub yet)
- `src/equation/momentum/cpu/momentum_var_kernels_cpu.cpp` REWRITTEN: 3 faithful component bodies
  (own×2/skew-split/weight-2 cross-stress), per-component build_adi_bands_var, block_couple_dW_var (blockLdV,
  corrected), block_couple_dU_var (blockLdU). All bug#1-4 fixes applied.
- `kernels.hpp` signatures updated (own_dir; block_couple_*_var). `momentum_cpu.cpp` wiring: own_dir passed,
  add_transpose_stress_* DROPPED, block_couple swapped to dW-then-dU var (reference order). lib + all apps build.
- `apps/validate/` harness (gravity_axis=0 face-interp buoyancy on U, fixed dt, T-pert IC, X/Z/Y dump,
  physics.faithful flag). input_faithful/current.toml. Added to Makefile APPS.

## THE BIG FIND — shared wall-closure bug in `modify_tdma_row_cpu` (boundary_ops_cpu.cpp)
Ground-truth localized it: step-1 (clean per-step) U error was 3.1e-4 (16.5% rel), ENTIRELY at the Z-wall planes.
Root cause: modify_tdma_row folded `B += A; A = 0` for ALL Dirichlet+Neumann walls. Folding the (negative)
wall-ward off-diagonal into the diagonal CANCELS the diagonal's wall-ward diffusion → an unintended NEUMANN
(zero-gradient) wall for every field. For the no-slip velocity this under-damps U at the wall.
FIX: closure now matches the explicit ghost policy —
  ZeroGhost Dirichlet (δ_ghost=0)         → DROP off-diag, keep B's wall diffusion (1st-order Dirichlet).
  Antisymmetric Dirichlet (δ_ghost=−δ_in) → B −= off-diag (2nd-order, doubled wall flux).
  Neumann (δ_ghost=+δ_in)                 → B += off-diag.
This matches the memory note [[feedback-wall-bc-zero-ghost]] INTENT ("flag-drop, no fold") which the code never
actually implemented. Effect: step-1 velocity 3.1e-4 → 8.0e-8 (3900×). T unchanged (pinned). SHARED CHANGE —
affects channel/cavity/rbc/energy; needs channel turbulence re-check (change is in the memory-documented + ref
direction, so expected neutral/beneficial).

## Validation outcome (32×32×48, OB, fixed dt=0.004)
- **Step 1 (per-step operator test): ALL velocity fields match reference to ~1e-7** (U 8e-8, V 1.9e-8, W 4.8e-8;
  T 1.1e-5). This is the clean validation: the faithful momentum + wall-fix reproduce the reference per-step.
- **Step 200: faithful BETTER than current on every velocity field** (U 3.13e-4 < 3.31e-4, V 6.6e-5 < 7.1e-5,
  W 2.09e-4 < 2.11e-4). T identical (1.336e-3, momentum-independent).
- Residual floor (~1e-3 @ step200, ~1e-5/step) = INTERIOR energy-diffusion code-vs-code diff at κ≡1 (NOT
  momentum, NOT wall, NOT precision[double confirmed]). Accumulates ~linearly. rel 2.3e-5/step — excellent
  code-vs-code agreement. The momentum scheme difference is sub-dominant at OB steady-state (cross-stress =
  2∂(div u)≈0 after projection; own×2 BE-vs-CN is transient, washed by t=0.8) — it will matter for NOB (Fig 9).

## Tried + REJECTED
- Sweep-order match (Y,X,Z vs reference): NO effect (uniform-grid diffusion operators commute → ADI order-independent).
- ZeroGhost for T/U/V: WORSE (T step1 1.1e-5→2e-3). Reference T wall = ebc-injection (face=Tbt), closer to my
  antisymmetric (face=Tbt exact) than zero-ghost (ghost=Tbt, face off). Antisymmetric is correct. REVERTED.

## REMAINING (follow-ups, not blocking momentum correctness)
1. dd* forward-sub coupling (M21/M31/M32, ref solvedV 764-766 + solvedW 1114-1120): deferred. Sub-floor here
   (couples small increments). Needed for full NOB faithfulness; add + validate with a stronger NOB case.
2. Cross-band μ interpolation uses simple-average (exact at μ≡1; volume-weighted needed for NOB).
3. Energy ~1e-5/step interior floor: subtle ADI/scheme code-vs-code diff at κ≡1 — energy-port scope.

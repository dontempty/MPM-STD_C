#include "equation/momentum/kernels.hpp"
#include "common/macros.hpp"

// =============================================================================
//  Variable-viscosity (NOB) FAITHFUL momentum kernels — line-by-line port of
//  PaScaL_TCS module_solve_momentum.f90 (solvedU/solvedV/solvedW + blockLdV/U).
//  Cross-checked against the reference Fortran 2026-06-11 (see
//  docs/drafts/momentum_integration_notes.md for the per-line map + corrections).
//
//  AXIS SWAP (ref → my):  x1(U,stream,periodic)→X(dU,own) ;
//                         x2(V,WALL-normal,Dirichlet)→Z(dW,own) ;
//                         x3(W,span,periodic)→Y(dV,own).   Wall lives ONLY on Z.
//  Component map:  my dU ≡ ref solvedU(own X) ;
//                  my dV ≡ ref solvedW(own Y, ref x3 span) ;
//                  my dW ≡ ref solvedV − buoyancy (own Z, ref x2 wall).
//  Index relabel (ref expr → my i=X,j=Y,k=Z): ref-i→i, ref-j(wall)→k, ref-k(span)→j.
//
//  Stage-1 delta-form residual R − M·qⁿ − cross-bookkeeping, ×dt. PRESSURE +
//  BUOYANCY skipped (composed in main). jum/jup masks + ubc skipped (wall via
//  zero-ghost + modify_tdma_row_cpu fold, like the energy port). invRhocCmu_half
//  = 0.5·Cmu·invRhoc(face). Own-dir diffusion ×2, cross ×1. The own-direction
//  μ-asymmetry (dU band uses Mu(i,j,k) twice while viscous_u1 uses two cell μ's;
//  dV/dW bands use two cells matching their explicit) is reproduced literally —
//  vanishes at μ≡1, faithful at μ≠1.
// =============================================================================

namespace mpmstd::equation::momentum::kernels {

namespace {
MPMSTD_HD int ix(int i, int j, int k, int n2, int n3) { return (i * n2 + j) * n3 + k; }
constexpr real_t kHalf = real_t{0.5};
constexpr real_t kQ    = real_t{0.25};
constexpr real_t kTwo  = real_t{2};

// volume-weighted 2-pt face interp of a cell field along an axis:
//   0.5·(dxA·fA + dxB·fB)/dmx  — the ref pattern 0.5*(dx(im)*F(i)+dx(i)*F(im))/dmx(i)
MPMSTD_HD real_t mface(real_t dxA, real_t fA, real_t dxB, real_t fB, real_t dmx) {
  return kHalf * (dxA * fA + dxB * fB) / dmx;
}
} // anonymous namespace

// =============================================================================
//  (1) Stage-1 delta-form residual  D = dt·(R − M·qⁿ − bookkeeping)
// =============================================================================
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
  //  Component dU  (ref solvedU 225-413).  own = X.  q == U.
  //  Cross-Z = ref x2(wall, v3/v4,mu3/mu4) ; Cross-Y = ref x3(span, w5/w6,mu5/mu6).
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

          // face advecting velocities at the U (x-face) location
          const real_t u1 = kHalf*(U[cim] + U[c]);                               // ref u1 295
          const real_t u2 = kHalf*(U[c]   + U[cip]);                             // ref u2 296
          const real_t w5 = mface(dx1[im], W[c],   dx1[i], W[cim],               dmx1[i]); // ref v3 298 → z-lo
          const real_t w6 = mface(dx1[im], W[ckp], dx1[i], W[ix(im,j,kp,n2,n3)], dmx1[i]); // ref v4 299 → z-hi
          const real_t v3 = mface(dx1[im], V[c],   dx1[i], V[cim],               dmx1[i]); // ref w5 301 → y-lo
          const real_t v4 = mface(dx1[im], V[cjp], dx1[i], V[ix(im,jp,k,n2,n3)], dmx1[i]); // ref w6 302 → y-hi

          // gradients of U
          const real_t dudx1 = (U[c]   - U[cim])/dx1[im];                        // ref 304
          const real_t dudx2 = (U[cip] - U[c]  )/dx1[i];                         // ref 305
          const real_t dudz5 = (U[c]   - U[ckm])/dmx3[k];                        // ref dudy3 306 → z-lo
          const real_t dudz6 = (U[ckp] - U[c]  )/dmx3[kp];                       // ref dudy4 307 → z-hi
          const real_t dudy3 = (U[c]   - U[cjm])/dmx2[j];                        // ref dudz5 308 → y-lo
          const real_t dudy4 = (U[cjp] - U[c]  )/dmx2[jp];                       // ref dudz6 309 → y-hi
          // cross-stress source grads
          const real_t dwdx5 = (W[c]   - W[cim])/dmx1[i];                        // ref dvdx3 311 → z-lo
          const real_t dwdx6 = (W[ckp] - W[ix(im,j,kp,n2,n3)])/dmx1[i];          // ref dvdx4 312 → z-hi
          const real_t dvdx3 = (V[c]   - V[cim])/dmx1[i];                        // ref dwdx5 314 → y-lo
          const real_t dvdx4 = (V[cjp] - V[ix(im,jp,k,n2,n3)])/dmx1[i];          // ref dwdx6 315 → y-hi

          // μ faces — z (ref mu3/mu4 317-322 wall) & y (ref mu5/mu6 324-329 span)
          const real_t za = mface(dx1[im], mu[ckm], dx1[i], mu[ix(im,j,km,n2,n3)], dmx1[i]);
          const real_t zc = mface(dx1[im], mu[c],   dx1[i], mu[cim],               dmx1[i]);
          const real_t zb = mface(dx1[im], mu[ckp], dx1[i], mu[ix(im,j,kp,n2,n3)], dmx1[i]);
          const real_t muz5 = kHalf*(dx3[km]*zc + dx3[k]*za)/dmx3[k];            // ref mu3 → z-lo
          const real_t muz6 = kHalf*(dx3[kp]*zc + dx3[k]*zb)/dmx3[kp];           // ref mu4 → z-hi
          const real_t ya = mface(dx1[im], mu[cjm], dx1[i], mu[ix(im,jm,k,n2,n3)], dmx1[i]);
          const real_t yb = mface(dx1[im], mu[cjp], dx1[i], mu[ix(im,jp,k,n2,n3)], dmx1[i]);
          const real_t muy3 = kHalf*(dx2[jm]*zc + dx2[j]*ya)/dmx2[j];            // ref mu5 → y-lo (yc==zc)
          const real_t muy4 = kHalf*(dx2[jp]*zc + dx2[j]*yb)/dmx2[jp];           // ref mu6 → y-hi

          const real_t irc = mface(dx1[im], invrho[c], dx1[i], invrho[cim], dmx1[i]); // ref 331
          const real_t inh = kHalf*Cmu*irc;                                     // ref 332

          // explicit viscous (ref 335-339; viscous_u1 denom is /dmx1(i))
          const real_t vu1  = (mu[c]*dudx2 - mu[cim]*dudx1)/dmx1[i];             // ref viscous_u1 335
          const real_t vuZ  = (muz6*dudz6 - muz5*dudz5)/dx3[k];                  // ref viscous_u2 336
          const real_t vuY  = (muy4*dudy4 - muy3*dudy3)/dx2[j];                  // ref viscous_u3 337
          const real_t vuZc = (muz6*dwdx6 - muz5*dwdx5)/dx3[k];                  // ref viscous_u12 338
          const real_t vuYc = (muy4*dvdx4 - muy3*dvdx3)/dx2[j];                  // ref viscous_u13 339
          real_t R = inh*(kTwo*vu1 + vuZ + vuY + vuZc + vuYc);                   // ref 341

          // bands for −(M·Uⁿ)
          // Report/34-class fix: own-X band used mu[c] (cell) on BOTH faces
          // (ported PaScaL bug, module_solve_momentum.f90:371,375 pre-fix) --
          // matches own-Y/own-Z below, which already differ mu[c] vs neighbor.
          const real_t mACI =  inh/dmx1[i]*(mu[c]/dx1[i] + mu[cim]/dx1[im])*kTwo
                            + kQ*(kHalf*dudx2 + kHalf*dudx1 - u2/dx1[i] + u1/dx1[im]);   // ref 371
          const real_t mAPI = -inh/dmx1[i]*mu[c]/dx1[i]*kTwo    + kQ*( u2/dx1[i]  + kHalf*dudx2); // ref 373
          const real_t mAMI = -inh/dmx1[i]*mu[cim]/dx1[im]*kTwo + kQ*(-u1/dx1[im] + kHalf*dudx1); // ref 375
          const real_t mACK =  inh/dx3[k]*(muz6/dmx3[kp] + muz5/dmx3[k]) + kQ*(-w6/dmx3[kp] + w5/dmx3[k]); // ref 379(wall)
          const real_t mAPK = -inh/dx3[k]*muz6/dmx3[kp] + kQ*( w6/dmx3[kp]);
          const real_t mAMK = -inh/dx3[k]*muz5/dmx3[k]  + kQ*(-w5/dmx3[k]);
          const real_t mACJ =  inh/dx2[j]*(muy4/dmx2[jp] + muy3/dmx2[j]) + kQ*(-v4/dmx2[jp] + v3/dmx2[j]); // ref 389(span)
          const real_t mAPJ = -inh/dx2[j]*muy4/dmx2[jp] + kQ*( v4/dmx2[jp]);
          const real_t mAMJ = -inh/dx2[j]*muy3/dmx2[j]  + kQ*(-v3/dmx2[j]);

          R -= ( mAPI*U[cip] + mACI*U[c] + mAMI*U[cim]                           // ref 396-399
               + mAPJ*U[cjp] + mACJ*U[c] + mAMJ*U[cjm]
               + mAPK*U[ckp] + mACK*U[c] + mAMK*U[ckm] );

          R -= ( kQ*(w6*dudz6 + w5*dudz5) - inh/dx3[k]*(muz6*dwdx6 - muz5*dwdx5) ); // ref 401-403 (wall)
          R -= ( kQ*(v4*dudy4 + v3*dudy3) - inh/dx2[j]*(muy4*dvdx4 - muy3*dvdx3) ); // ref 405-407 (span)

          rhs[c] = R * dt;                                                       // ref 413
        }
    return;
  }

  // ---------------------------------------------------------------------------
  //  Component dV  (ref solvedW 930-1112).  own = Y (ref x3 span).  q == V.
  //  Cross-X = ref x1(mu1/mu2,u1/u2) ; Cross-Z = ref x2(wall, mu3/mu4,v3/v4→my W).
  //  ref ddu*/ddv* increment-coupling (1114-1120) deferred to couple_*_into_rhs.
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

          const real_t v5 = kHalf*(V[c] + V[cjm]);                               // ref w5 1002 → y-lo (own)
          const real_t v6 = kHalf*(V[c] + V[cjp]);                               // ref w6 1003 → y-hi
          const real_t u1 = mface(dx2[jm], U[c],   dx2[j], U[cjm],               dmx2[j]); // ref u1 1005
          const real_t u2 = mface(dx2[jm], U[cip], dx2[j], U[ix(ip,jm,k,n2,n3)], dmx2[j]); // ref u2 1006
          const real_t w3 = mface(dx2[jm], W[c],   dx2[j], W[cjm],               dmx2[j]); // ref v3 1007 → z-lo
          const real_t w4 = mface(dx2[jm], W[ckp], dx2[j], W[ix(i,jm,kp,n2,n3)], dmx2[j]); // ref v4 1008 → z-hi

          const real_t dvdx1 = (V[c]   - V[cim])/dmx1[i];                        // ref dwdx1 1010
          const real_t dvdx2 = (V[cip] - V[c]  )/dmx1[ip];                       // ref dwdx2 1011
          const real_t dvdz3 = (V[c]   - V[ckm])/dmx3[k];                        // ref dwdy3 1012 → z
          const real_t dvdz4 = (V[ckp] - V[c]  )/dmx3[kp];                       // ref dwdy4 1013 → z
          const real_t dvdy5 = (V[c]   - V[cjm])/dx2[jm];                        // ref dwdz5 1014 → own Y
          const real_t dvdy6 = (V[cjp] - V[c]  )/dx2[j];                         // ref dwdz6 1015 → own Y
          const real_t dudy1 = (U[c]   - U[cjm])/dmx2[j];                        // ref dudz1 1017
          const real_t dudy2 = (U[cip] - U[ix(ip,jm,k,n2,n3)])/dmx2[j];          // ref dudz2 1018
          const real_t dwdy3 = (W[c]   - W[cjm])/dmx2[j];                        // ref dvdz3 1020 → W
          const real_t dwdy4 = (W[ckp] - W[ix(i,jm,kp,n2,n3)])/dmx2[j];          // ref dvdz4 1021 → W

          // μ faces — x (ref mu1/mu2 1033-1037) & z (ref mu3/mu4 1039-1043 wall)
          const real_t xa = mface(dx2[jm], mu[cim], dx2[j], mu[ix(im,jm,k,n2,n3)], dmx2[j]);
          const real_t xc = mface(dx2[jm], mu[c],   dx2[j], mu[cjm],               dmx2[j]);
          const real_t xb = mface(dx2[jm], mu[cip], dx2[j], mu[ix(ip,jm,k,n2,n3)], dmx2[j]);
          const real_t mu1 = kHalf*(dx1[im]*xc + dx1[i]*xa)/dmx1[i];             // ref mu1 1036 → x-lo
          const real_t mu2 = kHalf*(dx1[ip]*xc + dx1[i]*xb)/dmx1[ip];            // ref mu2 1037 → x-hi
          const real_t za = mface(dx2[jm], mu[ckm], dx2[j], mu[ix(i,jm,km,n2,n3)], dmx2[j]);
          const real_t zb = mface(dx2[jm], mu[ckp], dx2[j], mu[ix(i,jm,kp,n2,n3)], dmx2[j]);
          const real_t muz3 = kHalf*(dx3[km]*xc + dx3[k]*za)/dmx3[k];            // ref mu3 1042 → z-lo (zc==xc)
          const real_t muz4 = kHalf*(dx3[kp]*xc + dx3[k]*zb)/dmx3[kp];           // ref mu4 1043 → z-hi

          const real_t irc = mface(dx2[jm], invrho[c], dx2[j], invrho[cjm], dmx2[j]); // ref 1045
          const real_t inh = kHalf*Cmu*irc;                                     // ref 1046

          const real_t vX  = (mu2*dvdx2 - mu1*dvdx1)/dx1[i];                     // ref viscous_w1 1049
          const real_t vZ  = (muz4*dvdz4 - muz3*dvdz3)/dx3[k];                   // ref viscous_w2 1050 → Z
          const real_t vYo = (mu[c]*dvdy6 - mu[cjm]*dvdy5)/dmx2[j];              // ref viscous_w3 1051 own Y (cell μ's)
          const real_t vXc = (mu2*dudy2 - mu1*dudy1)/dx1[i];                     // ref viscous_w31 1052 cross-stress X
          const real_t vZc = (muz4*dwdy4 - muz3*dwdy3)/dx3[k];                   // ref viscous_w32 1053 cross-stress Z
          real_t R = inh*(vX + vZ + kTwo*vYo + vXc + vZc);                       // ref 1055

          const real_t mACI =  inh/dx1[i]*(mu2/dmx1[ip] + mu1/dmx1[i]) + kQ*(-u2/dmx1[ip] + u1/dmx1[i]); // ref 1078(x)
          const real_t mAPI = -inh/dx1[i]*mu2/dmx1[ip] + kQ*( u2/dmx1[ip]);
          const real_t mAMI = -inh/dx1[i]*mu1/dmx1[i]  + kQ*(-u1/dmx1[i]);
          const real_t mACK =  inh/dx3[k]*(muz4/dmx3[kp] + muz3/dmx3[k]) + kQ*(-w4/dmx3[kp] + w3/dmx3[k]); // ref 1085(wall)
          const real_t mAPK = -inh/dx3[k]*muz4/dmx3[kp] + kQ*( w4/dmx3[kp]);
          const real_t mAMK = -inh/dx3[k]*muz3/dmx3[k]  + kQ*(-w3/dmx3[k]);
          const real_t mACJ =  inh/dmx2[j]*(mu[c]/dx2[j] + mu[cjm]/dx2[jm])*kTwo
                            + kQ*(kHalf*dvdy6 + kHalf*dvdy5 - v6/dx2[j] + v5/dx2[jm]);   // ref 1094(own Y)
          const real_t mAPJ = -inh/dmx2[j]*mu[c]/dx2[j]*kTwo   + kQ*( v6/dx2[j]  + kHalf*dvdy6);
          const real_t mAMJ = -inh/dmx2[j]*mu[cjm]/dx2[jm]*kTwo+ kQ*(-v5/dx2[jm] + kHalf*dvdy5);

          R -= ( mAPI*V[cip] + mACI*V[c] + mAMI*V[cim]                           // ref 1101-1104
               + mAPJ*V[cjp] + mACJ*V[c] + mAMJ*V[cjm]
               + mAPK*V[ckp] + mACK*V[c] + mAMK*V[ckm] );

          R -= ( kQ*(u2*dvdx2 + u1*dvdx1) - inh/dx1[i]*(mu2*dudy2 - mu1*dudy1) ); // ref 1106-1108 (x)
          R -= ( kQ*(w3*dvdz3 + w4*dvdz4) - inh/dx3[k]*(muz4*dwdy4 - muz3*dwdy3) ); // ref 1110-1112 (wall)

          rhs[c] = R * dt;                                                       // ref 1126
        }
    return;
  }

  // ---------------------------------------------------------------------------
  //  Component dW  (ref solvedV 569-762, − buoyancy).  own = Z (ref x2 WALL).
  //  q == W.  Cross-X = ref x1 ; Cross-Y = ref x3(span, mu5/mu6,w5/w6→my V).
  //  Own-Z is staggered: own 2nd-deriv uses dx3 (neighbour) over dmx3 (ctrl vol).
  //  ref ddu* (764-766) deferred to couple_*_into_rhs.  Buoyancy composed in main.
  // ---------------------------------------------------------------------------
  {  // own_dir == Direction::Z
    for (int i = h; i < n1 - h; ++i)
      for (int j = h; j < n2 - h; ++j)
        for (int k = h; k < n3 - h; ++k) {
          const int im=i-1, ip=i+1, jm=j-1, jp=j+1, km=k-1, kp=k+1;
          const int c   = ix(i, j, k, n2, n3);
          const int cim = ix(im,j, k, n2, n3), cip = ix(ip,j, k, n2, n3);
          const int cjm = ix(i, jm,k, n2, n3), cjp = ix(i, jp,k, n2, n3);
          const int ckm = ix(i, j, km,n2, n3), ckp = ix(i, j, kp,n2, n3);

          const real_t w5 = kHalf*(W[ckm] + W[c]);                               // ref v3 643 → z-lo (own)
          const real_t w6 = kHalf*(W[c]   + W[ckp]);                             // ref v4 644 → z-hi
          const real_t u1 = mface(dx3[km], U[c],   dx3[k], U[ckm],               dmx3[k]); // ref u1 646
          const real_t u2 = mface(dx3[km], U[cip], dx3[k], U[ix(ip,j,km,n2,n3)], dmx3[k]); // ref u2 647
          const real_t v3 = mface(dx3[km], V[c],   dx3[k], V[ckm],               dmx3[k]); // ref w5 649 → y-lo
          const real_t v4 = mface(dx3[km], V[cjp], dx3[k], V[ix(i,jp,km,n2,n3)], dmx3[k]); // ref w6 650 → y-hi

          const real_t dwdx1 = (W[c]   - W[cim])/dmx1[i];                        // ref dvdx1 656
          const real_t dwdx2 = (W[cip] - W[c]  )/dmx1[ip];                       // ref dvdx2 657
          const real_t dwdz5 = (W[c]   - W[ckm])/dx3[km];                        // ref dvdy3 658 → own Z
          const real_t dwdz6 = (W[ckp] - W[c]  )/dx3[k];                         // ref dvdy4 659 → own Z
          const real_t dwdy3 = (W[c]   - W[cjm])/dmx2[j];                        // ref dvdz5 660 → Y
          const real_t dwdy4 = (W[cjp] - W[c]  )/dmx2[jp];                       // ref dvdz6 661 → Y
          const real_t dudz1 = (U[c]   - U[ckm])/dmx3[k];                        // ref dudy1 664
          const real_t dudz2 = (U[cip] - U[ix(ip,j,km,n2,n3)])/dmx3[k];          // ref dudy2 665
          const real_t dvdz3 = (V[c]   - V[ckm])/dmx3[k];                        // ref dwdy5 667 → V
          const real_t dvdz4 = (V[cjp] - V[ix(i,jp,km,n2,n3)])/dmx3[k];          // ref dwdy6 668 → V

          // μ faces — x (ref mu1/mu2 675-679) & y (ref mu5/mu6 681-685 span)
          const real_t xa = mface(dx3[km], mu[cim], dx3[k], mu[ix(im,j,km,n2,n3)], dmx3[k]);
          const real_t xc = mface(dx3[km], mu[c],   dx3[k], mu[ckm],               dmx3[k]);
          const real_t xb = mface(dx3[km], mu[cip], dx3[k], mu[ix(ip,j,km,n2,n3)], dmx3[k]);
          const real_t mu1 = kHalf*(dx1[im]*xc + dx1[i]*xa)/dmx1[i];             // ref mu1 678 → x-lo
          const real_t mu2 = kHalf*(dx1[ip]*xc + dx1[i]*xb)/dmx1[ip];            // ref mu2 679 → x-hi
          const real_t ya = mface(dx3[km], mu[cjm], dx3[k], mu[ix(i,jm,km,n2,n3)], dmx3[k]);
          const real_t yb = mface(dx3[km], mu[cjp], dx3[k], mu[ix(i,jp,km,n2,n3)], dmx3[k]);
          const real_t muy5 = kHalf*(dx2[jm]*xc + dx2[j]*ya)/dmx2[j];            // ref mu5 684 → y-lo (yc==xc)
          const real_t muy6 = kHalf*(dx2[jp]*xc + dx2[j]*yb)/dmx2[jp];           // ref mu6 685 → y-hi

          const real_t irc = mface(dx3[km], invrho[c], dx3[k], invrho[ckm], dmx3[k]); // ref 687
          const real_t inh = kHalf*Cmu*irc;                                     // ref 688

          const real_t vX  = (mu2*dwdx2 - mu1*dwdx1)/dx1[i];                     // ref viscous_v1 691
          const real_t vZo = (mu[c]*dwdz6 - mu[ckm]*dwdz5)/dmx3[k];              // ref viscous_v2 692 own Z (cell μ's)
          const real_t vY  = (muy6*dwdy4 - muy5*dwdy3)/dx2[j];                   // ref viscous_v3 693 → Y
          const real_t vXc = (mu2*dudz2 - mu1*dudz1)/dx1[i];                     // ref viscous_v21 694 cross-stress X
          const real_t vYc = (muy6*dvdz4 - muy5*dvdz3)/dx2[j];                   // ref viscous_v23 695 cross-stress Y
          real_t R = inh*(vX + kTwo*vZo + vY + vXc + vYc);                       // ref 697 (buoyancy SKIP)

          const real_t mACI =  inh/dx1[i]*(mu2/dmx1[ip] + mu1/dmx1[i]) + kQ*(-u2/dmx1[ip] + u1/dmx1[i]); // ref 726(x)
          const real_t mAPI = -inh/dx1[i]*mu2/dmx1[ip] + kQ*( u2/dmx1[ip]);
          const real_t mAMI = -inh/dx1[i]*mu1/dmx1[i]  + kQ*(-u1/dmx1[i]);
          const real_t mACK =  inh/dmx3[k]*(mu[c]/dx3[k] + mu[ckm]/dx3[km])*kTwo
                            + kQ*(kHalf*dwdz6 + kHalf*dwdz5 - w6/dx3[k] + w5/dx3[km]);   // ref 734(own Z)
          const real_t mAPK = -inh/dmx3[k]*mu[c]/dx3[k]*kTwo   + kQ*( w6/dx3[k]  + kHalf*dwdz6);
          const real_t mAMK = -inh/dmx3[k]*mu[ckm]/dx3[km]*kTwo+ kQ*(-w5/dx3[km] + kHalf*dwdz5);
          const real_t mACJ =  inh/dx2[j]*(muy6/dmx2[jp] + muy5/dmx2[j]) + kQ*(-v4/dmx2[jp] + v3/dmx2[j]); // ref 744(span)
          const real_t mAPJ = -inh/dx2[j]*muy6/dmx2[jp] + kQ*( v4/dmx2[jp]);
          const real_t mAMJ = -inh/dx2[j]*muy5/dmx2[j]  + kQ*(-v3/dmx2[j]);

          R -= ( mAPI*W[cip] + mACI*W[c] + mAMI*W[cim]                           // ref 751-754
               + mAPJ*W[cjp] + mACJ*W[c] + mAMJ*W[cjm]
               + mAPK*W[ckp] + mACK*W[c] + mAMK*W[ckm] );

          R -= ( kQ*(u2*dwdx2 + u1*dwdx1) - inh/dx1[i]*(mu2*dudz2 - mu1*dudz1) ); // ref 756-758 (x)
          R -= ( kQ*(v4*dwdy4 + v3*dwdy3) - inh/dx2[j]*(muy6*dvdz4 - muy5*dvdz3) ); // ref 760-762 (span)

          rhs[c] = R * dt;                                                       // ref 772
        }
  }
}

// =============================================================================
//  (2) FAITHFUL per-component ADI bands for sweep direction d
// =============================================================================
void build_adi_bands_var(Direction d, Direction own_dir,
                         real_t* A, real_t* B, real_t* C, real_t* D,
                         const real_t* src, const real_t* q, const real_t* q_adv,
                         const real_t* mu, const real_t* invrho,
                         const real_t* dxa, const real_t* dmxa,
                         const real_t* dxo, const real_t* dmxo,
                         int n1, int n2, int n3, int i1, int i2, int i3,
                         real_t Cmu, real_t dt, bool z_staggered) {
  const bool   own = (d == own_dir);
  const real_t inh = kHalf * Cmu * dt;   // × invrho × μ per cell
  const real_t cq  = kQ * dt;
  const int    h   = kHaloWidth;
  // own-direction flat stride: the component lives at the own-minus face, so the
  // reference interpolates cell fields THROUGH the own direction (own-minus
  // neighbour) onto that face before the sweep-direction blend.
  const int ostride = (own_dir == Direction::X) ? n2*n3 : (own_dir == Direction::Y) ? n3 : 1;

  // own-direction volume-weighted face interp of a cell field (ref invRhoc / the
  // first stage of mu1..mu6). b = own-axis index of the cell. =field at μ/ρ≡1.
  auto volw_o = [&](const real_t* f, int cell, int b) {
    return kHalf*(dxo[b-1]*f[cell] + dxo[b]*f[cell - ostride])/dmxo[b];
  };

  // emit the swept-direction tridiagonal triple (a,b,c) + packed D for cell (gi)
  // with minus/plus neighbours (gm,gp) along d at sweep index a, own index bo.
  auto emit = [&](int gi, int gm, int gp, int a, int bo, real_t& av, real_t& bv, real_t& cv) {
    const real_t ir = volw_o(invrho, gi, bo);            // own-face invρ (ref invRhoc); =1 at invρ≡1
    real_t ad, cd;          // diffusion
    if (own) {
      const real_t cmu = mu[gi];                         // + side cell μ (always)
      // Report/34-class fix: own-X previously fell through to mu[gi] here too
      // (ported PaScaL bug, module_solve_momentum.f90:371,375 pre-fix) -- the
      // own-minus face always uses the own-axis minus-neighbour's μ, matching
      // own-Y/own-Z (and the RHS's vu1/mACI-equivalent, already fixed above).
      const real_t mmu = mu[gm];
      ad = -inh*ir/dmxa[a]*mmu/dxa[a-1]*kTwo;
      cd = -inh*ir/dmxa[a]*cmu/dxa[a]  *kTwo;
    } else {
      // FAITHFUL doubly volume-weighted cross-face μ (ref mu1..mu6, e.g. 675-685):
      // interp μ THROUGH the own direction to the velocity face, THEN through the
      // sweep direction d. Matches the RHS faces (muz/muy/mu1/mu2) byte-for-byte;
      // a NO-OP at μ≡1 (dmx[a]=½(dx[a-1]+dx[a])). Replaces the old simple average,
      // which under-damped the implicit operator at strong μ-contrast (NOB).
      const real_t muc = volw_o(mu, gi, bo);
      const real_t mua = volw_o(mu, gm, bo);
      const real_t mub = volw_o(mu, gp, bo);
      const real_t mm = kHalf*(dxa[a-1]*muc + dxa[a]*mua)/dmxa[a];     // lo face (ref mu1/mu3/mu5)
      const real_t mp = kHalf*(dxa[a+1]*muc + dxa[a]*mub)/dmxa[a+1];   // hi face (ref mu2/mu4/mu6)
      if (z_staggered) { ad = -inh*ir/dmxa[a]*mm/dxa[a-1]; cd = -inh*ir/dmxa[a]*mp/dxa[a]; }
      else             { ad = -inh*ir/dxa[a]*mm/dmxa[a];   cd = -inh*ir/dxa[a]*mp/dmxa[a+1]; }
    }
    real_t q5, q6;
    real_t ac, cc, bc;
    if (own) {
      q5 = kHalf*(q_adv[gm]+q_adv[gi]);                   // own-dir: plain face mean
      q6 = kHalf*(q_adv[gi]+q_adv[gp]);
      const real_t dq5 = (q[gi]-q[gm])/dxa[a-1];          // ∂q lo
      const real_t dq6 = (q[gp]-q[gi])/dxa[a];            // ∂q hi
      ac = cq*(-q5/dxa[a-1] + kHalf*dq5);
      cc = cq*( q6/dxa[a]   + kHalf*dq6);
      bc = cq*( kHalf*dq6 + kHalf*dq5 - q6/dxa[a] + q5/dxa[a-1]);
    } else {
      // cross-dir: advecting vel VOLUME-WEIGHTED along the OWN axis (ref f90:301-302,
      // reused in the M bands f90:390/392/394; matches the RHS w5/w6/v3/v4). The old
      // plain swept-axis mean diverged from PaScaL and from the faithful −M·qⁿ residual.
      q5 = volw_o(q_adv, gi, bo);
      q6 = volw_o(q_adv, gp, bo);
      if (z_staggered) {
        ac = cq*(-q5/dxa[a-1]); cc = cq*( q6/dxa[a]);
        bc = cq*(-q6/dxa[a] + q5/dxa[a-1]);
      } else {
        ac = cq*(-q5/dmxa[a]); cc = cq*( q6/dmxa[a+1]);
        bc = cq*(-q6/dmxa[a+1] + q5/dmxa[a]);
      }
    }
    av = ad + ac;  cv = cd + cc;  bv = (real_t{1} - ad - cd) + bc;
  };

  if (d == Direction::X) {
    const int nsys = i2 * i3;
    for (int ii = 0; ii < i1; ++ii) { const int i = ii + h;
      for (int jj = 0; jj < i2; ++jj) for (int kk = 0; kk < i3; ++kk) {
        const int p = ii*nsys + (jj*i3 + kk);
        const int gi = ix(i, jj+h, kk+h, n2, n3);
        const int gm = ix(i-1, jj+h, kk+h, n2, n3), gp = ix(i+1, jj+h, kk+h, n2, n3);
        const int bo = (own_dir == Direction::X) ? i : (own_dir == Direction::Y) ? jj+h : kk+h;
        emit(gi, gm, gp, i, bo, A[p], B[p], C[p]);
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
        const int bo = (own_dir == Direction::X) ? ii+h : (own_dir == Direction::Y) ? j : kk+h;
        emit(gi, gm, gp, j, bo, A[p], B[p], C[p]);
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
        const int bo = (own_dir == Direction::X) ? ii+h : (own_dir == Direction::Y) ? jj+h : k;
        emit(gi, gm, gp, k, bo, A[p], B[p], C[p]);
        D[p] = src[ix(ii+h, jj+h, kk+h, n2, n3)];
      }
    }
  }
}

// =============================================================================
//  (3) FAITHFUL block coupling — blockLdV (→ my dV) and blockLdU (→ my dU)
//  MPM-STD_main arrangement (Report/RBC): the explicit post-hoc corrections go
//  to the HORIZONTAL increments only; the wall-normal dW receives none.
// =============================================================================

// ref MPM-STD blockLdV 1612-1688 (M23dWm at 1682): dV -= dt·M23·dW.
void block_couple_dV_var(real_t* dV, const real_t* dW, const real_t* V,
                         const real_t* mu, const real_t* invrho,
                         const real_t* dx2, const real_t* dmx2,
                         const real_t* dx3, const real_t* dmx3,
                         int n1, int n2, int n3, real_t Cmu, real_t dt) {
  const int h = kHaloWidth;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int jm=j-1, km=k-1, kp=k+1;
        const int c   = ix(i, j, k, n2, n3);
        const int cjm = ix(i, jm,k, n2, n3);
        const int ckm = ix(i, j, km,n2, n3), ckp = ix(i, j, kp,n2, n3);

        // dW interp to the V node at z-faces k/kp (ref dwm5/dwm6 1658-1659)
        const real_t ddw3   = mface(dx2[jm], dW[c],   dx2[j], dW[cjm],               dmx2[j]);
        const real_t ddw4   = mface(dx2[jm], dW[ckp], dx2[j], dW[ix(i,jm,kp,n2,n3)], dmx2[j]);
        // ∂y(dW) at z-faces k/kp (ref ddwdy5/6 1661-1662)
        const real_t ddwdy3 = (dW[c]   - dW[cjm])/dmx2[j];
        const real_t ddwdy4 = (dW[ckp] - dW[ix(i,jm,kp,n2,n3)])/dmx2[j];
        // ∂z(V) at z-faces k/kp (ref dvdz5/6 1664-1665)
        const real_t dvdz3  = (V[c]   - V[ckm])/dmx3[k];
        const real_t dvdz4  = (V[ckp] - V[c]  )/dmx3[kp];

        // ref mu5/mu6 1667-1671 (V-node μ interp through y, then z)
        const real_t xc = mface(dx2[jm], mu[c],   dx2[j], mu[cjm],               dmx2[j]);
        const real_t za = mface(dx2[jm], mu[ckm], dx2[j], mu[ix(i,jm,km,n2,n3)], dmx2[j]);
        const real_t zb = mface(dx2[jm], mu[ckp], dx2[j], mu[ix(i,jm,kp,n2,n3)], dmx2[j]);
        const real_t muz3 = kHalf*(dx3[km]*xc + dx3[k]*za)/dmx3[k];
        const real_t muz4 = kHalf*(dx3[kp]*xc + dx3[k]*zb)/dmx3[kp];
        const real_t irc = mface(dx2[jm], invrho[c], dx2[j], invrho[cjm], dmx2[j]); // ref 1673

        dV[c] -= dt*( kQ*(ddw3*dvdz3 + ddw4*dvdz4)
                    - kHalf*Cmu*irc/dx3[k]*(muz4*ddwdy4 - muz3*ddwdy3) );        // ref 1682-1683
      }
}

// ref blockLdU 1362-1460 : dU -= dt·(M12·dW(wall) + M13·dV(span)), full viscous.
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
        const int c   = ix(i, j, k, n2, n3), cim = ix(im,j,k, n2, n3);
        const int cjp = ix(i, jp,k, n2, n3), ckp = ix(i, j,kp,n2, n3);
        const real_t irc = mface(dx1[im], invrho[c], dx1[i], invrho[cim], dmx1[i]); // ref 1417

        // --- M12 : wall increment (my dW) feeds dU via ∂U/∂z (ref dvm/dudy/dvmdx) ---
        const real_t dWm5 = mface(dx1[i], dW[cim], dx1[im], dW[c],                dmx1[i]); // ref dvm3 1419 → z-lo
        const real_t dWm6 = mface(dx1[i], dW[ix(im,j,kp,n2,n3)], dx1[im], dW[ckp], dmx1[i]); // ref dvm4 1420 → z-hi
        const real_t dUdz5 = (U[c]   - U[ix(i,j,km,n2,n3)])/dmx3[k];             // ref dudy3 1422
        const real_t dUdz6 = (U[ckp] - U[c]  )/dmx3[kp];                         // ref dudy4 1423
        const real_t dWmx5 = (dW[c]  - dW[cim])/dmx1[i];                         // ref dvmdx3 1425
        const real_t dWmx6 = (dW[ckp]- dW[ix(im,j,kp,n2,n3)])/dmx1[i];           // ref dvmdx4 1426
        const real_t za = mface(dx1[im], mu[ix(i,j,km,n2,n3)], dx1[i], mu[ix(im,j,km,n2,n3)], dmx1[i]);
        const real_t zc = mface(dx1[im], mu[c],   dx1[i], mu[cim],               dmx1[i]);
        const real_t zb = mface(dx1[im], mu[ckp], dx1[i], mu[ix(im,j,kp,n2,n3)], dmx1[i]);
        const real_t muz5 = kHalf*(dx3[km]*zc + dx3[k]*za)/dmx3[k];              // ref mu3 1431
        const real_t muz6 = kHalf*(dx3[kp]*zc + dx3[k]*zb)/dmx3[kp];             // ref mu4 1432
        dU[c] -= dt*( kQ*(dWm6*dUdz6 + dWm5*dUdz5)
                    - kHalf*Cmu*irc/dx3[k]*(muz6*dWmx6 - muz5*dWmx5) );          // ref 1451 (M12 part)

        // --- M13 : span increment (my dV) feeds dU via ∂U/∂y (ref dwm/dudz/dwmdx) ---
        const real_t dVm3 = mface(dx1[i], dV[cim], dx1[im], dV[c],                dmx1[i]); // ref dwm5 1434 → y-lo
        const real_t dVm4 = mface(dx1[i], dV[ix(im,jp,k,n2,n3)], dx1[im], dV[cjp], dmx1[i]); // ref dwm6 1435 → y-hi
        const real_t dUdy3 = (U[c]   - U[ix(i,jm,k,n2,n3)])/dmx2[j];             // ref dudz5 1437
        const real_t dUdy4 = (U[cjp] - U[c]  )/dmx2[jp];                         // ref dudz6 1438
        const real_t dVmx3 = (dV[c]  - dV[cim])/dmx1[i];                         // ref dwmdx5 1440
        const real_t dVmx4 = (dV[cjp]- dV[ix(im,jp,k,n2,n3)])/dmx1[i];           // ref dwmdx6 1441
        const real_t ya = mface(dx1[im], mu[ix(i,jm,k,n2,n3)], dx1[i], mu[ix(im,jm,k,n2,n3)], dmx1[i]);
        const real_t yb = mface(dx1[im], mu[cjp], dx1[i], mu[ix(im,jp,k,n2,n3)], dmx1[i]);
        const real_t muy3 = kHalf*(dx2[jm]*zc + dx2[j]*ya)/dmx2[j];              // ref mu5 1446 (yc==zc)
        const real_t muy4 = kHalf*(dx2[jp]*zc + dx2[j]*yb)/dmx2[jp];             // ref mu6 1447
        dU[c] -= dt*( kQ*(dVm4*dUdy4 + dVm3*dUdy3)
                    - kHalf*Cmu*irc/dx2[j]*(muy4*dVmx4 - muy3*dVmx3) );          // ref 1452 (M13 part)
      }
}

// =============================================================================
//  (4) FAITHFUL dd* forward-substitution coupling (block LOWER-triangular)
//      — the M21/M31/M32 off-diagonals, applied to the RHS using the already-
//      solved preceding increments, BEFORE the component's own ADI solve.
//      MPM-STD_main solve order dU→dV→dW (wall-normal LAST); the upper
//      M23/M12/M13 off-diagonals are block_couple_dV_var/dU_var above.
// =============================================================================

// ref MPM-STD solvedW 1524-1540 (M31ddU + M32ddV): the dU AND dV increments
// feed the wall-normal (dW) equation RHS, so both pass through the implicit
// z-ADI. mu/inh faces are reproduced IDENTICALLY to the dW body. Call AFTER
// dU and dV are solved+synced and AFTER assemble built rhs_w, BEFORE the dW
// ADI solve.
void couple_dd_dW_var(real_t* rhs_w, const real_t* dU, const real_t* dV, const real_t* W,
                      const real_t* mu, const real_t* invrho,
                      const real_t* dx1, const real_t* dmx1,
                      const real_t* dx2, const real_t* dmx2,
                      const real_t* dx3, const real_t* dmx3,
                      int n1, int n2, int n3, real_t Cmu, real_t dt) {
  const int h = kHaloWidth;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int im=i-1, ip=i+1, jm=j-1, jp=j+1, km=k-1;
        const int c   = ix(i, j, k, n2, n3);
        const int cim = ix(im,j, k, n2, n3), cip = ix(ip,j, k, n2, n3);
        const int cjm = ix(i, jm,k, n2, n3), cjp = ix(i, jp,k, n2, n3);
        const int ckm = ix(i, j, km,n2, n3);

        const real_t irc = mface(dx3[km], invrho[c], dx3[k], invrho[ckm], dmx3[k]);
        const real_t inh = kHalf*Cmu*irc;
        const real_t xc = mface(dx3[km], mu[c],   dx3[k], mu[ckm],               dmx3[k]);

        // --- M31: dU increment (ref solvedW 1524-1531) ---
        const real_t ddu1 = mface(dx3[km], dU[c],   dx3[k], dU[ckm],               dmx3[k]);
        const real_t ddu2 = mface(dx3[km], dU[cip], dx3[k], dU[ix(ip,j,km,n2,n3)], dmx3[k]);
        const real_t dWdx1 = (W[c]   - W[cim])/dmx1[i];
        const real_t dWdx2 = (W[cip] - W[c]  )/dmx1[ip];
        const real_t ddudz1 = (dU[c]   - dU[ckm])/dmx3[k];
        const real_t ddudz2 = (dU[cip] - dU[ix(ip,j,km,n2,n3)])/dmx3[k];
        const real_t xa = mface(dx3[km], mu[cim], dx3[k], mu[ix(im,j,km,n2,n3)], dmx3[k]);
        const real_t xb = mface(dx3[km], mu[cip], dx3[k], mu[ix(ip,j,km,n2,n3)], dmx3[k]);
        const real_t mu1 = kHalf*(dx1[im]*xc + dx1[i]*xa)/dmx1[i];
        const real_t mu2 = kHalf*(dx1[ip]*xc + dx1[i]*xb)/dmx1[ip];

        rhs_w[c] -= dt*( kQ*(ddu2*dWdx2 + ddu1*dWdx1)
                       - inh/dx1[i]*(mu2*ddudz2 - mu1*ddudz1) );                    // ref 1524-1531

        // --- M32: dV increment (ref solvedW 1533-1540) ---
        const real_t dvm5 = mface(dx3[km], dV[c],   dx3[k], dV[ckm],               dmx3[k]);
        const real_t dvm6 = mface(dx3[km], dV[cjp], dx3[k], dV[ix(i,jp,km,n2,n3)], dmx3[k]);
        const real_t dWdy5 = (W[c]   - W[cjm])/dmx2[j];
        const real_t dWdy6 = (W[cjp] - W[c]  )/dmx2[jp];
        const real_t ddVz5 = (dV[c]   - dV[ckm])/dmx3[k];
        const real_t ddVz6 = (dV[cjp] - dV[ix(i,jp,km,n2,n3)])/dmx3[k];
        const real_t ya = mface(dx3[km], mu[cjm], dx3[k], mu[ix(i,jm,km,n2,n3)], dmx3[k]);
        const real_t yb = mface(dx3[km], mu[cjp], dx3[k], mu[ix(i,jp,km,n2,n3)], dmx3[k]);
        const real_t muy5 = kHalf*(dx2[jm]*xc + dx2[j]*ya)/dmx2[j];
        const real_t muy6 = kHalf*(dx2[jp]*xc + dx2[j]*yb)/dmx2[jp];

        rhs_w[c] -= dt*( kQ*(dvm5*dWdy5 + dvm6*dWdy6)
                       - inh/dx2[j]*(muy6*ddVz6 - muy5*ddVz5) );                    // ref 1533-1540
      }
}

// ref MPM-STD solvedV 1211-1218 (M21ddU): the dU increment feeds the dV
// equation RHS. mu1/mu2/inh reproduced IDENTICALLY to the dV body. Call AFTER
// dU is solved+synced and rhs_v assembled, BEFORE the dV ADI solve. The M23
// (dW) coupling is NOT here — dV is solved BEFORE dW in the reference
// arrangement; dW feeds dV afterwards via block_couple_dV_var (Report/RBC).
void couple_dd_dV_var(real_t* rhs_v, const real_t* dU, const real_t* V,
                      const real_t* mu, const real_t* invrho,
                      const real_t* dx1, const real_t* dmx1,
                      const real_t* dx2, const real_t* dmx2,
                      int n1, int n2, int n3, real_t Cmu, real_t dt) {
  const int h = kHaloWidth;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)        // j = Y = OWN
      for (int k = h; k < n3 - h; ++k) {
        const int im=i-1, ip=i+1, jm=j-1;
        const int c   = ix(i, j, k, n2, n3);
        const int cim = ix(im,j, k, n2, n3), cip = ix(ip,j, k, n2, n3);
        const int cjm = ix(i, jm,k, n2, n3);

        // reused μ-faces + inh (identical to the dV body)
        const real_t xa = mface(dx2[jm], mu[cim], dx2[j], mu[ix(im,jm,k,n2,n3)], dmx2[j]);
        const real_t xc = mface(dx2[jm], mu[c],   dx2[j], mu[cjm],               dmx2[j]);
        const real_t xb = mface(dx2[jm], mu[cip], dx2[j], mu[ix(ip,jm,k,n2,n3)], dmx2[j]);
        const real_t mu1 = kHalf*(dx1[im]*xc + dx1[i]*xa)/dmx1[i];
        const real_t mu2 = kHalf*(dx1[ip]*xc + dx1[i]*xb)/dmx1[ip];
        const real_t irc = mface(dx2[jm], invrho[c], dx2[j], invrho[cjm], dmx2[j]);
        const real_t inh = kHalf*Cmu*irc;

        // V gradients the increment multiplies
        const real_t dvdx1 = (V[c]   - V[cim])/dmx1[i];
        const real_t dvdx2 = (V[cip] - V[c]  )/dmx1[ip];

        // M21 : dU increment (ref solvedV 1211-1218)
        const real_t ddu1   = mface(dx2[jm], dU[c],   dx2[j], dU[cjm],               dmx2[j]);
        const real_t ddu2   = mface(dx2[jm], dU[cip], dx2[j], dU[ix(ip,jm,k,n2,n3)], dmx2[j]);
        const real_t ddudy1 = (dU[c]   - dU[cjm])/dmx2[j];
        const real_t ddudy2 = (dU[cip] - dU[ix(ip,jm,k,n2,n3)])/dmx2[j];

        rhs_v[c] -= dt*( kQ*(ddu2*dvdx2 + ddu1*dvdx1)
                       - inh/dx1[i]*(mu2*ddudy2 - mu1*ddudy1) );                    // ref 1211-1218
      }
}

} // namespace mpmstd::equation::momentum::kernels

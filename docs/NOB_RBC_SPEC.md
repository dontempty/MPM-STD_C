# NOB Glycerol RBC — port spec (extracted from reference PaScaL_TCS Fortran)

Source: `/shared/home/xmwang0202/PaScaL_TCS-main/src/` (MPM-STD Beam–Warming delta-form ADI). Reference axis: x1(U),x2(V,**wall-normal/vertical**),x3(W). My port: **z = wall-normal/gravity** (so reference x2 → my z; buoyancy on W). Properties are polynomials in **Δθ = DeltaT·θ** (θ∈[−0.5,0.5]); NO separate θm offset (baked into coeffs + normalized by Rho0/Cp0/KPP0/Mu0).

## 1. Glycerol coefficients (`module_global.f90:129-137`)
```
density ρ : a10=1.2477e3  a11=-4.789e-4  a12=-0.3795e-6  a13=a14=a15=0
Cp        : b10=2.5108e3  b11=22.511e-4  b12=b13=b14=b15=0
conduct κ : c10=2.9351e-3 c11=3.863e-4   c12..c15=0
mu-factor : d10=238.71e-6 d11=-702.83e-4 d12=2393.1e-6 d13=-6923.0e-8 d14=33131.3e-10 d15=-71517.5e-12
Rho0=a10  Cp0=b10  KPP0=c10  Mu0=a10*d10
a12pera11 = (|a12|<1e-14 ? 0 : a12/a11)
groups: Cmu=√(Pr/Ra)  Ct=1/√(Ra·Pr)  Cmt=1
```
(OB sanity = all poly coeffs except *10 set to 0 → ρ̂=Ĉp=κ̂=μ̂=1.)

## 2. Property fields (recompute each step; let t=Δθ, t2=t²,…)
```
μ̂      = [a10(1+a11 t+a12 t2+…) · d10(1+d11 t+d12 t2+…)] / Mu0        (mom coeffi :165)
1/ρ̂    = 1/(1 + a11 t + a12 t2 + a13 t3 + …)                          (:168, Rho0/(a10(1+…)))
κ̂      = (1 + c11 t + c12 t2+…)                · c10/KPP0  = (1+c11 t+…)   (thermal :137)
dκ̂/dt  = (c11 + 2c12 t + 3c13 t2 + 4c14 t3 + 5c15 t4)                  (:138)  [derivative wrt Δθ, not ×Δ]
1/(ρ̂Ĉp)= 1/[(1+a11 t+a12 t2+a13 t3)(1+b11 t+b12 t2+b13 t3)]            (:140)
d[1/(ρ̂Ĉp)]/dt = analytic (:142-150, ≤cubic terms)
```

## 3. Momentum viscous term — full variable-μ stress + cross-stress (`module_solve_momentum.f90`)
Per component, prefactor `0.5·Cmu·invRhoc` (invRhoc = **face-interpolated 1/ρ̂** on that component's face). μ face-interpolated (volume-weighted by dx). Own-direction 2nd-deriv ×2, cross ×1:
```
dU RHS (:317-341): 0.5 Cmu invRhoc·( 2·∂x(μ∂xu) + ∂y(μ∂yu) + ∂z(μ∂zu) + ∂y(μ∂xv) + ∂z(μ∂xw) )
dV     (:691-697): 0.5 Cmu invRhoc·( ∂x(μ∂xv) + 2·∂y(μ∂yv) + ∂z(μ∂zv) + ∂x(μ∂yu) + ∂z(μ∂yw) )
dW     (:1049-55): 0.5 Cmu invRhoc·( ∂x(μ∂xw) + ∂y(μ∂yw) + 2·∂z(μ∂zw) + ∂x(μ∂zu) + ∂y(μ∂zv) )
```
- **Implicit (ADI tri-diag)**: only own-velocity, own-direction 2nd-deriv (mAC*/mAP*/mAM*, e.g. mACI :371 = invRhocCmu_half/dmx·(μ/dx_i+μ/dx_im)·2 + 0.25·conv). Matrix = mAC·dt+1, mAP·dt, mAM·dt.
- **Explicit (RHS)**: full stress above (all 5 terms incl cross) + convection(0.25 coeff) + buoyancy + −(M·Uⁿ).
- **Cross-velocity increment coupling** (after 3 ADI sweeps, explicit):
  - blockLdV (:1345): `dV -= dt(0.25(dwm·dvdz) − 0.5 Cmu invRhoc/dx·(μ6·∂y dwm6 − μ5·∂y dwm5))`
  - blockLdU (:1450): `dU -= dt(M12·dVm + M13·dWm)` with the variable-μ cross terms.

## 4. Energy — variable-κ (`module_solve_thermal.f90:253-299`)
Prefactor `0.5·Ct·invRhoCp`. Variable-κ Laplacian with product-rule linearization + prefactor linearization:
```
viscous_e1 = (kPP2·dedx2 − kPP1·dedx1 − e2·dkPP2·dedx2 + e1·dkPP1·dedx1)/dx1   (+e2,e3 analog)
RHS += 0.5 Ct invRhoCp·(ve1+ve2+ve3)  −  0.5 Ct dinvRhoCp·θ·∇·(κ∇θ)
convection: −0.5·(0.5(U_i dedx1+U_ip dedx2)+…)        ! 0.5 skew form
```
kPP/dkPP face-interp; e=face θ; ded=face θ-grad. Implicit diag carries the dkPP correction (eAC* :316-339).

## 5. Buoyancy (`:347-352`) — put on WALL-NORMAL momentum (my W)
```
Tc = θ interpolated to the (wall-normal) face
F  = invRhoc · Cmt · ( Tc + a12pera11·Tc²·DeltaT )     ! Cmt=1; θ² term = NOB density nonlinearity
```
Fully explicit (RHS). OB: a12pera11=0 → F = invRhoc·θ.

## 6. RBC case (Fig 9 = paper §4.3; reference PARA_INPUT is a demo)
- **128×128×128**, x,y periodic + z wall (gravity); θ walls = +0.5 (hot bottom) / −0.5 (cold top).
- Γ = Lx/H = Lz/H = **6** (paper; reference demo used 4×2), H=Lz=1; tanh-stretch z (γ≈2.6, "ny twice Horn" for the thin glycerol thermal BL).
- **Ra=1e6, Pr=2547**, Δ ∈ {20,40,60} K (the polynomial argument scale); θm=313K baked into coeffs.
- Diagnostic **Θ_c** = ⟨θ⟩_{x,y,t} at mid-height z=Lz/2 (NOT in reference solver — post-processing). OB ⇒ Θ_c=0; NOB ⇒ Θ_c−Θ_m>0 growing with Δ (Fig 9). Θ_m=(θ_h+θ_c)/2=0.

## Port plan (mpmstd)
1. `physics/properties`: `update_properties_cpu` → fields μ̂,1/ρ̂,κ̂,dκ̂,1/(ρ̂Ĉp),d[1/(ρ̂Ĉp)] from θ (glycerol coeffs, arg Δθ). PropertyModel holds a*/b*/c*/d* + DeltaT.
2. `equation/momentum` `assemble_momentum_var_visc_cpu` + var-μ kernels (full stress + cross) + blockLd with variable μ/invRhoc.
3. `equation/scalar` var-κ assemble (κ̂ field + dκ̂ linearization).
4. `physics/buoyancy`: NOB form F=invRhoc·(θ+a12pera11·θ²·Δ) on W (extend BuoyancyParams with a12pera11, DeltaT, invRho field).
5. `apps/rbc`: NOB recipe (update_properties → energy(var κ) → momentum(var μ)+buoyancy → pressure). Θ_c diagnostic (have it).
6. Run Δ=20/40/60 (Ra=1e6,Pr=2547,Γ=6,128³) → Θ_c−Θ_m vs Δ → Fig 9.

⚠ CFL: Pr=2547 stiff (thin thermal BL); momentum CFL sets dt; long averaging for Θ_c. CPU-feasible on dedicated nodes (infinite walltime) but slow per Δ.

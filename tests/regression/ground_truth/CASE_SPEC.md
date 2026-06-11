# Reference (PaScaL_TCS) ground-truth — momentum validation case (2026-06-11)

PaScaL_TCS = TRUTH. This is a CLEAN, deterministic, fixed-dt, healthy run for
field-by-field validation of the faithful momentum port (and energy/pressure).

## Case (uniform grid → both codes share identical cell coordinates)
- Grid: n1m=32 (stream,x1,periodic), n2m=48 (wall,x2,Dirichlet), n3m=32 (span,x3,periodic). UNIFORM all dirs.
- Box: L1=2 (Aspect1=2), L2=H=1, L3=2 (Aspect3=2). Cell centre (uniform,serial): xc=(idx-0.5)*L/Nm.
- Physics: Ra=2000, Pr=1. OB (reference module_global currently a10=b10=c10=d10=1, rest 0 → props≡1, a12pera11=0).
  nu=Cmu=sqrt(Pr/Ra)=0.022360, alpha=Ct=1/sqrt(Ra*Pr)=0.022360. Buoyancy on U(x1): F=Cmt*(Tc)*invRho, Cmt=1, invRho=1.
- BC: T hot Tbt=+0.5 (x2=0 wall), cold Tup=-0.5 (x2=L2 wall). No-slip walls. Periodic x1,x3.
- Fixed dt: dtStart=dtMax=4.0e-3, MaxCFL=1e8 (CFL never limits → dt≡0.004). 200 steps → t=0.8.
- IC (mpi_thermal_initial, edited): T = Tbt + (Tup-Tbt)*xc2/L2 + amp*sin(2*pi*xc1/L1)*sin(pi*xc2/L2)*cos(2*pi*xc3/L3),
  amp=0.05; velocity U=V=W=0. Deterministic; breaks parallel symmetry → cross-stress/cross-conv active.

## Reference time-loop order (match in my code)
thermal_coeffi → thermal_solve → ghost(T) → momentum_coeffi → solvedU → ghost(dU) → solvedV → ghost(dV)
→ solvedW → ghost(dW) → blockLdV → ghost(dV) → blockLdU → pseudoupdate(U,V,W) → ghost(U,V,W)
→ pressure RHS → Poisson → ghost(dP) → projection → ghost(U,V,W,P) → CFL/div.
NOTE: buoyancy is INSIDE solvedU; in my code it's main-composed on U (gravity_axis=0=stream). Pressure projection equivalent.

## Dumps (rank0, layout: i=1:n1msub outer, j=1:n2msub mid, k=1:n3msub inner; raw <f8)
- dump_pascal_T.bin  ↔ my T
- dump_pascal_U.bin  ↔ my U      (ref x1 stream  → my X)
- dump_pascal_V.bin  ↔ my W      (ref x2 wall    → my Z)   ← AXIS SWAP
- dump_pascal_W.bin  ↔ my V      (ref x3 span    → my Y)   ← AXIS SWAP
Each 32*48*32 = 49152 doubles.

## Ground-truth stats (t=0.8, step 200)
T  : min -0.49055 max 0.49060 rms 0.28881
U  : min -0.19431 max 0.19447 rms 0.13329   (streamwise buoyancy response)
V(ref,→myW): min -0.00172 max 0.00177 rms 0.00058
W(ref,→myV): min -0.01004 max 0.01004 rms 0.00402   (secondary, cross-driven)

## My-side matching TODO (next)
1. Integrate faithful momentum port (docs/drafts/momentum_faithful_port_draft.md).
2. A DHVC-geometry run in MY code: gravity_axis=0 (stream X), z... NO: my DHVC walls are Z. Reference walls are x2→my Z.
   Map: my X=stream(periodic), my Z=wall(Dirichlet, T±0.5), my Y=span(periodic). Lx=2,Lz=1,Ly=2. n: X=32,Z=48,Y=32.
   Buoyancy on X (gravity_axis=0). IC: T linear in Z + amp*sin(2pi*xX/Lx)*sin(pi*xZ/Lz)*cos(2pi*xY/Ly); U=V=W=0.
   Fixed dt=0.004 (MaxCFL huge). 200 steps. Dump U,V,W,T (my layout → reorder to ref i=X/j=Z/k=Y for compare).
3. Run with CURRENT momentum (baseline discrepancy) AND faithful (should match ref better). compare_pascal_tcs.py per field.

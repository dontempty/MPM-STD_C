# Goal
- 최종적으로는 도심풍 해석을 현재 만들고 있는 C++ 코드를 이용해서 돌리는 것이다.

# Dependency
- Boundary Layer
- LES

# Validation

## Channel 
- $Re_{\tau} = 180, 550$ 에서 1, 2차 통계를 검증

[Kim, John, Parviz Moin, and Robert Moser. "Turbulence statistics in fully developed channel flow at low Reynolds number." Journal of fluid mechanics 177 (1987): 133-166.]

## RBC(NOB)
- Ra = 1e6, $\Delta T = 20, 40, 60$ 3가지 케이스에 대해서 center line Temperature를 검증한다.
- 가변 물성을 검증한다.

[Horn, Susanne, Olga Shishkina, and Claus Wagner. "On non-Oberbeck–Boussinesq effects in three-dimensional Rayleigh–Bénard convection in glycerol." Journal of Fluid Mechanics 724 (2013): 175-202.]

## DHVC(OB)
- Ra = 1e5, 1e6, 1e7, 1e8 케이스에서 $Re_{\delta^*}$를 검증한다.

[Ng, Chong Shen, et al. "Changes in the boundary-layer structure at the edge of the ultimate regime in vertical natural convection." Journal of fluid mechanics 825 (2017): 550-572.]

## Cavity(2D)
- 온도를 함께 푸는 cavity 이다.
- periodic 말고도 다른 조건에서도 동작 한다는 것을 검증한다.

[de Vahl Davis, Graham. "Natural convection of air in a square cavity: a bench mark numerical solution." International Journal for numerical methods in fluids 3.3 (1983): 249-264.]

## LES(channel)
- channel 에서 constant vreman model 을 검증한다.

[Vreman, A. W. "An eddy-viscosity subgrid-scale model for turbulent shear flow: Algebraic theory and applications." Physics of fluids 16.10 (2004): 3670-3681.]

## Boundary Layer
- 단계 A (2D 층류 ZPG): 유입/유출/자유류 BC 와 공간 발달 통계를 **Blasius 상사해**로 검증 — $C_f = 0.664/\sqrt{Re_x}$, $\delta^*$, $\theta$, $H = 2.591$, $u/U_\infty = f'(\eta)$. (완료: 0.2<x/Lx<0.8 에서 ≤0.07%)
- 단계 B (3D 난류 ZPG, $Re_\theta \le 1100$): 재순환-재스케일 유입(Lund 1998)으로 $Re_\theta$ = 670, 1000 에서 $C_f$, $H$, $Re_\tau$, $U^+$, $u'v'w'_{rms}^+$, $\overline{uv}^+$ 를 KTH DNS 와, $Re_\theta$ = 1100 은 Torroja DNS(같은 재순환 유입)와 교차 검증한다.

[Schlatter, Philipp, and Ramis Örlü. "Assessment of direct numerical simulation data of turbulent boundary layers." Journal of Fluid Mechanics 659 (2010): 116-126.]  — 주 참조 (데이터: www.mech.kth.se/~pschlatt/DATA, `vel_XXXX_dns.prof`)

[Jiménez, Javier, Sergio Hoyas, Mark P. Simens, and Yoshinori Mizuno. "Turbulent boundary layers and channels at moderate Reynolds numbers." Journal of Fluid Mechanics 657 (2010): 335-360.]  — 교차 참조 $Re_\theta$ 1100 (데이터: torroja.dmt.upm.es/turbdata/blayers; 코드·재순환 유입은 Simens, Jiménez, Hoyas, Mizuno, J. Comput. Phys. 228 (2009): 4218-4231 — "유입 뒤 300 θ 는 버려라")

[Lund, Thomas S., Xiaohua Wu, and Kyle D. Squires. "Generation of turbulent inflow data for spatially-developing boundary layer simulations." Journal of Computational Physics 140.2 (1998): 233-258.]  — 유입 생성법

[Schlichting, Hermann, and Klaus Gersten. Boundary-Layer Theory. 9th ed., Springer (2017), Ch. 6.]  — Blasius 상사해 (원전: Blasius, Z. Math. Phys. 56 (1908): 1-37)

## IBM (2D, 3D)
- solid 처리가 가능한지 검증한다.

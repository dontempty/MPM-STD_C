# MPM-STD_C vs PaScaL_TCS — 수치 방법론 충실도 비교 보고서 (2026-06-15)

> 레퍼런스(ground truth): `/shared/home/wel1come1234/workspace/PaScaL_TCS/src/` (Fortran)
> 대상: `/shared/home/wel1come1234/workspace/MPM-STD_C/` (C++/CUDA)
> 방법: 5영역 병렬 코드 대조(momentum 대류·잔차 / momentum 점성·커플링 / thermal / pressure·fractional-step / CFL·buoyancy·properties) + 적대적 재검증. 모든 주장은 양측 file:line + 실제 식으로 확인.
> 관련: [14_momentum_fidelity_vs_fortran.md](14_momentum_fidelity_vs_fortran.md)(자매코드 MPM-STD 대비), [17_dhvc_cpu_bottleneck.md](17_dhvc_cpu_bottleneck.md)(CFL).
> ⚠ 분석 산출물입니다. **소스 코드는 수정하지 않았습니다.**

---

## 0. 결론 (Executive Summary)

**MPM-STD_C는 단일 코드가 아니라, 충실도가 정반대인 두 경로를 가진다.**

- **NOB 경로**(var-μ/var-κ, `apps/rbc/main_cpu_nob.cpp`) = PaScaL_TCS의 **거의 줄 단위(line-by-line) 충실한 이식**. 응력 텐서 전항(own×2 + 전치 교차응력), delta-form `−M·qⁿ` 잔차, 6-블록 속도 결합, 0.25 반음해 대류, 변수 κ̂/dκ̂ 선형화까지 모두 재현(문서화된 축 스왑 x2↔z, x3↔y 포함).
- **OB 경로**(const-visc/const-diff, `apps/dhvc/main.cpp`) = **방법론적으로 발산하는 단순화 특수화**.

**THE 핵심 차이:** PaScaL_TCS는 *모든* 케이스에서 대류를 **반음해(semi-implicit)** 로 M-연산자 안에 넣고 `−M·qⁿ` 잔차를 RHS에서 빼는 단일 delta-form ADI를 쓴다(`module_solve_momentum.f90:371-399`). 반면 MPM-STD_C **OB 경로**는 대류를 **완전 양해(fully explicit)** 로 RHS에 전부 넣으면서(conv_factor=1.0) 정합되지 않는 0.25 음해 밴드까지 더하고 `−M·qⁿ` 잔차는 **완전히 누락**한다. 이 양해 대류가 **이류(advective) CFL 안정성 한계를 훨씬 빡빡하게** 만들어 **Ra=1e7 dt 붕괴의 직접 원인**이다.

> **사용자 의심 판정:** "운동량 등의 구현이 PaScaL_TCS와 다르다" → **OB(DHVC) 경로에 대해 사실로 확정**, **NOB(RBC) 경로에 대해서는 충실함으로 확정**.

> **중요한 인과 정정(적대적 검증):** "PaScaL_TCS엔 대류 안정성 벽이 *없다*"는 과장이다. PaScaL_TCS의 0.25/0.25 CN-skew 분할도 *무조건 안정은 아니며* 이류 CFL 상한이 존재한다 — 단지 훨씬 **관대**할 뿐. 정확히는: **C++ OB의 완전 양해 대류는 두 코드가 공유하는 이류 상한을 훨씬 더 조이고, 그래서 Ra=1e7(자유낙하 속도↑)에서 MaxCFL을 더 줄여야 해 dt가 붕괴**한다. 두 CFL 루틴 자체는 형식 동일(이류항만, 점성 한계 없음): C++ `cfl_cpu.cpp:24-26,33` ↔ ref `module_post.f90:207-209,224-225`.

---

## 1. 충실도 매트릭스 (Fidelity Matrix)

verdict: **faithful**(충실) / **equiv**(등가-다른형식) / **simplified**(단순화) / **divergent**(발산) / **missing**(누락)

| 영역 / 항목 | PaScaL_TCS | MPM-STD_C | OB | NOB | Impact |
|---|---|---|---|---|---|
| 운동량: 대류 음해성 | 0.25 반음해 밴드 in M (`mom.f90:371-394`), `ac=mACK·dt+1` (`:410`) | OB 양해+미정합 0.25밴드 / NOB 충실 (`mom_var:117-126`) | **divergent** | faithful | **high** |
| 운동량: `−M·qⁿ` 잔차 | RHS−=Σ(mAP·q_p+mAC·q_c+mAM·q_m) (`mom.f90:396-399`) | OB 누락 / NOB 충실 (`mom_var:128-133`) | **divergent** | faithful | **high** |
| 운동량: own방향 점성 ×2 | `2·viscous_u1` (`:341`), `mACI…·2.d0` (`:371`) | OB 순수 Laplacian, ×2 없음 (`mom_ker:69`) / NOB `kTwo·vu1` (`mom_var:114`) | **divergent** | faithful | **high** |
| 운동량: 전치 교차응력 | `viscous_u12/u13` (`:338-339`) | OB 부재 / NOB `vuZc/vuYc` (`mom_var:112-114`) | **divergent** | faithful | **high** |
| 운동량: 블록 후치환 결합 교차응력 | 대류+점성 (`:1345,1449`) | OB 대류만 (`mom_ker:356,404`) / NOB 둘다 (`mom_var:474-489`) | simplified | faithful | medium |
| 운동량: 하삼각 전치환 dd* | solvedV/W RHS−=dd* (`:764,1114`) | OB 부재 / NOB `couple_dd_dW/dV_var` (`mom_cpu:194-201`) | **divergent** | faithful | medium |
| 운동량: ADI 스윕/순환 TDMA | z→x→y, 벽축 마지막 (`:420,476,541`) | periodic 먼저, 벽축 마지막 (`mom_cpu:91-97`) | equiv | equiv | low |
| 에너지: 대류 음해성 | 0.5-skew 양해 + 0.25 음해 밴드 (`thm.f90:296-299,344-362`) | OB 완전 양해 (`scal_ker:90-94,131-133`) / NOB 충실 (`scal_var:85-89,133-136`) | **divergent** | faithful | **high** |
| 에너지: 대류 형식 | 이류형 0.5-skew (`:296`) | 보존형 flux-div (`scal_ker:83-92`) | equiv | — | medium |
| 에너지: 변수 κ̂+dκ̂, 1/(ρ̂Ĉp)·Ct | `viscous_e1`,`invRhoCpCt_half` (`:316,278`) | NOB 충실 (`scal_var:139-140,122-123`) | — | faithful | high |
| 압력: FFT+TDMA Poisson/파수/pin | FFT x1,x3; TDMA x2 (`pre.f90:309,636,628`) | FFT x,y; TDMA z (`pre_eng:293,728,765`) | faithful | faithful | low |
| 압력: OB projection (dt·∇P) | `U−=dt·Cmp·gradP` (`:805`) | `u−=dt·gradP` (`pre_base:127`) | faithful | faithful | low |
| 압력: NOB dPhat 예측자 | 2차 `dPhat=2dP−dP0` (`:820`) | 단순 복사 (`pre_eng:447`) | — | simplified | medium |
| 압력: 벽 P ghost | 2차 quadratic (`:855`) | 1차 Neumann (`pre_eng:342`) | simplified | simplified | medium |
| 분수단계 순서 | thermal→solvedUVW→blockLd→pseudoupdate→pressure→project (`main.f90:133-184`) | scalar→momentum→solve→update→pressure (`dhvc:194-213`/`rbc:104-119`) | equiv | equiv | medium |
| 순유량 보정 (DHVC) | 부재; 벽속도=0 (`global:109`) | DMpresg 추가 (`dhvc:206`) | **divergent** | — | medium |
| CFL 공식·속도척도·dt갱신 | 면평균 staggered Σ|u|/dx, `MaxCFL/maxUovX` cap (`post:207-209,224`) | 셀저장 staggered Σ|u|/dx, `min(MaxCFL/speed,cap)` (`cfl:24-33`) | faithful | faithful | low |
| 점성/확산 dt 한계 | 없음(이류만) | 없음(이류만) (`cfl:24-26`) | faithful | faithful | low |
| 부력 형식 θ+a12pera11·θ²·ΔT | `Cmt·(Tc+a12pera11·Tc²·ΔT)·invRhoc` (`:347-352`) | `th+a12pera11·th²·dT` (`buoy:47`) | — | faithful | medium |
| 부력 T 위치 | 면 보간 Tc (`:348`) | 셀중심 t[c] (`buoy:25,47`) | divergent | divergent | medium |
| 무차원 군 (free-fall) | Cmu=√(Pr/Ra), Ct=1/√(RaPr), Cmt=1 (`global:141`) | nu=√(Pr/Ra), alpha_T=1/√(RaPr) (`rbc:41`/`dhvc:115`) | faithful | faithful | low |
| 물성 다항식 (μ,κ,ρ,Cp+도함수) | mpi_*_coeffi (`thm:162`,`mom:165`) | properties_cpu.cpp:48-63 | — | faithful | low |

*약어: mom=module_solve_momentum.f90, thm=module_solve_thermal.f90, pre=module_solve_pressure.f90, post=module_post.f90, global=module_global.f90 / mom_ker=momentum_kernels_cpu.cpp, mom_var=momentum_var_kernels_cpu.cpp, mom_cpu=momentum_cpu.cpp, scal_ker=scalar_kernels_cpu.cpp, scal_var=scalar_var_kernels_cpu.cpp, pre_eng=pressure_engine_cpu.cpp, pre_base=pressure_base_cpu.cpp, buoy=buoyancy_cpu.cpp, cfl=cfl_cpu.cpp.*

---

## 2. 핵심 차이 상세 (HIGH-impact, 영향순)

### #1 운동량 대류: 반음해(ref) vs 완전 양해(C++ OB) — Ra=1e7 dt 붕괴의 근원

**PaScaL_TCS** (`module_solve_momentum.f90:371-394, 410`): 대류가 M-연산자 밴드에 0.25 가중으로 들어가 TDMA로 음해 풀림.
```fortran
mACI = invRhocCmu_half/dmx1(i)*(Mu/dx1(i)+Mu/dx1(im))*2.d0 &
       + 0.25d0*(dudx2*0.5d0+dudx1*0.5d0 - u2/dx1(i) + u1/dx1(im))   ! L371-372
...
ac(i,j,k) = mACK*dt + 1.d0   ! L410  (밴드를 TDMA로 음해 해)
```

**MPM-STD_C OB** (`momentum_cpu.cpp:27`; `momentum_kernels_cpu.cpp:43,83`):
```cpp
const real_t conv = real_t{1};                       // momentum_cpu.cpp:27
const real_t conv_coeff = -conv_factor * dt;         // = -1.0*dt   :43
rhs[c] = nu_dt * visc_rhs + conv_coeff * div_uq;     // 전체 div(u·q) 양해, :83 (마지막 문장)
```
- 대류 100%를 RHS 양해로 넣고, **추가로** 0.25 음해 대류 밴드(`build_adi_bands:114,136-138`)까지 만든다 → **대류가 ~1.25배 이중계상**(주석 `:24,:71`은 "0.25만 양해, 0.75는 밴드 흡수"라 하지만 실제 호출은 conv=1.0 양해 + 0.25 밴드로 주석과 모순).
- **결과:** C++ OB의 양해 대류는 두 코드가 공유하는 이류 CFL 상한을 ref의 반음해보다 훨씬 빡빡하게 만든다 → Ra=1e7에서 dt를 강제로 줄여 붕괴. **순수 발산이며 의심의 핵심.**

### #2 운동량 delta-form `−M·qⁿ` 잔차 누락 (C++ OB) — 정상상태 비자기정합

**PaScaL_TCS** (`module_solve_momentum.f90:396-399`):
```fortran
RHS = RHS - ( mAPI*U(ip)+mACI*U(i)+mAMI*U(im) &
            + mAPJ*U(jp)+mACJ*U(i)+mAMJ*U(jm) &
            + mAPK*U(kp)+mACK*U(i)+mAMK*U(km) )
```
→ 진정한 증분 방정식: U가 정상상태면 RHS→0 ⇒ Δq→0.

**MPM-STD_C OB**: `compute_mpmstd_rhs`가 `:83`에서 끝남 — 밴드 연산자를 q에 적용해 빼는 부분이 **전혀 없다**. 0.25 음해 대류 밴드에 대응하는 RHS 오프셋이 없어 OB 증분 방정식은 잔차형이 아니다(정상상태에서 비소거). (NOB는 `momentum_var_kernels_cpu.cpp:128-133`에서 충실 재현.) **순수 발산.**

### #3 점성 연산자(OB): 순수 Laplacian — own×2·전치응력 누락 + visc_factor 모순

**PaScaL_TCS** (`module_solve_momentum.f90:341`): `RHS = invRhocCmu_half*(2.d0*viscous_u1 + viscous_u2 + viscous_u3 + viscous_u12 + viscous_u13)` — own방향 ×2(수정 GE), 전치 교차응력 포함.

**MPM-STD_C OB** (`momentum_kernels_cpu.cpp:69`): `visc_rhs = Lx + Ly + Lz;` — 각 방향 1회, ×2 없음, 전치응력 없음.
- **적대적 검증 추가 발견:** `adi_component`가 own방향 포함 **모든 스윕에 visc_factor=0.5**를 넘긴다(`momentum_cpu.cpp:83`). 커널 주석(`momentum_kernels_cpu.cpp:22,94`: "1.0→backward Euler, own방향")과 **모순** — own방향이 backward-Euler조차 아니고 순수 CN 0.5다. (NOB는 `momentum_var_kernels_cpu.cpp:114`에서 충실.)

### #4 에너지(OB const-diff): 완전 양해 대류 vs ref 반음해

**PaScaL_TCS** (`module_solve_thermal.f90:296-299, 344-362`): 유일 스킴이 0.5-skew 반음해. RHS엔 `−0.5·Σ(0.5(U·de+U·de))`만, 나머지 0.25·U/dmx는 음해 밴드(`eAPI=…+0.25d0*U(ip)/dmx1(ip)`).

**MPM-STD_C OB** (`scalar_cpu.cpp:32`; `scalar_kernels_cpu.cpp:90-94,131-133`): 전체 ∇·(uφ)를 RHS 양해, 밴드는 `a=−0.5·dt·α/(dx·dmx)`, `b=1−a−c`의 **속도 0 순수확산**. **발산.**
- **중요(검증 강조):** DHVC는 운동량 *그리고* 에너지 **둘 다 양해 대류**를 쓴다(`apps/dhvc/main.cpp:194,199`). → **이류 상한이 두 수송량 모두에 걸린다.** 운동량만 고치면 Ra=1e7 dt 붕괴가 절반만 해결된다.

### 의도적 등가 단순화 vs 진정한 발산 (구분)
- **등가(equiv)**: ADI 스윕 순서/순환 TDMA, 분수단계 거시순서, CFL 면평균 vs 셀저장 속도척도, 벽 BC를 ghost+modify_tdma_row로 처리, 보존형 vs 이류형 대류 형식. 상수-μ·발산자유·동일 토폴로지에서 수학적 등가.
- **진정한 발산**: #1~#4 — 상수-μ·발산자유 가정 하에서도 **시간적분 안정성·정상상태 정합성이 ref와 다르다.**

---

## 3. OB vs NOB — NOB가 PaScaL_TCS의 진짜 쌍둥이

NOB(var-μ/var-κ, `apps/rbc`)는 OB가 놓친 거의 모든 것을 닫는다:

| 항목 | OB (DHVC) | NOB (RBC) |
|---|---|---|
| 대류 음해성 | 양해 + 미정합 0.25밴드 (발산) | 0.25 반음해 충실 (`mom_var:117-126`) |
| `−M·qⁿ` 잔차 | 누락 (발산) | 충실 (`mom_var:128-133`) |
| own×2 점성 | 순수 Laplacian (발산) | `kTwo` 충실 (`mom_var:114`) |
| 전치 교차응력 | 부재 (발산) | `vuZc/vuYc` (`mom_var:112-114`) |
| 하삼각 dd* 전치환 | 부재 (발산) | `couple_dd_dW/dV_var` (`mom_cpu:194-201`) |
| 에너지 대류 음해성 | 양해 (발산) | 0.5-skew 반음해 충실 (`scal_var:85-89`) |
| 변수-μ 면보간 | N/A | 이중 체적가중 충실 (`mom_var:94-102`) |

NOB는 PaScaL_TCS 줄 단위 이식(`// ref ...` provenance 주석 `mom_var:81-133`, 축 스왑 x2=wall→z, x3=span→y 포함). **남은 작은 발산 2건**: (a) 부력이 셀중심 T 사용(ref는 면보간), (b) NOB 압력 dPhat 예측자가 ref 2차 외삽(2dP−dP0) 대신 단순 복사(`pre_eng:447`).

---

## 4. 권고 (영향순 — 제안만, 코드 미수정)

| 순위 | 권고 | 위치 | 효과 |
|---|---|---|---|
| **1** | **OB 경로 폐기 → DHVC도 NOB 커널(μ≡1, invρ≡1) 사용.** `assemble_momentum_var_visc_cpu`는 상수물성에서 올바른 const-property 스킴(반음해 대류 + `−M·uⁿ` 잔차 + own×2 + 전치응력)으로 정확히 환원됨(`momentum_cpu.cpp:38-40` 주석). 가장 깨끗하고 #1·#2·#3 동시 해결 | `momentum_cpu.cpp`, `apps/dhvc/main.cpp:199` | **Ra=1e7 dt 붕괴(운동량분) 제거 + ref 정합** |
| **2** | **OB 에너지도 반음해로** (이미 구현된 `scalar_var_kernels_cpu.cpp:85-89` 재사용, 또는 DHVC scalar를 NOB var-κ로 통일) | `scalar_cpu.cpp:32`, `apps/dhvc/main.cpp:194` | **dt 붕괴의 나머지 절반(에너지분) 제거** — #1과 반드시 함께 |
| 3 | NOB 부력 T를 운동량 면으로 보간 | `buoyancy_cpu.cpp:25,47` ← ref `mom.f90:348` | 벽 근처 O(dz) 부력 위상오차 제거 |
| 4 | NOB 압력 dPhat 2차 예측자 복원 (2dP−dP0) | `pressure_engine_cpu.cpp:447` ← ref `pre.f90:820` | NOB 압력 시간정확도 |
| 5 | DHVC 순유량 보정(DMpresg) 정당성 검증 (ref 무대응물) | `apps/dhvc/main.cpp:206` | ref와 1:1 비교 시 분리평가 |

> ⚠ **#1과 #2는 반드시 함께.** DHVC는 운동량·에너지 둘 다 양해 대류이므로 한쪽만 고치면 이류 CFL 상한이 다른 쪽에 여전히 걸려 dt 붕괴가 절반만 해소된다.

> **GPU도 동일 적용 필요:** 동일 OB 발산이 GPU OB 경로에도 존재(단, GPU OB scalar는 NOB var-stencil을 kappa=1로 재사용해 *부분* 반음해 — [메모리 project_cpu_gpu_scalar_ob_mismatch] 참조). 가장 일관된 길은 CPU·GPU 모두 DHVC를 var 커널로 통일하는 것.

---

## 5. 교차참조 / 정정

- **Report 14**(자매코드 MPM-STD 대비)의 3대 발견(양해 대류, `−M·uⁿ` 누락, 순수 Laplacian)은 PaScaL_TCS 대조에서도 **OB 경로에 모두 참**이며 **NOB 경로에서 모두 닫혔음**을 본 분석이 확정.
- **Report 17(CFL) 정정:** dt 붕괴는 "ref엔 없는 안정성 벽"이 아니다. 두 코드 모두 이류 CFL 상한을 공유하고 점성 dt 한계는 없다. 정확히는 **C++ OB의 완전 양해 대류가 그 공유 상한을 ref의 반음해보다 훨씬 빡빡하게 조이는 것** — Report 17의 "음해 대류가 근본 해법"이라는 결론은 유지되나, "벽이 새로 생긴다"가 아니라 "상한이 크게 좁아진다"로 표현이 정확.

---

## 6. 한 줄 요약

> MPM-STD_C의 **NOB(RBC) 경로는 PaScaL_TCS의 충실한 줄 단위 이식**이지만, **OB(DHVC) 경로는 발산하는 단순화**다 — 대류를 ref의 반음해 대신 완전 양해로(게다가 0.25 밴드와 ~1.25× 이중계상) 처리하고 `−M·uⁿ` 잔차·own×2 점성·전치응력을 누락한다. 이 양해 대류(운동량+에너지 둘 다)가 두 코드가 공유하는 이류 CFL 상한을 크게 좁혀 Ra=1e7 dt 붕괴를 만든다. **해법은 DHVC를 (이미 충실한) NOB var 커널로 통일하는 것**이며, 이는 ref 정합과 1e7 실현을 동시에 달성한다.

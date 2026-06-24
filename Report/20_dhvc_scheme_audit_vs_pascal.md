# DHVC 계산 로직 전면 재감사 ↔ PaScaL_TCS (2026-06-15)

> ref(ground truth): `/shared/home/wel1come1234/workspace/PaScaL_TCS/src/` (Fortran)
> 대상: `/shared/home/wel1come1234/workspace/MPM-STD_C/` (DHVC = OB, unit-property 경로)
> 범위: 에너지·운동량·압력·구동(시간루프/무차원/부력/격자/CFL/IC) **전 방정식** 라인 단위 재대조.
> 방법: 4개 에이전트가 방정식별로 Fortran ↔ C++ 라인 대조, 핵심 2건(압력 투영·mass-flow)은 직접 라인 확인.
> 관련: [18_mpmstd_c_vs_pascal_tcs.md](18_mpmstd_c_vs_pascal_tcs.md), [19_dhvc_secondary_divergences_vs_pascal.md](19_dhvc_secondary_divergences_vs_pascal.md).

---

## 0. 결론 (요약)

**DHVC는 더 이상 옛 OB const 커널을 쓰지 않는다.** [apps/dhvc/main.cpp:210-234](../apps/dhvc/main.cpp) 가 `assemble/solve_scalar_var` + `assemble/solve_momentum_var`를 **unit props(μ=κ=1, dκ=0, invρ=1)** 로 호출 → PaScaL 스킴으로 환원. Report 18의 "OB 발산 4건"은 **이미 제거된 옛 커널 기준**이며 현 DHVC와 무관(stale).

| 영역 | 판정 | 비고 |
|---|---|---|
| 에너지 (thermal) | ✅ **동등** | delta-form 3-factor ADI, 50/50 반음해 대류, −M·θⁿ 잔차, 벽처리 모두 일치 |
| 운동량 (momentum) | ✅ **동등** | Beam-Warming delta, 반음해 skew 대류, −M·qⁿ, own×2 점성, 6-블록 결합 재현. spurious grad(div u) 없음 |
| 부력 | ✅ **동등** | 면보간 Tc, 계수 1, T_ref=0 (Report 19에서 수정 완료) |
| 무차원화 | ✅ **동등** | ν=√(Pr/Ra), α=1/√(Ra·Pr) |
| 격자 메트릭 | ✅ **동등** | uniform/tanh 동일식, dx/dmx 대수적 일치 |
| Poisson 연산자·divergence·projection 부호 | ✅ **동등** | stencil·modified wavenumber·zero-mean·singular-mode pin 일치 |
| 단계 순서 | ✅ **동등** | energy → momentum(+buoy) → pressure-project |
| **압력 투영 방식** | ⚠️ **실차이** | PaScaL=incremental(Pⁿ predictor + P+=dP), MPM=non-incremental(P 덮어쓰기). **Report 19 #3 정정** |
| mass-flow 보정 ⟨U⟩=0 | ⚠️ **실차이(의도적)** | MPM에만 있음, PaScaL 없음 |
| CFL 레짐·IC·벽 압력 ghost | ⚠️ **하네스 차이** | 스킴 아님, 정량 비교 시 정렬 필요 |

→ **이산화 코어는 PaScaL_TCS와 충실 일치.** 본질적 잔여 스킴 차이는 **압력 투영(incremental vs non-incremental)** 한 건이며, mass-flow·CFL·IC는 의도적/검증하네스 수준.

---

## 1. ✅ 동등성 확인 (대수적 항등 또는 bit 일치)

### 1.1 에너지 (thermal)
ref [module_solve_thermal.f90](../../PaScaL_TCS/src/module_solve_thermal.f90) ↔ [scalar_var_kernels_cpu.cpp](../src/equation/scalar/cpu/scalar_var_kernels_cpu.cpp) / [scalar_cpu.cpp:85-154](../src/equation/scalar/cpu/scalar_cpu.cpp).
- **시간이산**: 동일 delta-form 3-factor ADI, 벽법선 방향(PaScaL=y, MPM=z) 마지막 스윕.
- **대류**: 동일 50/50 분할 — 명시 0.5-skew advective in R(인수 `−0.25·Σ`, **spurious 0.5/1.25 없음, 이중계상 없음**) + 0.25 반음해 보존면 in ADI 밴드. 면속도 인덱싱 일치.
- **확산**: 동일 Laplacian stencil·메트릭·dκ̂ 선형화(여기선 0)·방향별 음해 ADI. 내부 밴드 bit 일치.
- **−M·θⁿ 잔차**: [scalar_var_kernels_cpu.cpp:219](../src/equation/scalar/cpu/scalar_var_kernels_cpu.cpp) 존재, PaScaL 367-383(+`dirc·θ·div` 포함)과 구조 동일.
- **벽 BC**: PaScaL mask+ebc 주입 ≡ MPM antisymmetric-ghost+fold. 벽 ghost 메트릭이 2×(`dmx[1]`)지만 antisymmetric ghost가 **정확히 상쇄** → 조립된 벽행 계수 자릿수까지 일치(coeff(T1)=−356914, coeff(T2)=+118029, coeff(T_wall)=+238885 양측 동일).

### 1.2 운동량 (momentum)
ref [module_solve_momentum.f90](../../PaScaL_TCS/src/module_solve_momentum.f90) ↔ [momentum_var_kernels_cpu.cpp](../src/equation/momentum/cpu/momentum_var_kernels_cpu.cpp) / [momentum_cpu.cpp:41-62,171-213](../src/equation/momentum/cpu/momentum_cpu.cpp). 축 매핑: ref-x1(stream,U)→X(dU), ref-x2(wall,V)→Z(dW), ref-x3(span,W)→Y(dV).
- **대류 시간가중**: 동일 skew-split 50/50 반음해 — **완전양해 아님**(Report 18의 "완전양해+1.25× 이중계상"은 옛 const 커널 `momentum_kernels_cpu.cpp` 기준, DHVC 미사용).
- **가변점성 응력**: own×2 + 4개 명시 교차응력 항을 term-for-term 재현. **추가 grad(div u) 없음** — 전치응력 커널 [momentum_cross_stress_cpu.cpp](../src/equation/momentum/cpu/momentum_cross_stress_cpu.cpp)는 `assemble_momentum_var_visc_cpu`가 호출하지 않는 **dead code**(호출 시 spurious ∂ᵢ(∇·u) 유발하므로 안 부르는 게 맞음).
- **−M·qⁿ 잔차**: 존재. own-direction ×2 확산 + own-only `0.5·∂q` self-derivative(교차방향엔 없음 — 알려진 미묘한 버그류 회피).
- **6-블록 속도결합**: blockLdV(M23)/blockLdU(M12+M13) + forward dd*(M21/M31/M32) 모두 축스왑 재현.
- **벽 BC**: PaScaL 명시 ubc/vbc/wbc + jum/jup 마스크 ≡ MPM zero-ghost + `modify_tdma_row_cpu`. no-slip(ZeroGhost)에서 동등.

### 1.3 부력·무차원·격자·Poisson 코어
- **부력** [buoyancy_cpu.cpp:33-34](../src/physics/buoyancy/cpu/buoyancy_cpu.cpp): `Tc=0.5(dx·T+dx·T)/dmx` 면보간, `rhs+=dt·coeff·(Tc−T_ref)`, coeff=1, T_ref=0 = PaScaL `mom.f90:347-352`(Cmt=1, invRhoc=1, a12pera11=0). 크기 인자 불일치 없음.
- **무차원**: ν=√(Pr/Ra), α=1/√(Ra·Pr) = PaScaL Cmu/Ct (`module_global.f90:141-143`).
- **격자**: uniform/tanh 동일식([stretching.cpp:22-45](../src/grid/stretching.cpp)); `dmx[i]=xc[i]−xc[i-1]` = PaScaL `0.5(dx[i-1]+dx[i])` (대수적 항등).
- **Poisson**: divergence MAC-face forward-diff·1/dt 부호 일치; 벽법선 tridiagonal 계수·modified wavenumber(`2(1−cos)/Δ²`) 동등; projection `u−=dt·∇p` 부호·dt·면기울기 일치; zero-mean + singular-mode pin 동일.

---

## 2. ⚠️ 잔여 실차이 (우선순위순)

### 2.1 압력 투영 방식 — incremental vs non-incremental  ★본질적 (Report 19 #3 정정)

> **Report 19 #3 정정**: Report 19는 압력 단계를 "단위밀도에서 PaScaL과 동일(faithful)"로 결론냈으나, 이는 **dPhat/ExtraTerm 항(모두 `(invRhoc−1)` 인수 → 단위밀도서 0)에 한해서만 맞다.** 그와 **별개로** PaScaL의 *incremental projection 커플링*(predictor의 −∇Pⁿ + `P+=dP` 누적)이 MPM에서 누락되어 있으며, 이 커플링은 `(invRhoc−1)`을 인수로 갖지 **않으므로 단위밀도에서도 사라지지 않는다.** 따라서 압력 단계는 단위밀도에서도 PaScaL과 완전히 동일하지 않다.

**PaScaL_TCS** — incremental(압력보정) projection:
```fortran
! predictor RHS에 옛 전압력 기울기 포함 (module_solve_momentum.f90:344-345)
RHS(i,j,k) = RHS(i,j,k) - Cmp*invRhoc*(P(i,j,k) - P(im,j,k))/dmx1(i)
! Poisson은 증분 dP를 풀고, 누적 (module_solve_pressure.f90:818)
P(i,j,k) = P(i,j,k) + dP(i,j,k)
```
→ `u* = uⁿ + dt(RHS − ∇Pⁿ)`, `∇²dP = div(u*)/dt`, `uⁿ⁺¹ = u* − dt∇dP`, `Pⁿ⁺¹ = Pⁿ + dP`. **압력분리 2차.**

**MPM-STD_C** — non-incremental fractional-step:
```cpp
// momentum var 커널 주석 (momentum_var_kernels_cpu.cpp:18): "PRESSURE + BUOYANCY skipped (composed in main)"
//   → 그러나 main.cpp에서도 predictor에 압력항을 합성하지 않음 (buoyancy만 합성)
// Poisson 후 P를 매스텝 덮어씀 (pressure_engine_cpu.cpp:336)
p[(i*n2t+j)*n3t+k] = dp_C_[idx_C(ii,jj,kk,...)] - mean_dp;   // P += 가 아니라 P =
```
→ `u* = uⁿ + dt·RHS`(압력 없음), `∇²P = div(u*)/dt`, `uⁿ⁺¹ = u* − dt∇P`. **압력분리 1차.**

**본질**: 두 방법 모두 uⁿ⁺¹은 발산-자유. 정상상태(또는 통계정상) 속도장은 동일. 그러나 (a) **transient 분리오차 차수가 다르고**(2차 vs 1차), (b) MPM의 저장 "P"는 누적 전압력이 아니라 **매스텝 증분**이다. → Fig7 정상통계 검증엔 무해(실제 1e5 div~1e-13 + Re_δ* 발달 확인됨)하나, **PaScaL의 줄단위 충실 이식은 아니다.**

> 참고: dPhat 의사압력 외삽(`dP0=(dPhat+dP0)/2; dPhat=2dP−dP0`)과 ExtraTerm은 모두 `(invRhoc−1)`/`(1−invRho)` 인수 → 단위밀도서 0. MPM OB 경로는 `inv_rho==nullptr`로 호출되어 이 NOB 기계 전체가 dead code(Report 19 #3 결론은 이 부분에 한해 유효).

### 2.2 mass-flow 보정 ⟨U⟩=0 — MPM에만 있음 (의도적)

[main.cpp:226-227](../apps/dhvc/main.cpp) → [forcing_cpu.cpp:58-75](../src/physics/forcing/cpu/forcing_cpu.cpp) 가 매스텝 bulk-mean U를 빼서 순 연직유량을 0으로 클램프:
```cpp
const double Ub = channel_bulk_velocity_cpu(domain, fields, total_vol);
const double DMpresg = (dt>1e-15) ? (Ub - Ub_target)/dt : 0.0;   // Ub_target=0
u[...] += -dt*DMpresg;   // = −Ub, U에 균일 시프트
```
PaScaL `main.f90`엔 대응 로직 **전무**(grep `flux|bulk|mean.*u|massflow|net` → 해당 없음). PaScaL도 streamwise(X) 주기지만 net-flux 모드를 자유롭게 둔다. → Report 19 #2와 동일: **사용자 의도 유지**. 대칭 T(±0.5)면 ∫θ≈0 → net flux≈0일 때 무해. **ref와 1:1 byte 비교 시 비활성화 필요(플래그화 권장).**

### 2.3 검증 하네스 차이 (스킴 아님 — 정량 비교 시 정렬)

| 항목 | PaScaL | MPM-STD | 영향 |
|---|---|---|---|
| **CFL 레짐** | MaxCFL=1E8 → dt가 dtMax=4e-3에 고정(사실상 fixed-dt) | CFL 제한(MaxCFL=0.4), dt=min(dt_cfl,dt_cap) | dump 비교 전 dt 정렬 필요 |
| **CFL 속도 정의** | 면평균 `½(U_i+U_{i+1})/dx` (`post.f90:207`) | 면값 그대로 `\|u[p]\|/dx` ([cfl_cpu.cpp:26](../src/driver/cpu/cfl_cpu.cpp)) | CFL 분기 binding 시만, 소폭 |
| **초기조건** | 정지 U=V=W=0 + **T 섭동** | 해석적 base-shear jet U(z) + **V/W 속도 섭동** ([main.cpp:179-186](../apps/dhvc/main.cpp)) | **순간장/dump 비교 불일치**, 장시간 통계 무관 |
| **벽 압력 ghost** | 2차 one-sided 외삽 (`pre.f90:455-474`) | 0차 Neumann 복사 ([pressure_engine_cpu.cpp:342-356](../src/equation/pressure/cpu/pressure_engine_cpu.cpp)) | 속도장 영향 미미 |

### 2.4 사소 — jwm/jwp 마스크 (운동량)

[momentum_var_kernels_cpu.cpp](../src/equation/momentum/cpu/momentum_var_kernels_cpu.cpp) `couple_dd_dV_var`(라인 ~598-599)가 PaScaL `solvedW:1030-1031`의 jwm/jwp 벽 마스크를 드롭하고 **dW 벽 halo가 ZeroGhost임에 의존**. `solve_momentum_var_cpu`가 호출 전 `sync_field_cpu(mom.dW, bc.W, …)`로 dW halo를 0으로 만들면 동등 → **`bc.W` Z면이 ZeroGhost Dirichlet인지 assert 권장.**

---

## 3. 권고 (영향순)

| 순위 | 항목 | 조치 | DHVC 효과 |
|---|---|---|---|
| 1 | (2.1) 압력 투영 | 충실 이식 원하면: predictor에 −∇Pⁿ 추가 + Poisson을 증분 dP로 + `P+=dP`. **정상통계만 보면 현행 유지 가능** | 압력분리 1→2차, 누적압력 복원. 정상상태 속도 불변 |
| — | (2.2) mass-flow | **현행 유지**(의도). ref 1:1 byte 비교 시만 플래그 off | net flux≈0이면 무영향 |
| 2 | (2.3) 하네스 | dump 비교 전 MaxCFL/dt + IC를 PaScaL `PARA_INPUT.dat`(MaxCFL=1E8, dt=4e-3, 정지+T섭동)에 정렬 | 순간장 byte 비교 가능 |
| 3 | (2.4) jwm/jwp | `bc.W` Z면 ZeroGhost assert 추가 | 안전성(현 동작은 정상) |

> **한 줄**: DHVC 이산화 코어(에너지·운동량·부력·격자·Poisson 연산자)는 PaScaL_TCS와 충실히 일치한다. 본질적으로 남은 단 하나의 스킴 차이는 **압력 투영이 non-incremental(MPM) vs incremental(PaScaL)** 이라는 점이며 — 이는 정상통계엔 무해하나 줄단위 충실 이식은 아니다 — 그 외 mass-flow·CFL·IC 차이는 의도적이거나 검증 하네스 수준이다.

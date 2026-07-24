# Momentum 구현 충실도 분석 — C++ (MPM-STD_C) vs Fortran 원본 (MPM-STD)

> 작성일: 2026-06-08
> 대상:
> - C++: [/shared/home/wel1come1234/workspace/MPM-STD_C/src/equation/momentum/](../src/equation/momentum/)
> - Fortran(원본/레퍼런스): [/shared/home/wel1come1234/workspace/MPM-STD_main/MPM-STD/src/core/core_momentum.f90](../../MPM-STD_main/MPM-STD/src/core/core_momentum.f90)
> 목적: MPM-STD_C 가 모방 목표로 삼는 Fortran MPM-STD 의 **문제 풀이 방법론(momentum solution methodology)** 을 얼마나 충실히 재현했는지, 어디가 어떻게 다른지 코드 레벨로 정밀 대조.
> 비교 기준: 채널 유동 케이스 (Channel-periodic, z=벽수직, x·y 주기, 상수 물성).

---

## 0. 결론 (Executive Summary)

**정확히 일치하지 않는다.** 전체 **전략(상위 방법론)** 은 충실히 이식되었으나, C++ 는 Fortran 의 일반(가변 물성 + IBM) Beam–Warming ADI 스킴을 **상수 물성 채널 전용으로 의도적으로 단순화한 특수화(specialization) 버전**이다.

- ✅ **일치하는 골격**: 증분(delta-form) ADI 예측자 · 3방향 인수분해 implicit + TDMA · 블록 하삼각 속도 결합 · pseudo-update · 정질량유량(Ub=1) 구동 · 압력 투영 last · z=벽수직.
- ❌ **다른 이산 연산자**: 점성항 형태(라플라시안 vs 전체 응력텐서) · 델타형 잔차항 `−M11·uⁿ` 누락 · implicit 점성계수 처리 · RHS 대류 이산화 · 평균 압력구배 적용 위치.
- ⛔ **미이식 기능**: 가변 물성(μ, 1/ρ) · IBM(H-masking) · 부력 · GPU(CUDA/PaScaL).

난류 통계를 Fortran 과 정량적으로 일치시키려면 아래 **§3.1 / §3.2 / §3.3** 이 1순위 점검 대상이다.

---

## 1. 풀이 전략 (양쪽 공통 — 충실히 모방됨)

두 코드 모두 모멘텀을 **증분(Δu) 델타형 Beam–Warming ADI 예측자 → 블록 하삼각 속도 결합 → pseudo-update → 압력 투영** 순으로 푼다.

### 1.1 타임스텝 내 호출 순서

| 단계 | Fortran ([submodule.f90](../../MPM-STD_main/MPM-STD/src/submodule.f90) 198–292) | C++ ([apps/channel/main.cpp](../apps/channel/main.cpp) 711–807) |
|---|---|---|
| 1 | `solvedU('U')` → du | `mom_U.predict(dt)` → δU |
| 2 | `solvedU('V')` → dv | `mom_V.predict(dt)` → δV |
| 3 | `solvedU('W')` → dw | `mom_W.predict(dt)` → δW |
| 4 | `blockLdU('dV')` (dW→dV) | `block_couple_dV` (dW→dV) |
| 5 | `blockLdU('dU')` (dV,dW→dU) | `block_couple_dU` (dV,dW→dU) |
| 6 | `pseudoupdateUVW` (q+=δ) | `apply_increment` (q+=δ) |
| 7 | (RHS 내부에서 presgrad1) | `forcing.apply_body_force` (dPdx 명시적) |
| 8 | `masscorrection` (투영 직전) | `forcing.correct` (mass_flow) |
| 9 | `pressure` + `projection` (last) | `pressure_solver->solve` (last) |

각 예측자는 1회 완전 ADI 사이클(RHS 구성 → TDMA-z → 전치 → TDMA-x → 전치 → TDMA-y → 증분 update → ghost)이다.

### 1.2 충실히 일치하는 항목

| 항목 | Fortran | C++ | 비고 |
|---|---|---|---|
| 증분(delta-form) ADI 예측자 | Δu | δ | ✔ |
| 3방향 인수분해 implicit + TDMA | (I+dt·Mx)(I+dt·My)(I+dt·Mz)Δd = dt·RHS | 동일 구조 | ✔ |
| 방향 sweep / 주기축 처리 | z→x→y, 주기축 cyclic-TDMA / 벽축 Thomas | periodic-first, wall-z last, cyclic/Thomas | ✔ |
| **블록 하삼각 속도 결합** | dW→dV, {dV,dW}→dU (dW 미보정) | block_couple_dV/dU 동일 | ✔ |
| pseudo-update q+=δ + ghost | ✔ | ✔ | ✔ |
| 정질량유량 구동 (Ub=1) | masscorrection→presgrad1 | mode="mass_flow", Ub=1 | ✔(방식) |
| 압력 투영 last, dt·∇(dP) | ✔ | ✔ | ✔ |
| z = 벽수직 컨벤션 | ✔ | ✔ | ✔ (y↔z swap 불필요) |
| 대류 = 2차 중심, 수치 소산 off | alpha=0.0 | dissipation 없음 | ✔ |

> 메모리에 기록된 **블록 결합**(white-noise IC 에서의 relaminarize 를 막은 핵심 수정)은
> C++ [kernels_cpu.cpp:329-408](../src/equation/momentum/kernels/kernels_cpu.cpp) 가
> Fortran [core_momentum.f90:1612-1785](../../MPM-STD_main/MPM-STD/src/core/core_momentum.f90)
> 의 `blockLdV/blockLdU` 구조(off-diagonal M23, M12, M13 블록)를 정확히 재현한다.

---

## 2. 블록 결합 구조 상세 (일치)

하삼각(lower-triangular) Gauss–Seidel 순서로, 대각 ADI 예측 후·증분 적용 전에 off-diagonal 속도 블록을 가한다. 결합 그래프: **dW → dV; dW,dV → dU; dW 미보정**.

- `block_couple_dV` = M23·dW: `dV -= dt·0.25·(dwm5·∂V/∂z5 + dwm6·∂V/∂z6)`
- `block_couple_dU` = M12·dV + M13·dW (**보정된 dV** 사용):
  `dU -= dt·[0.25·(dvm3·∂U/∂y3 + dvm4·∂U/∂y4) + 0.25·(dwm5·∂U/∂z5 + dwm6·∂U/∂z6)]`

C++ 은 가변 점성 교차응력항을 **상수 μ 가정으로 생략**(대류성 교차항만 유지)하는데, 이는 §3.1 의 점성항 단순화와 동일 맥락이다.

---

## 3. 방법론적 차이 (영향도 순)

### 3.1 점성항 형태 — plain Laplacian vs 전체 응력텐서 **[영향 큼]**

- **Fortran**: 완전 응력발산 `∇·(μ(∇u+∇uᵀ))`. 자기방향에 `×2.0` 수직응력 + 교차응력 전치항(`viscous_u12`, `viscous_u13`) 포함.
  [core_momentum.f90:813-859](../../MPM-STD_main/MPM-STD/src/core/core_momentum.f90)
- **C++**: 단순 라플라시안 `ν∇²u`, 각 방향 1회, ×2 없음, 전치항 없음.
  [kernels_cpu.cpp:22-66](../src/equation/momentum/kernels/kernels_cpu.cpp)
- **등가 조건**: 상수 μ + 발산 0 일 때만 연속식 수준에서 동일. **투영 전 이산 단계에서는 다름.** C++ 은 "발산(연속식) 보정을 투영이 대신한다"는 논리로 전치항을 생략.

### 3.2 델타형 잔차항 `−M11·uⁿ` 누락 **[영향 큼]**

- **Fortran**: implicit 연산자를 uⁿ 에 적용해 RHS 에서 빼서 **진짜 잔차(residual)** 를 만든다.
  → 0.25 implicit 대류 밴드가 증분에 대해 self-consistent(정상상태에서 소멸).
  [core_momentum.f90:862-864](../../MPM-STD_main/MPM-STD/src/core/core_momentum.f90)
- **C++**: 이 항이 **없다.** RHS = full-explicit `ν·dt·L(uⁿ) + 대류(conv_factor=1.0)` 이고,
  0.25 implicit 대류 밴드는 대응 offset 없이 더해진다.
  [momentum_equation.cpp:169-185](../src/equation/momentum/momentum_equation.cpp) 주석이 이 단순화를 명시적으로 인정.

### 3.3 implicit 점성계수 — 균일 CN 0.5 vs 자기방향 강화 **[영향 중]**

- **Fortran**: 자기방향(U의 x)이 물리적 ×2 를 받아 사실상 전체 응력에 CN 적용.
- **C++**: `is_own` 플래그를 `(void)is_own` 으로 **무시**, 모든 방향 `visc_factor=0.5` 균일.
  [momentum_equation.cpp:184-185](../src/equation/momentum/momentum_equation.cpp)
- 저자는 상수 μ 에서 등가라 주장(주석 169–183)하나 구성 방식은 다르다.

### 3.4 RHS 대류 이산화 **[영향 중]**

- **Fortran**: 선형화 분해 `M11Un / M12Vn / M13Wn` (0.25 중심 + 교차속도 대류항 분리).
- **C++**: 보존형 플럭스 발산 `div(u·q)`, conv_factor=1.0 통째.
  [kernels_cpu.cpp:72-83](../src/equation/momentum/kernels/kernels_cpu.cpp)
- 물리 내용(U 를 V·W 가 이송)은 양쪽 다 존재하나 이산형이 다르다.

### 3.5 평균 압력구배 적용 위치 **[영향 중]**

- **Fortran**: `presgrad1` 을 **모멘텀 RHS 내부**에서 차감([core_momentum.f90:875](../../MPM-STD_main/MPM-STD/src/core/core_momentum.f90)),
  보정도 `dp_d`(압력 증분) 포함해 투영 안에서 계산.
- **C++**: dPdx 를 모멘텀 **외부에서 명시적**으로 가함(`apply_body_force`),
  보정은 `(Ub−target)/dt` 단순식 ([apps/channel/channel_forcing.hpp](../apps/channel/channel_forcing.hpp)).
- 둘 다 Ub=1 정질량유량을 유지하지만 보정 공식과 적용 위치가 다르다.

---

## 4. 미이식 기능 (틀린 게 아니라 안 옮긴 것)

| 기능 | Fortran | C++ | 영향 |
|---|---|---|---|
| 가변 물성 (μ, 1/ρ) | `Mu_d`, `invRho_d` 필드 전반 | 상수 ν 하드코딩 | RBC/부력/물성모델 불가 |
| IBM (H-masking) | `is_fluid_face`, solid identity row 전반 | **없음** | 순수 채널/박스 전용 |
| W 벽면 (MAC z-face) | ZMBC fold 로 처리 | `W[wall]=0` 명시적 workaround 필요 | 메모리 기록 이슈 |
| 부력 | W-모멘텀 활성(`Cmt·(T-Ground_T)`) | source hook 만, 채널 미사용 | — |
| 실행 백엔드 | CUDA + PaScaL TDMA (device 전치) | CPU ([kernels_cpu.cpp](../src/equation/momentum/kernels/kernels_cpu.cpp)) | 방법론 아닌 구현 차이 |

---

## 5. 향후 작업 (정량 일치를 위한 우선순위)

Fortran 과 난류 통계를 정량적으로 맞추려면 아래 순서로 점검·수정:

1. **§3.1** 점성항을 전체 응력텐서 형태로 (교차응력 전치항 + 자기방향 ×2 복원).
2. **§3.2** 델타형 잔차항 `−M11·uⁿ` 을 RHS 에 복원하여 implicit 대류 밴드와 self-consistent 하게.
3. **§3.3** implicit 점성계수 자기방향 강화(`is_own` 반영) 재검토.
4. (선택) **§3.4 / §3.5** 대류 이산형·압력구배 적용 위치를 Fortran 과 정렬.

> 비고: 위 1~3 은 상수 μ + 발산 0 에서 연속식 수준 등가이므로, 현재 C++ 가 채널 난류를 *유지*는 한다(메모리: restart turbulent field 유지 확인됨). 다만 **이산 단계의 정확한 일치**(특히 벽 근방 통계·전이 거동)는 보장되지 않는다.

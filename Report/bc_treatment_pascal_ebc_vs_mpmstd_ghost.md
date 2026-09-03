# 경계조건 처리 방식 분석 — PaScaL_TCS(ebc/mbc 명시적 RHS) vs MPM-STD_C(ghost + TDMA 행접기)

- 작성일: 2026-07-24
- 질문: PaScaL_TCS는 경계조건을 ebc/mbc(ubc/vbc/zbc/cbc)로 RHS에 명시적으로 처리하는데, MPM-STD_C는 ghost cell + `modify_tdma_row`로 "간단하게만" 처리한다. **이 단순화가 정확도 문제를 만들지 않는가? 만든다면 어디인가?**
- 근거: 양쪽 코드 전수 추출(PaScaL `module_solve_thermal/momentum/pressure.f90`, MPM `boundary_ops_cpu.cpp`·`grid.cpp`·equation 커널) + 기존 검증 이력(Report/progress 24~36). 핵심 3개 지점(§4.1 버그, 행접기 규칙, 미러 ghost 격자)은 본 보고서 작성 시점에 소스에서 직접 재확인함.

---

## 1. 요약

**결론부터: "간단해서 부정확"은 성립하지 않는다.** 두 방식은 표기가 다를 뿐, **무활주(no-slip)·등온·불투과 벽 조건에서는 항별로 대수적으로 동일한 이산식**을 만든다(§3에서 확산·음해행·대류·압력 각각 증명). 이는 이미 Report/progress/30 §5와 36(Bug 2 재검증)에서 독립적으로 확인되었고, DHVC ±2.5%, RBC ΔT=40 θ_c 참조 +0.05% 재현이라는 실증도 있다.

**그러나 문제가 없는 것은 아니다.** ghost 방식의 등가성은 코드 전반에 흩어진 4개의 불변식(§4.3) 위에 서 있고, 그 불변식이 깨진 사고가 지금까지 3건 있었으며(전부 이 프로젝트에서 발견·수정), **이번 분석에서 그중 1건이 CPU NOB 경로에 아직 살아있음을 확인했다**:

> ★ **`apps/rbc/main_cpu_nob.cpp:129-134`의 T-클리핑이 ghost 평면까지 clamp한다** — Report/progress/31에서 GPU는 내부 셀만 clamp하도록 수정되었으나 CPU는 미수정. 결과: CPU NOB 런의 벽 열전도도가 정확히 절반이 되는 그 버그가 재현된다. 즉시 수정 권고(§4.1).

그 외의 실제 차이: 채널의 U/V zero-ghost는 PaScaL과 다른 **의도된 1차 정확도 벽 배치**(§4.2), W 벽면의 3중 핀은 기능상 등가지만 구조적으로 취약(§4.3-iv), 비영 벽속도(취출·이동벽)와 열유속 벽은 현 구조에 잠재 함정이 있음(§4.4).

---

## 2. 두 방식의 구조

### 2.1 PaScaL_TCS — "벽 위 ghost 좌표 + 마스크 + 명시적 재주입"

(y=x2가 벽수직. 이하 벽수직 격자간격을 dz로 표기해 MPM과 통일)

- **ghost는 존재하지만 그 좌표가 벽 위로 접혀 있다**: `dx2_sub(0)=0` (`module_mpi_subdomain.f90:237`) → 벽인접 중심간 거리 `dmx2(1)=0.5·dx2(1)`=**반칸**. ghost 중심 = 벽면. ghost 값 = Dirichlet 벽값 그대로(`T(:,0,:)=TBCbt`, `module_solve_thermal.f90:120-128`). 비주기 y방향은 `MPI_PROC_NULL`이라 halo 교환이 ghost를 덮어쓰지 않음.
- **음해 행렬**: 벽측 비대각을 0/1 마스크로 제거하는 것이 전부 — `eAPJ*dble(jep)`, `eAMJ*dble(jem)` (`module_solve_thermal.f90:520-521`), 모멘텀은 `jup/jum/jvp/jvm/jwp/jwm` (`module_solve_momentum.f90:533-534` 등).
- **명시적 RHS**: 행렬에서 떨어뜨린 벽 결합 × 기지의 벽값을 `ebc/ubc/vbc/zbc`로 재주입 — 확산뿐 아니라 **대류 계수(벽면 V)와 교차 점성항까지** 포함 (θ: `:322-334`, U: `:358-371`, W: `:1073-1087`). 즉 "마스크로 뺀 것을 ebc로 되돌려 넣는" 상보 구조이며, 보정은 **명시적 RHS에만** 있고 행렬은 마스크 외 수정 없음.
- **벽수직 속도 V**: 벽면이 V-노드와 일치. 벽 소유 랭크는 `j_indexS=2`로 벽면 행을 **해석에서 구조적으로 제외**하고 `V(i,0)=V(i,1)=VBC` 고정, `dV(0:1)=0` (`module_solve_momentum.f90:121-138, 920-921`).
- **압력**: 발산에서 벽 V-플럭스를 마스크로 빼고 `cbc=VBC/dx2`로 재가산(`module_solve_pressure.f90:198-203`); y-TDMA는 벽 leg 순수 drop(Neumann, `:609-611`); dP/dPhat ghost는 2차 외삽(`Pbc_a/Pbc_b`, `:462-466`).

### 2.2 MPM-STD_C — "미러 ghost + 행접기(fold)"

(z가 벽수직, `h=kHaloWidth=1`)

- **ghost 좌표는 벽 밖 미러 위치**: `face_global(g<0)=2·gf[0]−gf[1]` ([grid.cpp:43-48](../src/grid/grid.cpp#L43-L48)) → ghost 중심 = 벽−dz/2, 첫 내부 중심 = 벽+dz/2, **`dmx3[h]=dz` 전칸 간격**. 반칸 정보는 좌표가 아니라 **ghost 값**이 담는다.
- **ghost 값 채움** ([boundary_ops_cpu.cpp:26-60](../src/core/cpu/boundary_ops_cpu.cpp#L26-L60)):
  - Dirichlet + Antisymmetric: `ghost = 2·v_wall − φ₁` (T, RBC의 U/V) — 벽면에서 정확히 v_wall이 되도록 하는 2차 미러
  - Dirichlet + ZeroGhost: `ghost = v_wall` (W, 채널의 U/V) — 미러 위치에 벽값을 그대로 놓음
  - Neumann: `ghost = φ₁ ± v` (P, v=0이면 zero-gradient 복사)
- **음해 행 접기** `modify_tdma_row_cpu` ([boundary_ops_cpu.cpp:97-108](../src/core/cpu/boundary_ops_cpu.cpp#L97-L108)): 벽측 비대각을 0으로 만들되, ghost 증분 규칙에 맞게 대각으로 접는다 —
  - Antisymmetric(δghost=−δφ₁) → `B −= A`
  - ZeroGhost(δghost=0) → drop만
  - Neumann(δghost=+δφ₁) → `B += A`
  **RHS(D)는 건드리지 않는다** — 벽값은 명시적 잔차와 −M·qⁿ 팩 안의 ghost를 통해서만 들어간다. ebc류 항은 코드 전체에 존재하지 않음(주석으로 명시: `scalar/kernels.hpp:113-116`, `momentum_var_kernels_cpu.cpp:19` — "레퍼런스의 ebc/jem/jep 기계를 ghost+fold로 대체").
- **W 벽면(엇갈림)**: 하부벽 W-면이 내부 인덱스 k=h에 옴 → ① z-스윕 벽면 행을 항등행으로 덮음(`momentum_cpu.cpp:99-104/149-154`), ② 블록 커플링 소비 전 `dW` 핀(`zero_dw_lower_wall_face`, Report/progress/27), ③ 속도 갱신 후 필드 핀(`rbc::zero_w_wall`).
- **압력**: z-TDMA 벽 leg 순수 drop(`pressure_engine_cpu.cpp:743-801`); 발산은 W=0 핀 덕에 특별항 불필요; dPhat ghost는 Neumann 복사로 채워 ExtraTerm 벽 leg가 자동 0(PaScaL의 jvp/jvm 마스크와 같은 효과).

**요약 대비**: PaScaL은 "좌표를 접고(반칸) 값은 벽값 그대로, 빠진 결합을 RHS에 손으로 재주입"; MPM은 "좌표는 그대로(전칸) 값을 접고(미러), 빠진 결합을 대각에 접어 넣음". 다른 표기의 같은 수학이다 — 아래에서 증명한다.

---

## 3. 항별 등가성 증명 (무활주·등온·불투과 벽)

기호: 첫 내부 셀 φ₁, 벽값 φ_w, 균일 격자간격 dz, 확산계수 묶음 c ≡ κ̂/dx₃ (셀 나눗셈 포함).

### 3.1 확산 — 명시적 벽면 플럭스

- PaScaL (반칸): `(φ₁ − φ_ghost)/dmx2(1) = (φ₁ − φ_w)/(dz/2) = 2(φ₁−φ_w)/dz`
- MPM (전칸+미러): `(φ₁ − φ_ghost)/dmx3[h] = (φ₁ − (2φ_w−φ₁))/dz = 2(φ₁−φ_w)/dz`

**동일.** 미러 ghost의 "2"가 반칸의 "1/2"를 정확히 재현한다. (역으로, 이 등가성이 ghost 값에 실려 있기 때문에 ghost를 훼손하면 — §4.1 — 곧바로 벽 플럭스가 절반이 된다.)

### 3.2 확산 — 음해(δ-form) 벽인접 행

- PaScaL: 비대각 `am = −c/dmx2(1)·mask = 0`, 대각은 반칸 덕에 자연히 `+c/(dz/2) = +2c/dz`. (벽값은 고정 → δ=0 → drop이 정확.)
- MPM: 밴드 조립은 `A = −c/dz`, `B ∋ +c/dz`로 만든 뒤 fold `B −= A` → 대각 `+2c/dz`, `A=0`. (δghost = δ(2φ_w−φ₁) = −δφ₁이므로 `A·δghost = −A·δφ₁`을 대각으로 이항한 것 — fold가 곧 ghost 갱신 규칙의 정확한 대수적 표현.)

**동일.** 잔차(−M·φⁿ)까지 전개해도 양쪽 모두 벽면 기여가 `−2c(φ₁−φ_w)/dz`로 일치한다(부록 A).

### 3.3 대류 — 벽면 속도 계수

- PaScaL: ebc에 `0.25·V_wall/dmx2·φ_w`(θ), `−0.25·dudy·V_wall`(U의 교차 대류) 등을 명시 포함. 무활주·불투과면 V_wall=0이라 수치상 0이지만 식은 살아 있음.
- MPM: 벽면 대류 플럭스 = W_face·(벽면 보간값). W_face는 0으로 핀; 벽면 보간값은 antisym ghost 덕에 `0.5(φ₁+φ_ghost) = φ_w` **정확히** 벽값. 음해 쪽은 fold가 대류 계수가 포함된 비대각 전체에 적용되는데, δghost=−δφ₁ 규칙은 비대각이 어떤 물리(확산+대류)를 담고 있든 성립하므로 역시 정확.

**V_wall=0에서 동일.** (비영 벽속도의 경우는 §4.4 — PaScaL은 식이 살아 있고 MPM은 핀이 0으로 고정하므로 현재 미지원.)

### 3.4 NOB 변물성 선형화(dκ̂) — 유일하게 남은 마스크

PaScaL은 벽인접 행 대각의 변물성 deferred 항도 `dble(jep/jem)`으로 마스크한다(`module_solve_thermal.f90:514-515`). 이 항은 ghost+fold로 자연 표현이 안 되는 유일한 항이라, MPM도 여기만은 **진짜 마스크를 유지**했다: `scalar_var_stencil.hpp:68-70` `if (wall_lo) dkPP_lo = 0;` (물리 벽 소유 랭크의 벽면에서만 발동, rank 이음새에서는 발동 안 함 — D5 감사에서 검증). **동일.**

### 3.5 압력 Poisson

| 요소 | PaScaL | MPM | 판정 |
|---|---|---|---|
| 벽 Neumann (y/z-TDMA) | 벽 leg `fft_amj/apj=0` 순수 drop (`:609-611`) | 벽 leg `a_val/c_val=0` 순수 drop (`pressure_engine_cpu.cpp:757-760`) | 동일 |
| 발산의 벽 플럭스 | V-플럭스 마스크 + `cbc=VBC/dx2` 재가산 | W=0 핀 → 발산이 자동으로 옳음 | VBC=0에서 동일 |
| 평균 모드 고정 | (1,1,1) 모드 핀 | (0,0) 모드 핀 + 전역 평균 제거 | 상수차이로 동일 |
| NOB ExtraTerm 벽 leg | `dble(jvp/jvm)` 마스크 (`:226`) | dPhat에 Neumann ghost 적용 → 벽 leg `(ph[c]−ph[zm])/dmx = 0` 자동 | 동일 (공개판 PaScaL의 마스크 누락 버그를 MPM은 ghost 방식으로 자연 회피) |
| dP/dPhat ghost 값 | 2차 외삽(`Pbc_a/Pbc_b`) | Neumann 복사(`ghost=p₁`) | **소비처가 벽 leg를 0으로 만드는 목적에는 둘 다 충분** — 벽 leg가 마스크(PaScaL)나 zero-gradient(MPM)로 소거되므로 결과 동일 |

### 3.6 벽수직 속도의 벽면 행

- PaScaL: `j_indexS=2`로 벽면 행을 해석 배열에서 **구조적으로 제외** + `dV` 0 고정.
- MPM: 전 길이로 풀되 벽면 행을 **항등행(A=C=0,B=1,D=0)** 으로 덮음 → δW_face=0이고 삼중대각 전개상 이웃 행에 영향 없음(비대각 0). + dW 핀 + 필드 핀.

**정확 산술에서 동일.** 단, "제외"는 실수할 수 없는 구조이고 "항등행+핀×3"은 하나라도 빠지면 깨지는 구조다(§4.3-iv — 실제로 Report/progress/27 사고가 그것).

### 등가성 요약

| 항 | 등가? | 조건/근거 |
|---|---|---|
| 확산(명시·음해) | ✅ 대수적 동일 | antisym ghost + fold ↔ 반칸 + 마스크+ebc (§3.1-3.2) |
| 대류 벽 계수 | ✅ | V_wall=0; antisym 보간이 벽면값 정확 재현 (§3.3) |
| NOB dκ̂ | ✅ | MPM도 동일 마스크 유지 (§3.4) |
| 압력 전 요소 | ✅ | VBC=0 (§3.5) |
| V/W 벽면 행 | ✅ (기능) | 구조적 견고성은 PaScaL 우위 (§3.6, §4.3) |
| 채널 U/V (zero-ghost) | ❌ **의도적 차이** | 1차 벽 배치 — §4.2 |
| CPU NOB T-clip | ❌ **버그** | ghost 오염 — §4.1 |

독립 검증: Report/progress/30 §5(마스크 생략은 무활주 벽에서 항별 등가 — "이중 안전장치"였음), 36(Bug 2 해당 없음 재확인 + CPU/GPU 스모크 궤적 일치 div~1e-15), 그리고 momentum 검증에서 행접기 규칙을 ghost 정책과 일치시키자 벽 오차가 3900× 감소한 이력 자체가 "일치했을 때 정확하다"는 실증이다.

---

## 4. 실제 문제 목록 (우선순위순)

### 4.1 ★ 살아있는 버그 — CPU NOB T-클리핑이 ghost를 오염 (즉시 수정 권고)

[main_cpu_nob.cpp:129-134](../apps/rbc/main_cpu_nob.cpp#L129-L134):

```cpp
if (DeltaT > 0.0) {
  real_t* t_ptr = fields[core::Var::T].data();
  const std::size_t nf = n_total[0]*n_total[1]*n_total[2];   // ← ghost 포함 전체
  for (q...) t_ptr[q] = clamp(t_ptr[q], -0.5, +0.5);
}
```

- 열벽 antisym ghost는 `2·0.5 − T₁ ≈ 0.5155`여야 하는데 매 스텝 0.5로 잘린다. 다음 스텝의 `update_properties_cpu`(110행)와 `solve_scalar_nob_cpu`(112행)는 T-sync(114행) **이전에** 이 오염된 ghost를 읽는다.
- 효과는 Report/progress/31에서 정량화된 그대로: 유효 Dirichlet 점이 벽 밖 반칸으로 밀림 → **벽 열전도도 정확히 절반**, 유효 ΔT −3%, θ_c 하향 편향. GPU는 31에서 내부 셀 전용 3D 커널(`k_clip_T_gpu`, `main_gpu_nob.cu:36-49`)로 수정 완료 — **CPU만 미수정 상태**.
- 영향 범위: CPU NOB 검증 런 전부 + CPU↔GPU A/B 비교의 신뢰성(ΔT>0 케이스). OB/DHVC는 clip이 꺼져 있어 무관.
- **수정안**: GPU와 동일하게 내부 셀만 clamp(권장, 3중 루프), 또는 clamp 직후 `sync_field_cpu(T)` 재호출. 1~5줄 수정, "계산 로직 동결" 원칙의 예외가 아니라 **이미 GPU에 적용된 버그 수정의 CPU 이식**이다.

### 4.2 채널 U/V zero-ghost — PaScaL과 다른 의도적 1차 벽

zero-ghost는 `U=0`을 벽면이 아니라 **벽 밖 dz/2의 미러 ghost 중심**에 놓는다 → 벽면에서 U≈U₁/2 (무활주 O(dz) 위반), 스킴이 보는 벽전단 계수는 `U₁/dz`로 PaScaL(반칸, `2U₁/dz`)의 절반. PaScaL·antisym이 2차인 곳에서 채널만 1차다.

이것은 실수는 아니고 팀의 의도적 선택이다(경위: antisym이 `+ν/dz²·U₁`의 추가 벽 감쇠를 M·Uⁿ에 접어 넣어 아임계 천이를 죽였고, zero-ghost + drop 조합이 채널 검증을 통과) — 그러나 **수치해석적으로는 antisym 쪽이 PaScaL과 동일한 "물리적으로 옳은" 벽전단**이므로, "왜 zero-ghost일 때만 천이가 되는가"는 미해결 질문으로 남아 있다. 벽전단·Re_τ에 O(dz) 편향이 들어갈 수 있으므로:

- **권고**: 채널 격자수렴 A/B — 같은 난류 restart 필드에서 zero vs antisym × 격자 2단으로 Re_τ·WSS를 비교(해석해 WSS=3νU_b/h 기준). RBC/DHVC는 antisym이므로 이 문제와 무관.

### 4.3 구조적 취약성 — 등가성이 4개의 분산된 불변식에 의존

ebc 방식은 장황하지만 **국소적**이다: 벽 물리 전체가 솔버 루틴 한 곳에 눈에 보이게 있고, 다른 코드가 건드릴 수 없다. ghost 방식은 간결하지만 **전역적**이다: 다음 4개 불변식이 서로 다른 파일에서 각자 지켜져야 등가성이 성립한다.

| # | 불변식 | 깨졌던 사고 |
|---|---|---|
| i | ghost 채움 정책 ↔ 행접기 방향 일치 (antisym↔`B−=A`, zero↔drop, Neumann↔`B+=A`) | 과거 "모든 Dirichlet에 `B+=A`" → 의도치 않은 Neumann 벽, 무활주 감쇠 소실; 수정 후 벽 오차 3900×↓ |
| ii | 모든 필드 변경 직후 sync(halo+ghost) — 소비자보다 먼저 | 공개판 PaScaL의 dPhat ghost 미갱신(Bug 9)과 동형; MPM은 sync 규율로 회피 |
| iii | **전체 배열을 훑는 어떤 연산도 ghost를 변경하면 안 됨** | T-클리핑 사고 — GPU에서 발견·수정(R31), **CPU에 아직 잔존(§4.1)**. 같은 유형 사고가 두 번째 |
| iv | 엇갈림 W 벽면의 3중 핀(항등행+dW핀+필드핀)이 모두 제자리에 | 하부벽 dW 누설 → 1/dmx₃(~300×) 증폭 스파이크 (R27, 수정) |

사고 3건 중 2건(i, iv)은 수정 후 재발 방지가 코드 구조에 내장되었지만, (iii)은 "누구든 T를 clamp/필터/후처리하는 코드를 추가하면 재발"하는 유형이다. **권고**: 디버그 빌드용 ghost 무결성 assert — 스텝 말미에 벽 소유 랭크에서 `|T_ghost − (2T_w − T₁)| < ε` 검사(한 함수, 회귀 스크립트에 편입). 이 한 개의 검사가 불변식 ii·iii을 동시에 감시한다.

### 4.4 잠재 함정 (현재 케이스에는 무해, 확장 시 발동)

- **열유속(Neumann T) 벽**: `fill_neumann`은 `ghost = φ₁ ± v`로 **원시 값**을 더한다 — 기울기 g를 주려면 사용자가 `v = g·dmx`를 직접 환산해야 하며 어디에도 문서화되어 있지 않다. 현재는 v=0(압력)만 사용되어 무해. 등온이 아닌 열유속 RBC/채널 도입 시 함정.
- **비영 벽속도(취출·흡입·이동벽)**: PaScaL은 ebc(대류 벽 계수)·cbc(벽 질량 플럭스)에 식이 살아 있으나, MPM은 W 벽면을 상수 0으로 핀(`zero_w_wall`)하고 발산도 그에 의존 → 현 구조로는 표현 불가. 도입하려면 핀 값을 BC 값으로 일반화하고 발산에 cbc 상당 항이 필요.
- **물성 ghost 평가(고차 미세 차이)**: PaScaL은 ghost 물성을 μ(T_w)로, MPM은 μ(2T_w−T₁)로 평가 → 벽면 보간 후 차이는 O((T₁−T_w)²·μ″)의 2차 소량. 글리세롤처럼 물성 변화가 큰 유체에서도 벽인접 ΔT가 작으면 무시 가능하나, 완전 등가는 아님을 기록해 둔다.

### 4.5 CPU/GPU 비대칭 (정확도 감사 관점)

- T-클리핑 수정: GPU만 (§4.1) — **해소 필요**
- 블록 커플링 완화 θ(`MPMSTD_BLKRELAX`): GPU만 — CPU에서 같은 한계 안정성 재점화가 원리상 가능(현재 CPU 런 조건에서는 미발현)
- OB 스칼라 스킴: CPU=완전 명시 보존형, GPU=var-스텐실 준음해 — BC 문제는 아니지만 A/B 비교 시 혼동 요인

---

## 5. 실증 증거 — "ghost 방식이라서 부정확"이라는 신호는 없다

| 증거 | 결과 |
|---|---|
| momentum 검증 (행접기 수정 후) | 벽 집중 오차 3900× 감소 — 규칙 일치 시 정확함의 직접 증거 |
| Report/progress/36 CPU/GPU 스모크 (32×32×48, 3000스텝) | 궤적 사실상 동일, div ~1e-15 |
| DHVC (OB) | 참조 대비 ±2.5% (Ra=1e5 +1.6%, 1e6 −2.5%) |
| RBC ΔT=40 (Hormes) | θ_c = 0.1383, 참조 +0.05% |
| RBC Pan(글리세롤) Nu | dt→0 외삽 9.4–9.6 vs 참조 9.61 ✅ (T-clip 수정 후 λ-sum=1/Nu 자기일치 회복 — 벽 열플럭스 회로가 닫혀 있음의 증거) |
| RBC Pan θ_c | 평형값 ~0.065 vs 0.071 (−13%) — **미해결이나, 벽 플럭스 자기일치가 회복된 상태의 잔차이므로 BC 이산화가 아니라 해상도(참조의 Batchelor 격자 기준 미충족)·평형시간 쪽에 귀속** |

주의할 교훈: 초기의 "참조에 가까웠던" θ_c 0.066은 **T-클리핑 버그와의 부분적 오차 상쇄**였다(Report/progress/31). 즉 이 계보의 사고들은 "ghost 방식이 원리적으로 부정확"해서가 아니라 "ghost가 훼손되면 조용히, 그럴듯한 값으로 틀린다"는 성질 때문에 위험했다 — §4.3의 감시 장치가 필요한 이유다.

---

## 6. 권고 (우선순위순)

1. **[즉시, 1~5줄] CPU T-클리핑 내부-셀 한정 수정** (§4.1) — GPU `k_clip_T_gpu` 수정의 CPU 이식. 수정 후 CPU/GPU NOB 스모크 A/B로 확인.
2. **[단기] ghost 무결성 assert + λ-sum=1/Nu 자기일치 검사를 회귀 스크립트에 편입** (§4.3) — 불변식 ii/iii의 상시 감시. 추상화 완화 계획(Report/convenience/01)의 3케이스 회귀와 같은 스크립트에 넣으면 된다.
3. **[단기] 본 보고서 §3의 등가성 표를 `docs/`에 이관** — "ghost+fold ↔ ebc/mbc 사전"으로. 신규 사용자가 PaScaL 논문/코드와 대조할 때의 해독표이며, 추상화 완화 4단계(문서화)와 접합.
4. **[중기] 채널 zero-ghost vs antisym A/B** (§4.2) — Re_τ·WSS 격자수렴으로 O(dz) 벽전단 편향 정량화, "antisym이 천이를 죽인" 원인 재조사(당시 행접기 버그(i)와의 교란 가능성 있음 — 지금은 fold가 옳으므로 재현되지 않을 수 있다).
5. **[확장 시] 열유속 벽·비영 벽속도 도입 전에 §4.4의 두 함정 해소** — `fill_neumann`의 기울기 환산 내장, W 핀의 BC 값 일반화 + 발산 cbc 항.

---

## 부록 A — δ-form 잔차의 벽면 기여 등가 유도 (θ, 하부벽)

PaScaL (반칸, 마스크+ebc): 벽인접 행의 잔차 중 벽면 기여
= −[대각의 벽면 성분]·θ₁ + ebc = −(c/(dz/2))·θ₁ + (c/(dz/2))·θ_w = **−2c(θ₁−θ_w)/dz**

MPM (전칸, 미러 ghost): −[A·θ_ghost + B의 벽면 성분·θ₁] = −[(−c/dz)(2θ_w−θ₁) + (c/dz)θ₁] = **−2c(θ₁−θ_w)/dz** ∎

음해 증분 쪽: PaScaL은 δθ_wall=0이므로 마스크(drop)가 정확; MPM은 δθ_ghost=−δθ₁이므로 `B−=A`가 정확. 두 경우 모두 벽면 대각 = +2c/dz. ∎

대류 포함 일반화: 비대각 A가 확산+대류 어떤 계수든, MPM의 fold는 "δghost=−δφ₁" 규칙의 대수적 이항이므로 항 구성과 무관하게 성립. PaScaL의 drop+ebc는 "벽값 고정" 규칙의 이항. 두 규칙은 각자의 ghost 좌표 규약(벽 위 vs 미러) 아래에서 같은 연속 문제의 같은 이산화를 표현한다. ∎

## 부록 B — 근거 파일

- PaScaL_TCS: `module_solve_thermal.f90` (ebc :322-334, 마스크 :513-525, 벽값 :117-128), `module_solve_momentum.f90` (ubc :358-371, vbc :717-728, zbc :1073-1087, j_indexS :121-138·920-921), `module_solve_pressure.f90` (cbc :198-229, Neumann drop :603-613, Pbc :462-466), `module_mpi_subdomain.f90` (dx2(0)=0 :237, dmx2 :242, jC/jS_BC :304-374)
- MPM-STD_C: `src/core/cpu/boundary_ops_cpu.cpp` (ghost 채움 :26-60, 행접기 :82-118), `src/grid/grid.cpp` (미러 ghost 면 :43-48, dmx :70-77), `src/boundary/problem_loader.cpp` (정책 기본값 :173-186), `src/equation/scalar/scalar_var_stencil.hpp` (dκ̂ 마스크 :68-70), `src/equation/momentum/cpu/momentum_cpu.cpp` (W 항등행 :99-104·149-154, dW 핀 :197-206), `src/equation/pressure/cpu/pressure_engine_cpu.cpp` (Neumann drop :743-801, ExtraTerm :385-434), `apps/rbc/main_cpu_nob.cpp` (★T-clip :129-134), `apps/rbc/main_gpu_nob.cu` (수정된 clip :36-49)
- 검증 이력: `Report/progress/27, 30, 31, 32, 36` 및 26 §3.2, 28 §4.2-4.3

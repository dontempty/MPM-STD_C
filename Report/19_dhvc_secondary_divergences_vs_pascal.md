# DHVC 부차 항목 ↔ PaScaL_TCS 차이 정리 (2026-06-15)

> ref(ground truth): `/shared/home/wel1come1234/workspace/PaScaL_TCS/src/` (Fortran)
> 대상: `/shared/home/wel1come1234/workspace/MPM-STD_C/` (DHVC, unit-property)
> 범위: 운동량 **대류** 핵심 fix(Report 18 #1 + 교차밴드 이류속도) 이후 남은 부차 3항목.
> 방법: 양측 file:line 직접 대조. 관련: [18_mpmstd_c_vs_pascal_tcs.md](18_mpmstd_c_vs_pascal_tcs.md).

---

## 0. 결론 (요약)

| 항목 | DHVC(단위밀도)에서 차이? | 판정 | 비고 |
|---|---|---|---|
| (1) 부력 온도 위치 | **있었음** (밀도 무관) | **✅ 수정됨** | 셀중심 T[c] → PaScaL 면보간 Tc로 교체 (2026-06-15) |
| (2) mass-flow 보정 (DMpresg) | **있음** (추가 연산) | **divergent (의도적)** | C++가 추가, PaScaL엔 없음. net flux≈0이면 무해. 사용자 의도 |
| (3) 압력 dPhat 예측자 | **없음** | **faithful @ 단위밀도** | PaScaL dPhat 항 전부 ×(invRhoc−1)=0 → C++ pure-Laplacian과 정확히 동일 |

→ **DHVC에서 진짜 차이는 (1)·(2)뿐. (3)은 단위밀도에서 PaScaL과 수학적으로 동일** (예측자는 가변밀도=NOB-RBC에서만 의미).

---

## (1) 부력 온도 위치 — 면보간 vs 셀중심 ✅ 수정됨 (2026-06-15)

> **수정 완료**: OB 부력(`add_buoyancy_force_gpu/cpu`)을 아래 PaScaL 면보간 `Tc`로 교체. 시그니처에 `Domain` 추가(중력축 메트릭), OB 호출부 7곳 갱신, 빌드 OK. NOB-RBC 부력은 미변경(별도).


**PaScaL_TCS** `module_solve_momentum.f90:348-352` — 부력을 x(중력/연직)-운동량 RHS에 더하되, 셀중심 T를 **U-면(i−1/2)으로 체적가중 보간**:
```fortran
Tc = 0.5d0*(dx1(im)*T(i,j,k) + dx1(i)*T(im,j,k))/dmx1(i)
RHS(i,j,k) = RHS(i,j,k) + Cmt*(Tc + a12pera11*Tc**2.*DeltaT)*invRhoc
```

**MPM-STD_C** `buoyancy_gpu.cu:13` (= `buoyancy_cpu.cpp:25`) — **셀중심 T[c]를 그대로** 사용, 면보간 없음:
```cpp
rhs[c] += dt * coeff * (T[c] - t_ref);
```

**차이의 본질**: U-운동량 DOF는 U-면(i−1/2)에 사는데, PaScaL은 그 면으로 보간한 Tc(셀 i와 i−1의 가중평균)를 쓰고, C++는 셀 i의 T만 쓴다 → **O(dx) 반셀 위상 시프트**. 부력 강제 위치가 반셀 어긋남. **밀도와 무관**하므로 단위밀도 DHVC에서도 그대로 존재하는 진짜 발산.
- 단, C++ 주석(`buoyancy_cpu.cpp:4-8`)은 이 셀중심 방식이 "검증된 src/ 스킴(momentum source_name='T')과 일치"한다고 명시 — **자매 MPM-STD 관례를 의도적으로 따른 것**이지 PaScaL 관례는 아님.
- **수정안(원하면)**: `Tc = 0.5*(dxg[bo-1]*T[c] + dxg[bo]*T[c-gstride])/dmxg[bo]` (gravity-axis 메트릭) 후 `rhs[c]+=dt*coeff*(Tc-t_ref)`. GPU·CPU 동일. 중력축 메트릭(DHVC=x → dx1/dmx1)을 커널에 전달 필요.

---

## (2) mass-flow 보정 (DMpresg) — C++ 추가, PaScaL 없음 ⚠ divergent (의도적)

**PaScaL_TCS**: DHVC에 mass-flow/bulk/DMpresg 강제 **전무** (전 소스 grep: `dmpresg|mass.*flow|bulk|net.*flux|dpdx` 해당 없음 — `ddpdx*`는 dPhat 기울기, 무관). DHVC는 닫힌 부력구동 대류: 벽 무활(z), x·y 주기, 부과 dP/dx=0.

**MPM-STD_C** `forcing_gpu.cu:81-97`, 호출 `main_gpu.cu:531` / `main.cpp:227` — 매 스텝 순 연직유량을 0으로 강제:
```cpp
const double Ub      = channel_bulk_velocity_gpu(domain, fields, total_vol);
const double DMpresg = (dt>1e-15) ? (Ub - Ub_target)/dt : 0.0;   // Ub_target = 0
k_shift_u<<<...>>>(U, ..., -dt*DMpresg);   // U에 균일 시프트
net_dpdx += DMpresg;
```

**판정**: PaScaL에 없는 추가 연산. C++ 주석(`main.cpp:200,225`)의 근거 — DHVC는 x=연직=주기라 순 부력 ∫θ를 잡아줄 벽이 없어 평균 연직류가 표류할 수 있어 안전장치로 넣음. 대칭(antisymmetric T=±0.5)이면 ∫θ≈0 → **net flux≈0일 때 무해**(≈0을 빼므로). **사용자 의도 추가**이므로 제거하지 않음. ref와 1:1 byte 비교 시에는 비활성화 필요(플래그화 권장).

---

## (3) 압력 dPhat 예측자 — 단위밀도에선 차이 없음 ✅ faithful

**PaScaL_TCS** `module_solve_pressure.f90` — 가변밀도 2차 예측자 보정:
```fortran
! 속도보정 (807-817): pure gradient + (invRhoc-1)·grad(dPhat)
U = U - dt*Cmp*( (dP-dP)/dmx1 + (invRhoc-1.d0)*(dPhat-dPhat)/dmx1 )
! 예측자 (820-821)
dP0   = (dPhat + dP0)/2.d0
dPhat = 2.d0*dP - dP0
! Poisson RHS ExtraTerm (221): ((1-invRho2)*ddpdx2 - (1-invRho1)*ddpdx1)/dx1 + ...
```

**MPM-STD_C** `pressure_gpu.cu:15-28` (DHVC가 호출하는 `solve_pressure_gpu`→OB `eng.solve()`) — 순수 상수계수 P-Laplacian 투영:
```cpp
U -= dt*Cmp*(P[i]-P[im])/dmx1;   // dPhat·예측자·ExtraTerm 전부 없음
```

**판정 — 차이 없음 @ 단위밀도**: PaScaL의 dPhat 관련 항은 **모두 `(invRhoc−1)` 또는 `(1−invRho)`를 인수**로 갖는다. DHVC는 invRho≡1 → 이 인수들이 **전부 0** → dPhat 속도보정도, ExtraTerm도 소멸 → PaScaL이 **C++의 pure-Laplacian 투영으로 정확히 환원**. 즉 단위밀도 DHVC에서 압력 단계는 **PaScaL과 수학적으로 동일**, 발산 아님.
- 예측자는 **가변밀도(NOB-RBC)에서만** 작동. 거기선 GPU `solve_nob`이 `dPhat=2dP−dP0` 대신 단순복사(`pressure_engine_gpu.cuh:1354`)라 차이 존재 — 하지만 **DHVC 무관**.

---

## 4. 권고 (영향순)

| 순위 | 항목 | 조치 | DHVC 효과 |
|---|---|---|---|
| 1 | (1) 부력 T 면보간 | `Tc` 체적가중 보간으로 교체 (GPU+CPU) | O(dx) 반셀 부력 위상오차 제거, PaScaL 충실 |
| — | (2) mass-flow | **현행 유지**(사용자 의도). ref 1:1 비교 시만 플래그로 off | 무영향(net flux≈0) |
| — | (3) 압력 | **조치 불요** (단위밀도에서 이미 동일) | 없음 |

> **한 줄**: 운동량 대류는 이제 PaScaL 충실. 남은 부차 차이는 **(1) 부력 셀중심 T(진짜 O(dx) 발산, 수정 가능) + (2) 의도적 mass-flow 안전장치**뿐이며, **(3) 압력은 단위밀도에서 PaScaL과 동일**해 차이가 아니다.

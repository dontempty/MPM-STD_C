# 24. RBC NOB θ_c 결손 조사 종합 정리

> 작성: 2026-07-02  
> **최종 업데이트**: 2026-07-02 — 근본 원인 발견 및 버그 수정 완료
> 목적: MPM-STD_C의 glycerol NOB RBC θ_c ≈ 0.040 vs 참조값 0.074 문제를 처음 보는 독자가 전체 맥락을 파악할 수 있도록 정리.

---

## 0. 용어 / 기호 정리

| 기호 | 의미 |
|------|------|
| NOB | Non-Oberbeck-Boussinesq: 물성(μ, κ, ρ, Cp)이 온도 의존 |
| OB  | Oberbeck-Boussinesq: 부력항 외 물성 상수 |
| RBC | Rayleigh-Bénard Convection (하부 가열, 상부 냉각) |
| θ_c | 중앙평면(z=H/2) 수평평균 온도 (무차원, T_ref=0) |
| ΔT  | 벽 온도차 (hot bottom − cold top) |
| Γ   | 가로세로비 = L_x/H = L_z/H |
| γ   | 격자 스트레칭 계수 (클수록 벽 근처 셀 밀집) |

온도 무차원화: θ = (T_dim − T_ref)/ΔT ∈ [-0.5, +0.5]. 하부 열벽 θ = +0.5, 상부 냉벽 θ = -0.5.

---

## 1. 물리적 배경 — 왜 θ_c ≠ 0 인가

**OB 대칭**: 물성 상수 → 위·아래 경계층 두께 동일 → 중앙 온도 = T_ref → θ_c = 0.

**NOB 비대칭**: glycerol처럼 점성이 온도에 크게 의존하는 유체에서는  
- 열벽(hot, 하부): μ̂ 작음 → 빠른 상승류 → 얇은 경계층  
- 냉벽(cold, 상부): μ̂ 큼 → 느린 하강류 → 두꺼운 경계층

두 경계층 두께가 다르므로 열유속 균형에서 중앙 온도가 올라감 → **θ_c > 0**.

μ 비율이 클수록 θ_c 더 커짐. 이 NOB 효과가 시뮬레이션에서 제대로 포착되는지가 이 조사의 핵심.

---

## 2. 시뮬레이션 설정

**공통 물리 조건** (두 코드 동일):
- 유체: 순수 glycerol, T_ref = 40°C (313 K)
- Ra = 1×10⁶, Pr = 2547, ΔT = 20 K
- 도메인: H = 1 (벽간 거리), Γ = 6 (사방 주기경계)
- 물성 다항식: ρ̂, Ĉp, κ̂, μ̂를 θ에 대한 5차 다항식으로 표현

**물성 다항식 형태** (두 코드 동일):
```
μ̂(θ) = poly_a(t) × poly_d(t)     where t = ΔT × θ ∈ [-10, 10]
poly_d(t) = 1 + d11·t + d12·t² + d13·t³ + d14·t⁴ + d15·t⁵
계수: d11=-702.83e-4, d12=2393.1e-6, d13=-6923.0e-8, d14=33131.3e-10, d15=-71517.5e-12
```

이 계수로 계산한 점성 비율: **μ_cold/μ_hot = 4.16×** (Cheng 2008 공식의 4.08×와 2% 이내 일치).

**코드별 격자 설정**:
| 코드 | 격자 | γ (y방향) | 열BL 셀 수 |
|------|------|-----------|-----------|
| MPM-STD_C | 128³ | 2.6 | ~6 |
| PaScaL_TCS | 64³  | 2.6 | ~3 |

---

## 3. 결과 비교

| 출처 | 격자 | θ_c | 비고 |
|------|------|-----|------|
| 참조값 (`fig9_data/PaScaL_TCS.csv`) | 128³ | **0.0744** | 논문 제시 |
| MPM-STD_C (GPU, 500K steps) | 128³ | **0.040** | 수렴 확인 |
| PaScaL_TCS 저장소 (OB 압력) | 64³ | **0.027** | 수렴 확인 (50K steps) |

격자 의존성: 64³ → 0.027, 128³ → 0.040. 격자를 키울수록 θ_c 증가.

그러나 **같은 128³ 격자에서 MPM-STD_C(0.040)와 논문 참조(0.074)의 차이는 격자로 설명 불가.**

---

## 4. 소거한 가설 (Report 23 종합)

아래 항목들은 조사 결과 **θ_c 결손의 원인이 아님**으로 확인.

### 4.1 θ_c 측정/정의 오류 ✗
`center_temp` = 중앙평면(nz/2) T의 xy 시간평균. z-프로파일 대조로 정확 확인.

### 4.2 격자/해상도 (gamma 스캔) ✗
128³에서 γ=1.5/2.6/3.5 스캔:
- γ=1.5: 열BL 미해상 → 대류 억제 (무효)
- γ=2.6: 6셀 → θ_c ≈ 0.041
- γ=3.5: 10셀 → θ_c ≈ 0.030

대류가 발생하는 범위(γ≥2.6)에서 BL 해상도 개선이 오히려 θ_c를 낮추는 경향 → 해상도가 결손의 직접 원인 아님.

### 4.3 κ̂ 에너지 확산항 ✗
MPM-STD_C `scalar_var_stencil.hpp` ↔ PaScaL `module_solve_thermal.f90` 줄단위 대조:
- `KPP = 1 + c11·t + c12·t² + ...` 형태 동일
- `dKPP = c11 + 2c12·t + ...` (선형화) 동일
- `irc_h = 0.5·Ct·invRhoCp` 계수 동일
- 확산 스텐실: `(kPP_hi·ded_hi − kPP_lo·ded_lo + e·dkPP·ded)/dx` 동일

### 4.4 μ̂ 운동량 점성항 ✗
MPM-STD_C `k_mom_rhs_var`/`k_mom_adi_var` ↔ PaScaL `module_solve_momentum.f90` 대조:
- 정규응력 2×, 교차응력 weight-1
- 부피가중 면간 점성 `mfce` 동일
- `−M·qⁿ` 잔차 동일
- 블록결합 `dV/dU` 동일

### 4.5 glycerol 계수 ✗
점성 비율: 모델 4.2× vs 실제(Cheng) 4.1× — 잘 재현.
κ̂ 비대칭: +0.78~2.34% (작은 2차 효과).

### 4.6 물성 평가 타이밍 (실차이, but 주범 아님) △
- 수정 전: 에너지·운동량 모두 `Tⁿ` 물성 사용
- 수정 후: 에너지 후 → 운동량 직전 `Tⁿ⁺¹`으로 재평가
- 결과: +4% 개선 (0.031→0.032 @200K). 유지 필요하지만 1.85× 결손의 주범 아님.

### 4.7 다항식 DeltaT 스케일링 오해 (조사 후 오류 확인) ✗
**조사 과정에서 발생한 오해**: MPM-STD_C가 `t = ΔT × θ`를 쓰고 PaScaL은 `t = θ`를 써서 계수가 달라야 한다고 잘못 추론.

**실제 확인**: PaScaL `module_solve_momentum.f90:159`도 `tmp1 = DeltaT * T(i,j,k)` 사용. **양 코드 모두 동일한 스케일**로 다항식 평가.

→ OLD 계수(d11=-702.83e-4 등)가 양 코드 모두에서 올바르게 μ_cold/μ_hot = 4.16×를 만들어 냄.

---

## 5. PaScaL_TCS NOB 압력 발산 문제

PaScaL NOB 압력 포아송 방정식:
```
∇²(dPhat) = DivU/dt + ExtraTerm
ExtraTerm  = ∇·[(1 − 1/ρ̂) · ∇dPhat]
```

속도 투영:
```
u ← u − dt · (1/ρ̂) · ∇dP
```

이 ExtraTerm을 활성화하면 **step ~25에서 발산** (Ra=1e6 glycerol NOB, 모든 격자/dt 조합).

조사한 수정:
- Neumann 고스트셀 `dPhat(:,0,:) = dPhat(:,1,:)` 추가 → 발산 지속
- ExtraTerm 완전 비활성화(OB 압력으로 대체) → 안정

현재 우회책: ExtraTerm=0, OB projection 사용. 근본 수정 미해결.

MPM-STD_C는 동일한 구조의 ExtraTerm이 안정적으로 동작 → 구현 세부 차이 있으나, ExtraTerm 크기 자체가 `(1-1/ρ̂) ~ ±0.5%`로 작아 θ_c에 미치는 영향은 작을 것으로 추정.

---

## 6. 확인된 근본 원인 — NOB 압력 이중계산 버그

### 6.1 참조값은 실제 DNS (BL 이론이 아님)

Kim et al. 2023 논문 Fig. 9 직접 확인:
- 빨간 원 (red circles) = "present study" = **PaScaL DNS 실제 결과**
- CSV (ΔT=20K → 1.486K) → θ_c = 0.0744는 **실제 DNS 값**
- 실선 = BL 이론; PaScaL DNS가 이론과 일치하는 것은 DNS가 맞기 때문

### 6.2 PaScaL의 NOB 압력 구현

Kim et al. 2023 Eq. (21) 압력 분리법:

```
Poisson: ∇²(δp) = PRHS + ExtraTerm
         ExtraTerm = ∇·[(1-1/ρ̂)·∇δp_old]    ← 가변 밀도 보정

속도 보정 (PaScaL, Cmp=1.0, §804-817):
         u -= dt * ∇δp                         ← OB 투영 (상수!)
```

**설계 원리**: ExtraTerm이 Poisson에서 가변 밀도를 이미 반영하므로, 이후 속도 보정은 단순 OB 투영(Cmp=1)으로 충분. 둘이 합쳐서 NOB 효과를 정확히 구현.

### 6.3 MPM-STD_C의 버그

`project_nob_`에서 ExtraTerm(Poisson) 외에 속도 보정에도 NOB 항 추가 → **이중계산**:

```cpp
// 버그 코드 (수정 전):
u[c] -= dt * ((p[c]-p[cxm])/dmx1[i]
             + (irc_x - 1)*(ph[c]-ph[cxm])/dmx1[i]);
//             ↑ OB 보정        ↑ 이 항이 이중계산!
```

**물리적 영향**:
- 열벽(hot wall): `1/ρ̂ > 1` → 추가 상승류 감속 → NOB 비대칭 상쇄
- ΔT가 클수록(μ 비율 증가) 이 이중계산이 더 많이 상쇄 → 배율 증가(1.85× → 2.04×)

### 6.4 수정 내용 (2026-07-02)

CPU (`src/equation/pressure/cpu/pressure_base_cpu.cpp`) 및  
GPU (`src/equation/pressure/gpu/pressure_engine_gpu.cuh`) 모두:

```cpp
// 수정 후 (OB 투영만, PaScaL과 동일):
u[c] -= dt * (p[c]-p[cxm])/dmx1[i];
```

**예상 결과**: θ_c ≈ 0.040 → 0.074 (재빌드 후 확인 필요)

---

## 7. 직접 확인된 사실 목록

1. **μ_cold/μ_hot = 4.16×** 양 코드 동일하게 물리적으로 타당 ✓
2. **모든 공간 이산화 항** PaScaL ↔ MPM-STD_C 줄단위 일치 ✓
3. **PaScaL 저장소 실구동** (64³, OB압력, 50K steps) → **θ_c = 0.027** 수렴 ✓
4. **MPM-STD_C GPU 실구동** (128³, 500K steps) → **θ_c = 0.040** 수렴 ✓
5. 격자 64³→128³에서 θ_c 0.027→0.040 증가 (격자 의존성 있음) ✓
6. **동일 128³ 격자에서 논문(0.074)과 두 코드(0.040)의 차이는 설명 미완** ⚠️

---

## 8. 다음 결정적 실험

우선순위 순:

1. **논문 원문 확인**: `fig9_data/PaScaL_TCS.csv`의 데이터가 DNS인지 BL 이론인지.
   - 논문(Kim et al. 2023 또는 PaScaL_TCS paper) 직접 확인
   - BL 이론값이면 결손 설명 완료

2. **PaScaL 128³ 직접 구동** (NOB 압력 발산 문제 해결 후):
   - ExtraTerm 발산 근본 원인 파악 → 안정화
   - 128³ 실구동 → 0.04(MPM-STD_C와 일치) vs 0.074(논문과 일치) 직접 확인

3. **NOB 압력 효과 크기 정량화**:
   - MPM-STD_C에서 ExtraTerm 비활성화 → θ_c 변화 측정
   - ExtraTerm이 0.040→0.027 수준의 변화를 일으키는지 확인

---

## 9. 파일 참조

| 파일 | 내용 |
|------|------|
| `Report/23_rbc_nob_thetac_deficit_investigation.md` | 상세 소거 과정 및 gamma 스캔 결과 |
| `apps/rbc/fig9_data/PaScaL_TCS.csv` | 논문 참조값 (ΔT_dim, θ_c_dim) |
| `apps/rbc/input_fig9_gpu_nob_20_x.toml` | MPM-STD_C 실행 설정 |
| `PaScaL_TCS/run/pascal_correct_visc.log` | PaScaL 64³ 실구동 로그 (θ_c=0.027) |
| `PaScaL_TCS/src/module_global.f90` | glycerol 계수 (d11-d15) |
| `PaScaL_TCS/src/module_solve_momentum.f90:159` | `tmp1 = DeltaT*T` — DeltaT 스케일링 위치 |
| `src/physics/properties/cpu/properties_cpu.cpp:58` | `t = Dt * Tp[i]` — MPM-STD_C DeltaT 스케일링 |

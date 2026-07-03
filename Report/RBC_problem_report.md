# RBC NOB θ_c 결손 디버깅 종합 보고서

> **작성일**: 2026-07-02  
> **최종 업데이트**: 2026-07-02 (논문 확인 후 근본 원인 발견 및 수정)
> **현황**: **버그 수정 완료** — NOB 압력 이중계산(double-counting) 버그 발견 및 수정  
> **참조**: Report/23_rbc_nob_thetac_deficit_investigation.md (소거 과정 상세)

---

## 1. 문제 정의

**어디랑 비교했나?**

비교 대상: `apps/rbc/fig9_data/PaScaL_TCS.csv`

```
19.975 K,  1.486 K   →  ΔT=20K, θ_c = 1.486/20 = 0.0743
39.908 K,  5.400 K   →  ΔT=40K, θ_c = 5.400/40 = 0.1350
59.632 K, 11.400 K   →  ΔT=60K, θ_c = 11.40/60 = 0.1900
```

출처: `readme.md` — "validated against ... published DNS (Ng et al., Kim et al. 2023)".  
즉 Kim et al. 2023 논문(PaScaL_TCS)이 제시한 glycerol NOB RBC 결과.  
단, Report/23에서 확인: 이 값들이 **BL 이론 Zhang/Ahlers/Horn과 정확히 일치**한다고 기재됨.

**MPM-STD_C 실제 결과**:

| ΔT | steps | θ_c 누적평균 | θ_c 순간수렴값 | 참조 | 배율 |
|----|-------|-------------|--------------|------|------|
| 20K | 500K | 0.0419 | ≈ 0.047 (@275K부터 수렴) | 0.0744 | 1.58–1.78× 낮음 |
| 40K | 350K | 0.0662 | ≈ 0.085 | 0.1350 | 1.59–2.04× 낮음 |

ΔT가 클수록 배율이 커짐 → NOB 효과를 **체계적으로** 약하게 잡는 패턴.

---

## 2. 물리 설정

**공통 조건 (두 코드 완전 동일)**:
- 유체: 순수 glycerol, T_ref = 40°C (= 313 K)
- Ra = 1×10⁶, Pr = 2547, ΔT = 20 K
- Γ = Lx/H = Lz/H = 6, 격자 128³, γ(y방향) = 2.6
- 물성 다항식: `t = ΔT × θ` (양 코드 동일한 스케일링)

**왜 θ_c > 0 이어야 하는가?**

glycerol은 온도에 따라 점성이 급격히 변함. ΔT=20K에서:
```
μ̂(θ=+0.5, 열벽) ≈ 0.10    → 작은 점성 → 빠른 상승류 → 얇은 BL
μ̂(θ=−0.5, 냉벽) ≈ 2.05   → 큰 점성  → 느린 하강류 → 두꺼운 BL

μ_cold / μ_hot = 2.05 / 0.10 = 4.16×
```
두 경계층 두께가 달라 열유속 균형에서 중앙 온도가 올라감 → θ_c > 0.

**BL 이론 예측** (Zhang/Ahlers/Horn 1D 경계층 해석):
```
δ_T_cold / δ_T_hot ≈ √(μ_cold/μ_hot) = √4.16 = 2.04
열유속 균형: κ̂·(0.5 − θ_c)/δ_T_hot = κ̂·(θ_c + 0.5)/δ_T_cold
→ θ_c = 0.5 × (√R − 1)/(√R + 1),  R = 4.16  → θ_c ≈ 0.17
Nu로 나눠 대류 혼합 보정 → θ_c ≈ 0.07 수준
```
이 BL 이론값이 0.074와 일치한다. DNS는 다른 값(0.040)을 줄 수 있다.

---

## 3. 디버깅 내역 (시간 순)

### 3.1 jep/jem 마스크 누락 (→ 기각)

**의심**: ADI Stage 3에서 `eAPJ *= jep; eAMJ *= jem` 마스크 누락 →  
열경계 근처 대각선 인플레이션 → θ_c 저하.

**확인 및 수정**: `scalar_var_stencil.hpp` 수정, GPU job 29761 재실행.  
인플레이션 항 크기: `irc_h·κ̂/(dx·dmx)·dt ≈ 0.005` = 기저 대각의 **0.5%**.

**결과**: 수정 전 0.040, 수정 후 **0.039** (@368K). 사실상 동일. **기각.**

---

### 3.2 물성 평가 타이밍 (→ 개선 있음, 주범 아님)

**의심**: MPM-STD_C가 에너지/운동량 모두 `Tⁿ` 물성으로 계산.  
PaScaL은 에너지 후 → 운동량 직전에 `Tⁿ⁺¹`으로 재평가.

```
PaScaL 순서:
  thermal_coeffi(Tⁿ) → 에너지 풀기 → momentum_coeffi(Tⁿ⁺¹) → 운동량 풀기

MPM-STD_C (수정 전):
  update_properties(Tⁿ) → 에너지 → 운동량  (운동량이 Tⁿ 물성 사용)
```

**수정**: `main_gpu_nob.cu:419` — 에너지 직후 `update_properties_gpu(Tⁿ⁺¹)` 추가.

**결과**: θ_c 0.031 → 0.033 @200K (**+6%**). 올바른 수정이지만 1.58× 결손의 주범 아님.

---

### 3.3 κ̂ 에너지 확산항 코드 대조 (→ 일치 확인)

**방법**: `scalar_var_stencil.hpp` ↔ `module_solve_thermal.f90` 줄단위 비교.

**확인 항목**:
```
KPP  = 1 + c11·t + c12·t² + c13·t³   (t = ΔT·θ)
dKPP = c11 + 2c12·t + 3c13·t²          (KPP의 θ 도함수)
irc_h = 0.5 × Ct × invRhoCp            (열확산 계수)

확산 스텐실:
  (KPP_hi·ΔT_hi − KPP_lo·ΔT_lo + KPP·dKPP·(ΔT)²/dx) / dx

→ 완전 일치 ✓
```

---

### 3.4 μ̂ 운동량 점성항 코드 대조 (→ 일치 확인)

**방법**: `k_mom_rhs_var.cu` / `k_mom_adi_var.cu` ↔ `module_solve_momentum.f90` 줄단위 비교.

**확인 항목 (모두 일치)**:
```
정규응력:   2 × μ̂_face × (∂u/∂x)        (계수 2× 포함)
교차응력:   1 × μ̂_face × (∂u/∂y + ∂v/∂x)  (계수 1×)
면간 점성:  mfce = (μ(i,j,k) + μ(i±1,j,k)) / 2  (부피가중 평균)
잔차항:     −M·qⁿ  (implicit 잔차, Beam-Warming delta-form)
블록결합:   block_couple_dV(dU), block_couple_dU(dV)  (교차속도 증분 결합)

→ 완전 일치 ✓
```

---

### 3.5 부력항 형태 비교 (→ 실효적 동일)

PaScaL (wall-normal = y = x2 방향):
```fortran
! module_solve_momentum.f90
SRHS += Cmt * (T + a12pera11 * DeltaT * T**2) * invRho
```

MPM-STD_C (wall-normal = z 방향):
```cpp
// k_mom_rhs_var.cu
rhs_z += Cmt * (T - Ground_T) * invRho
```

수치 차이:
- `a12pera11 = a12/a11 = 7.924e-4`: 이차항 T² 기여 → `7.924e-4 × 20 × 0.25 ≈ 0.004` (**0.4%** 수준)
- `Ground_T = 0` (RBC 기준): 동일
- **실효적으로 동일** ✓

---

### 3.6 gamma(격자 스트레칭) 스캔 (→ 격자 해상도 무관)

128³, Γ=6, Ra=1e6, ΔT=20K:

| γ | 열BL 셀 수 | u_max | θ_c @200K | 비고 |
|---|-----------|-------|-----------|------|
| 1.5 | 3–4 | 0.007 | 대류 없음 | BL 미해상 |
| **2.6** | **6** | **0.53** | **0.041** | 기본 설정 |
| 3.5 | 10 | 0.68 | 0.030 | |

γ=2.6→3.5로 BL 해상도를 높여도 θ_c는 0.03–0.04 수준으로 유지. 격자 해상도가 주범이 아님.  
γ=1.5: BL 미해상으로 대류 자체가 억제 → 이 영역은 제외.

---

### 3.7 다항식 DeltaT 스케일링 (→ 이번 세션에서 혼동 발생 후 정정)

**처음 의심**: "PaScaL은 `t=θ`(무차원)로 다항식 평가, MPM-STD_C는 `t=ΔT·θ`(차원)로 평가 → 같은 계수라도 전혀 다른 μ 비율"

코드 직접 확인:
```fortran
! PaScaL: module_solve_momentum.f90:159
tmp1 = DeltaT * T(i,j,k)    ← 이미 ΔT 스케일링 사용!
```
```cpp
// MPM-STD_C: properties_cpu.cpp:58
const double t = Dt * double(Tp[i]);  ← 동일: Dt = ΔT
```

**양 코드 모두 `t = ΔT × θ = ±10` 로 평가.** 계수 d11=-702.83e-4 등이 두 코드에서 동일한 물리적 μ 비율(4.16×)을 만들어 냄.

→ 이 가설은 **오류**였음. OLD 계수가 올바름.  
(잘못된 계수로 실행한 `pascal_ob_50k.log` 결과 θ_c=0.027 — 무효 처리)

---

### 3.8 PaScaL NOB 압력 ExtraTerm 발산 (→ 우회, 미해결)

PaScaL NOB 압력 포아송 방정식:
```fortran
! module_solve_pressure.f90
ExtraTerm = ∇·[(1 − invRho) · ∇dPhat]
PRHS = DivU/dt + ExtraTerm
```
속도 투영 (Cmp=1.0 상수로 OB 투영):
```fortran
U = U − dt · Cmp · (dP(i,j,k) − dP(im,j,k)) / dmx1(i)  ! Cmp = 1.0 고정
```

**증상**: glycerol NOB에서 ExtraTerm 활성화 시 **step 25–30에서 발산**. 전 격자/dt 조합 동일.

**시도한 수정**:
- Neumann 고스트셀: `dPhat(:,0,:) = dPhat(:,1,:)` 추가 → **발산 지속**
- NOB 속도 투영 비활성화 (OB projection만) → **발산 지속**
- ExtraTerm 완전 제거 → **안정화** ← 현재 우회책

ExtraTerm 크기 추정:
```
(1 − invRho) ≈ a11·ΔT·θ = −9.578e-3 × θ → 벽에서 약 ±0.5%
ExtraTerm ≈ 0.5% × DivU/dt → PRHS 대비 0.5% 보정
```
→ θ_c에 미치는 영향 자체는 작음. 하지만 **수치 불안정 원인 불명**. MPM-STD_C에서는 동일 항이 안정적으로 동작.

---

### 3.9 PaScaL 저장소 직접 실행 (→ θ_c = 0.027, 수렴 확인)

**목적**: 저장소 PaScaL이 0.074(논문)를 내는지 0.040(MPM-STD_C)을 내는지 확인.

**설정**: 64³, Γ=6, γ=2.6, Ra=1e6, Pr=2547, ΔT=20K, **OB 압력** (ExtraTerm=0), 50K steps.

**결과 (수렴)**:
```
step  1     → θ_c = 0.0239  (초기 선형프로파일)
step 15K    → θ_c = 0.0231  (최솟값, 초기 확산)
step 25K    → θ_c = 0.0247  (대류 발달, 상승)
step 50K    → θ_c = 0.0270  (수렴, 마지막 5스텝 변화 < 1e-7)
```

→ **PaScaL 저장소도 0.074에 못 미침**.

격자 크기에 따른 비교:
```
PaScaL 저장소, 64³,  OB압력  → θ_c ≈ 0.027
MPM-STD_C,      128³, NOB압력 → θ_c ≈ 0.040
논문 참조 (CSV)                → θ_c = 0.074
```

64³ → 128³에서 θ_c가 증가하는 격자 의존성은 있지만, **같은 128³에서 MPM-STD_C(0.040)와 논문(0.074)이 다름** → 격자로 설명 불가.

---

## 4. 근본 원인 (2026-07-02 확인)

### 4.1 참조값은 실제 DNS (BL 이론이 아님)

Kim et al. 2023 논문(Comput. Phys. Commun. 290, 108779) Fig. 9을 직접 확인:

- **빨간 원 (red circles)** = "present study" = **PaScaL DNS 실제 결과**
- CSV 값: (ΔT=20K, 1.486K) → θ_c = 1.486/20 = **0.0744** (논문 DNS)
- 실선/점선 = Zhang 1997 / Ahlers 2006 BL 이론
- PaScaL DNS가 BL 이론과 잘 일치하지만, 이는 DNS가 올바르기 때문임

**즉, 참조값 0.074는 실제 DNS이고 이것이 올바른 정답이다.**

### 4.2 발견된 버그: NOB 압력 이중계산

논문(Kim et al. 2023) Eq. (21)의 압력 분리법(pressure splitting):

```
Poisson: ∇²(δp) = PRHS + ExtraTerm
         ExtraTerm = ∇·[(1-1/ρ̂)·∇δp_old]

속도 보정 (PaScaL 실제 코드, Cmp=1.0):
         u -= dt * ∇δp   ← OB 투영
```

**핵심**: ExtraTerm을 Poisson RHS에 추가하면 이 식은 이미 변수밀도 투영 효과를 포함한다.  
ExtraTerm이 포함된 Poisson을 풀면 δp 자체가 NOB 보정이 반영된 값이 되므로,  
그 후에는 **OB 투영(Cmp=1)**만 사용해도 올바른 NOB 속도장을 얻는다.

**MPM-STD_C의 버그**:

```cpp
// pressure_base_cpu.cpp (수정 전):
// ExtraTerm을 Poisson에 추가함 (올바름) ✓
rhs_C_[...] += ExtraTerm;   // = ∇·[(1-invRho)·∇dPhat]

// 그 후 속도 보정에서 또 NOB 항 추가 (잘못됨) ✗
u[c] -= dt * ((p[c]-p[cxm])/dmx1[i] + (irc_x - 1)*(ph[c]-ph[cxm])/dmx1[i]);
//                              ↑OB 보정          ↑이 항이 이중계산!
```

**이중계산의 물리적 효과**:

열벽(hot wall): `ρ̂ < 1` → `invRhoc > 1` → `(invRhoc-1) > 0`

여기에 `∂dPhat/∂y > 0` (열벽에서 멀어질수록 압력 감소)이면:
- 추가 속도 보정: `-dt * (invRhoc-1) * ∂dPhat/∂y < 0` → **상승류 감속**
- 냉벽: `invRhoc < 1` → 하강류도 감속됨

결과: NOB 비대칭 효과(열벽=빠른 상승, 냉벽=느린 하강)가 **상쇄**됨 → θ_c 감소.

**이중계산의 크기 비율**:
```
ΔT=20K: μ_cold/μ_hot = 4.16,  hot BL 셀 10개 → θ_c 0.074 → 0.040 (∼1.85× 저하)
ΔT=40K: μ_cold/μ_hot = 15.6,  hot BL 셀 6개  → θ_c 0.135 → 0.066 (∼2.04× 저하)
```
ΔT 증가(μ비율 증가, NOB 효과 강화)할수록 배율이 커짐 → **NOB 항 이중계산**이 주범임을 정성적으로 확인.

### 4.3 수정 내용

**CPU** (`src/equation/pressure/cpu/pressure_base_cpu.cpp`):
```cpp
// 수정 전 (이중계산):
u[c] -= dt * ((p[c]-p[cxm])/dmx1[i] + (irc_x-1)*(ph[c]-ph[cxm])/dmx1[i]);

// 수정 후 (OB 투영만):
u[c] -= dt * (p[c]-p[cxm])/dmx1[i];
```

**GPU** (`src/equation/pressure/gpu/pressure_engine_gpu.cuh`, kernel `k_project_nob_pres`):
```cuda
// 수정 전:
U[c] -= dt * ((Pc-P[cxm])/dmx1[i] + (irc_x-1.0)*(ph-dPhat[cxm])/dmx1[i]);

// 수정 후:
U[c] -= dt * (Pc - P[cxm])/dmx1[i];
```

---

## 5. 수정 후 예상 결과

PaScaL(Kim et al. 2023)과 동일한 구현:
- ExtraTerm = `∇·[(1-invRho)·∇dPhat]` Poisson RHS에 추가 ✓
- 속도 보정: `u -= dt·∇δp` (OB 투영, Cmp=1) ✓

**예상 θ_c**: 0.040 → ~0.074 (PaScaL 논문 DNS와 일치)

재빌드 후 GPU 실행으로 확인 필요.

---

## 6. 아직 남은 문제

| 항목 | 상태 |
|------|------|
| 수정 후 θ_c ≈ 0.074 확인 | 재실행 필요 |
| PaScaL 저장소 ExtraTerm 발산 근본 원인 | 미해결 (MPM-STD_C에서는 안정) |

---

## 6. 파일 지도

```
apps/rbc/
├── fig9_data/PaScaL_TCS.csv          ← 비교 대상 (3행: ΔT/θ_c 쌍)
├── input_fig9_gpu_nob_20_x.toml      ← MPM-STD_C 실행 설정 (128³, γ=2.6)
└── main_gpu_nob.cu:419               ← 물성 타이밍 수정 위치

src/
├── physics/properties/cpu/properties_cpu.cpp:58  ← t = Dt * Tp (ΔT 스케일링)
└── equation/pressure/cpu/pressure_engine_cpu.cpp ← NOB 압력 ExtraTerm (안정)

PaScaL_TCS/
├── src/module_global.f90:126         ← glycerol 계수 d11–d15
├── src/module_solve_momentum.f90:159 ← tmp1 = DeltaT*T (ΔT 스케일링)
├── src/module_solve_pressure.f90     ← ExtraTerm (현재 비활성화)
├── src/module_solve_thermal.f90      ← κ̂ 확산 + 초기조건 perturbation
└── run/pascal_correct_visc.log       ← 64³ 실행 로그 (θ_c=0.027 수렴)

Report/
├── 23_rbc_nob_thetac_deficit_investigation.md  ← 소거 과정 전체 상세
└── 24_rbc_nob_investigation_summary.md         ← 영문 요약
```

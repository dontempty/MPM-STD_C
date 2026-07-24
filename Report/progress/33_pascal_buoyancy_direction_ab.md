# 33. PaScaL_TCS 부력 방향 A/B — 공개판은 RBC 구성이 아니다

> **작성**: 2026-07-10
> **동기**: 사용자가 공개판 PaScaL_TCS를 직접 돌려도 검증 데이터(Kim 2023 Fig9 /
> Pan 2022 Table 5)와 맞지 않음을 확인. 원인 후보 = 공개판의 부력 방향.

## 1. 발견 — 공개판의 부력은 x1(주기방향)에 걸려 있음

`module_solve_momentum.f90` (공개판, 커밋 기준):

- **solvedU (x1-모멘텀, ~347-352행): 부력 활성**
  `RHS += Cmt·(Tc + a12pera11·Tc²·ΔT)·invRhoc`
- **solvedV (x2-모멘텀 = 벽수직, ~703-709행): 부력 전체 주석**
- solvedW (x3): 부력 없음

즉 공개판은 중력이 **주기방향 x1**을 향하는 **수직채널(DHVC)형 구성**이다. RBC는
중력이 벽수직(x2)이어야 하므로, 공개판을 그대로 RBC 파라미터로 돌리면 물리가
완전히 다른 문제를 푼다 — **공개판 PaScaL이 Fig9 값을 재현하지 못하는 직접 원인
후보**. (Kim 2023 "128³" 격자 기재 불일치와 함께, 논문 프로덕션 코드 ≠ 공개판이라는
증거가 하나 더 쌓임.)

## 2. A/B 구성 (2026-07-10)

| 변형 | solvedU 부력 | solvedV 부력 | 물리 |
|---|---|---|---|
| **A = buoyU** (공개판 그대로) | ON | OFF | 중력 ∥ x1 (주기방향) — 채널형 |
| **B = buoyV** (RBC 교정) | **OFF** | **ON** | 중력 ∥ x2 (벽수직) — RBC |

- B의 solvedV 부력은 주석 해제 그대로 (Tc = x2-면 보간, invRhoc 곱 유지 —
  형태는 우리 포트의 add_buoyancy_force_nob와 동일 계열).
- 소스는 B 상태로 남김(마커 주석 `!! BUOY A/B (Report/33)`); A 바이너리는
  `run/PaScaL_TCS_buoyU.ex`로 보존, B는 `run/PaScaL_TCS_buoyV.ex`.
- 기존 로컬 수정(글리세롤 물성 블록, Mu/invRho i-인덱스 수정, Theta_c 모니터)은
  양쪽에 공통.

## 3. 런 설정

- **256³, 64랭크(4×4×4)**, 각 1 CPU 노드 (jobs 30103=A/cpu01, 30104=B).
- Ra=1e6, Pr=2547, Δ=20K, Γ=6(Aspect1=Aspect3=6, H=1), γ₂=2.6, MaxCFL=0.3,
  dtStart=0.02, Timestepmax=50000, ContinueFileout=1(연장 가능).
- 격자·물리 모두 우리 포트의 h-256 프로덕션(E10~E16)과 직접 비교 가능한 구성.
- 디렉토리: `PaScaL_TCS/run_ab_buoy{U,V}_256/` (run.log에 Theta_c 모니터 출력).

## 5. 실행 과정에서 발견한 공개판 투영 버그 2건 (2026-07-10)

A/B가 시작 1–2스텝에서 div 게이트(1e-3)로 연쇄 블로업 → divPRE/divPOST 계측 패치로
원인 비섹션. **변밀도 정확상쇄쌍(ExtraTerm ↔ 투영 (1/ρ̂−1)∇dPhat 항)이 공개판에서
깨져 있었다**:

1. **투영 루프 내 dPhat/dP0 제자리 갱신** (module_solve_pressure.f90
   mpi_pressure_Projection): 오름차순 루프에서 이웃(im/jm/km)의 dPhat이 이미 새
   값으로 덮인 채 속도 갱신에 읽힘 → RHS가 본 dPhat 장과 투영이 쓴 dPhat 장이 달라
   상쇄 붕괴. **수정 = 외삽을 루프 종료 후 별도 패스로 분리** (우리 포트의
   k_dphat_extrapolate 분리 커널과 동일 구조).
2. **ExtraTerm의 벽 레그 마스크 누락** (mpi_pressure_RHS): DivUm·투영은 벽 V-면
   레그를 jvm/jvp로 드롭하는데 ExtraTerm만 미충전 dPhat 벽 고스트(j=0/n2sub)를
   읽음 → 벽 행 RHS 오염. **수정 = ExtraTerm y-레그에 동일 마스크.**

검증: 수정 전 divPOST = 0.086·divPRE(step1) / 39× 증폭(step2) → 수정 후 **매 스텝
1e-15 (기계 정밀도)**.

- **잠복 조건**: 두 버그 모두 (1/ρ̂−1)과 dPhat의 스텝 간 변화량에 비례 — **OB에서는
  항등적으로 0**(1/ρ̂≡1), NOB 준정상 유동에서도 미미. 글리세롤 NOB + 벽수직 부력 +
  콜드스타트(δp가 0→정수압으로 점프)에서만 발화 → 원저자 사용례에서 보이지 않음.
  "공개판 ≠ 논문 프로덕션"의 **세 번째 독립 증거**.
- 부수 수정(물리 불변): dtStart 2e-5 + Post_CFL dt 성장 리미터(≤5%/스텝; 원본은
  속도≈0 초기장에서 dt를 dtMax로 즉시 점프), IC를 셀 노이즈(구세션 로컬 수정) →
  매끈한 저파수 섭동으로 교체, 블로업 프린트에 maxDivU/dt 추가, 첫 10스텝
  divPRE/divPOST 계측.
- **divPRE 크기 해석**: step1 divPRE ≈ 2.6e-2 = dt·부력이 아니라 **dt·(부력 × 벽
  근처 암시 점성 감쇠 프로파일의 기울기)** — 벽 점성수 dt·ν/Δy_w²≈17의 감쇠가 몇
  셀 만에 풀리며 ∂y(dV) ~ dt·240 (실측 dt·130, factor 2 일치). 투영이 지울 대상으로
  정상이며 감쇠 궤적(스텝마다 하강)도 정합.

## 6. 수렴 런 (2026-07-10, jobs 30124/30125)

vOFF/vON × 256³/64랭크, Timestepmax=150000 (dt 캡 0.0078 → ≈1170 t.u., ~33h),
ContinueFileout=1(연장 가능). 판정 = θ_c(center) 수렴값: vOFF(중력 x1만) 대조군 vs
vON(중력 x1+x2, V항만 추가) — vON이 RBC NOB θ_c>0으로 발달하는지, 참조 0.071 및
우리 포트 평형 0.052와 3자 비교.

## 4. 판정 기준

1. **A vs B**: θ_c 궤적이 갈라지는가 — A(중력 x1)는 RBC 대류 부재/이상 거동,
   B는 정상 RBC NOB (θ_c > 0 발달) 기대.
2. **B vs 참조**: B의 θ_c/Nu가 Pan 0.071/9.61로 가는가 — 가면 공개판의 부력
   방향이 "PaScaL 불일치"의 전부; 우리 포트의 잔여 갭(θ_c 평형 0.052 vs 0.071,
   Report/31)과의 대조가 다음 단서.
3. **B vs 우리 포트**: 동일 격자·물리에서 B와 MPM-STD_C h-256의 θ_c 평형값
   직접 비교 — 남은 −25% 갭이 참조값 문제인지 포트 문제인지 분리.

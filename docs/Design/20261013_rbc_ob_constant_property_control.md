# RBC OB 대조 — NOB 윗벽 속도 rms 스파이크는 가변 물성 경로에서만 생긴다 (2026-10-13 ~ 14)

## 1. 질문과 이전 확인
질문(사용자): rbc_nob(글리세롤 ΔT 40 K, Ra 1e6, Pr 2547, 128×128×256, γ_z 2.6)의 **윗벽(차가운 벽) 근처 속도 rms 스파이크**가
NOB 경로의 문제인지 확정하기 위해, NOB 를 빼고 OB 로만 같은 조건에서 통계를 내어 윗벽 속도·온도 rms 가 튀는지 본다.

이미 확인돼 있던 것:
- 옛 코드(MPM-STD_C, `Report/progress/27·28·30`): 아랫벽 U 스파이크(dW 벽면 증분 누수, **OB 에서도 발생**)는 코드 결함으로 수정(R27).
  h-256 **윗벽** U 체커보드(R28 §6.3, R30)는 NOB 글리세롤에서만 발화했고 원인은 명시 블록 커플링 보정의 한계안정성(언더릴랙스 0.5 로 억제).
  OB 에서 윗벽을 직접 대조한 기록은 없다.
- 새 코드 `docs/Design/20261009_rbc_nob_dt_criterion.md` §6: OB 앱(apps/rbc, Pr 0.7, 192×192×96)의 기존 통계가 상·하 대칭 — 그러나 다른
  앱·다른 Pr·다른 z 격자라 **통제된 비교가 아니었다**.

## 2. 1차 시도 — legacy `rbc` 앱은 이 조건에서 쓸 수 없다 (벽 미끄럼)
`apps/rbc`(OB 앱)를 NOB 본런 덱과 같은 조건(물성만 상수, 냉시작 → t 27000, 통계 16900~)으로 돌렸다(잡 36040, gpu01; 통계 모듈에 T·u·v rms
와 벽 인접 평면 max|U| 모니터를 추가). 윗벽 스파이크는 없었지만(윗/아랫 rms 비 1.00) **물리가 틀렸다**:

| 창 16900~27000 | legacy rbc (Pr 2547, dt 0.05) | NOB visc12 아랫벽 | 가변경로 ΔT=0 |
|---|---|---|---|
| 벽 인접 u_h,rms (k=1) | 2.11e-3 (평탄, 멱지수 0.13) | 3.6e-4 (∝ z^0.76) | 1.7e-4 (∝ z^0.80) |
| w_rms 첫 내부 면 / 멱지수 | 4.2e-5 / **1.03 (∝ z)** | 5.9e-6 / 1.6 | 2.9e-6 / 1.6 |
| Nu | **14.05** | 9.51 | 9.8 |

무활주 벽이면 u_h,rms ∝ z, w_rms ∝ z² 인데 legacy 는 u_h 가 벽까지 평탄하고 w ∝ z — **미끄럼 벽의 서명**. 원인은 `equation/momentum/legacy_ob/rhs.hpp`
머리말에 이미 적힌 결함("보존형 RHS + 짝 없는 이류형 음해항 — 운동량이 dt 에 비례해 샘, 새 코드에서 쓰지 말 것, rbc 전환 후 삭제 예정")이
큰 ν dt/Δz²(Pr 2547, dt 0.05, Δz_wall 1.5e-3 → ~10³)에서 벽 미끄럼으로 드러난 것. 소형 진단(48×48×96, t 200~400; `apps/rbc/input_rbc_diag_*.ini`,
`apps/rbc_nob/input_rbc_nob_diag_var_pr2547.ini`, 잡 36040 할당 안 `srun --overlap`):

| 케이스 | u_h 멱지수(셀 2~6) | w 멱지수(면 2~6) | u_h(k=1) | 벽 Nu | dt |
|---|---|---|---|---|---|
| legacy rbc, Pr 2547, dt_max 0.05 | 0.23 | 1.05 | 2.5e-3 | 17.1 | 0.050 |
| legacy rbc, Pr 2547, dt_max 0.005 | 0.44 | 1.32 | 7.1e-4 | 13.6 | 0.005 |
| legacy rbc, Pr 0.7 (CFL 이 dt 를 0.004 로) | 0.59 | 1.50 | — | 10.3 | 0.004 |
| 가변경로 ΔT=0, Pr 2547, dt_max 0.05 | 0.61 | 1.56 | 3.8e-4 | 12.7 | 0.050 |

→ dt 비례 누수가 맞다(dt/10 에 완화, Pr 0.7 의 작은 dt 에선 무활주 — 10-04 Kerr 검증이 통과했던 이유). 10-05 리팩터 뒤 회귀는 아니다
(rbc 스모크 Nu 30.6631 이 당시 게이트 값과 동일). 결정(사용자): **새 앱을 만들어 결과를 분리**한다.

## 3. 새 앱 `apps/rbc_ob` — 충실 상수물성 커널
- 운동량 `momentum::add_advection_diffusion_{u,v,w}_constant_property` + `solve_constant_property`(완전결합 δ-form ADI), 스칼라
  `scalar::solve_constant_property`(융합 ADI), **증분 투영**(P 누적, ∇Pⁿ 예측자) — DHVC 가 검증한 경로 그대로. IC 투영 1회(DHVC 와 동일).
- 루프 순서·IC·벽 ghost(U/V zero-ghost, W 벽면 0, T mirror)·T 내부 클립·체크포인트(U V W T P)·통계는 `rbc_nob` 과 1:1. 통계 모듈
  `rbc_nob/statistics`(Horn 2013 전 항목)를 심볼릭 링크로 공유(물성 필드 ≡ 1 로 한 번 채움) → 파일 형식이 NOB 와 같다(`rbc_nob_stats_*.dat`).
  설정은 rbc 와 같은 키(`rbc_settings.cpp` 1줄 TU). `delta_temperature` 는 0 만 허용(NOB 는 rbc_nob).
- 모니터: `theta_c Nu Tmin Tmax Nu_b Nu_t u_bot_max u_top_max sec/step`(D_× 없음).
- 스모크(32³, Pr 2547, 20 time): θ_c ≈ 0, Nu 77 (가변경로 ΔT=0 스모크 95 — 과도기 값, 예측자 유무 차).
- **Kerr 검증**(잡 36042, gpu02; Ra 1e6, Pr 0.7, 192×192×96, 통계 125~250): Nu 8.63, 벽 Nu 8.62/8.62, 윗/아랫 rms 비 0.98, 멱지수 0.67/1.54.
  legacy 앱은 같은 격자에서 8.87(격자외삽 8.33 = Kerr). 새 앱이 정상.
- 덱 `input_rbc_ob_pr2547_z256.ini`: NOB 본런 덱(`apps/rbc_nob/input_rbc_nob_dt40.ini`)과 같은 조건(냉시작, CFL 0.3, dt_max 0.05, 통계 16900 부터
  1000 간격, 표본 5 스텝, 체크포인트 50000 스텝), 종료만 27000(NOB base 확장통계 창 25300~27000 과 겹치도록). 잡 36041, gpu01, 2.4 h(16 ms/step).

## 4. 결과 — 세 경로의 벽 인접 rms (노트북 `apps/rbc_ob/plot/plot_rbc_ob.ipynb`)
창: NOB cold 16900~25000(w_rms 만) / NOB base 25300~27000 / NOB visc12 25300~26500 / 가변경로 ΔT=0 16900~27000 / rbc_ob 16900~27000.
"k" 는 벽에서 k번째 셀(w 는 k번째 내부 면). 비 = 윗벽/아랫벽.

| | NOB CFL dt (D_×≈40) | NOB visc12 (D_× 12) | 가변경로 ΔT=0 (같은 커널, D_× 35) | rbc_ob 상수 커널 |
|---|---|---|---|---|
| u_h,rms k=1 아래 / 위 | 3.6e-4 / **1.1e-1** (비 305) | 3.6e-4 / 4.6e-5 (0.13) | 1.74e-4 / 1.72e-4 (**0.99**) | 1.75e-4 / 1.74e-4 (**1.00**) |
| u_h,rms k=3 비 | 27 | 0.14 | 0.99 | 0.99 |
| w_rms 면 1 아래 / 위 | 5.9e-6 / **3.8e-3** (비 642) | 6.0e-6 / 8.9e-7 (0.15) | 2.82e-6 / 2.83e-6 (1.00) | 3.25e-6 / 3.23e-6 (0.99) |
| w_rms 면 3 비 | 122 | 0.16 | 0.99 | 0.99 |
| T_rms k=1 비 | 0.78 | 0.74 | 0.97 | 0.99 |
| 모니터 u_top_max / u_bot_max (끝) | 0.47 / 6.4e-4 | 9.9e-5 / 5.5e-4 | 2.9e-4 / 2.3e-4 | 2.6e-4 / 2.6e-4 |
| 멱지수 u_h / w (아랫벽) | 0.76 / 1.6 | 0.76 / 1.6 | 0.80 / 1.5 | 0.80 / 1.5 |
| Nu · θ_c | 9.51 · 0.133 | 9.51 · 0.133 | 9.98 · 0.009 | 9.88 · 0.006 |
| χ_λθ · χ_λu | 1.71 · (무의미) | 1.70 · 2.01 | 1.04 · 1.00 | 1.02 · 1.03 |

가변경로 ΔT=0 런(잡 36042, gpu02, 2.7 h, 19 ms/step)은 t 27000 완주, 창 16900~27000.
- **스파이크 상태 재시작 시험**(`input_rbc_nob_ob_limit_restart.ini`, 잡 36038): NOB ΔT 40 t 25000 체크포인트(u_top_max 0.46)에서 물성만 ≡1 로
  바꾸자 36 스텝 만에 1.8e-2, 11 time 뒤 2e-3, 100 time 뒤 5e-4 로 붕괴 → 아랫벽 값에 수렴. dt 는 즉시 cap 0.05(D_× 35)로.
- 가변경로 ΔT=0 냉시작은 CFL 0.3 이 dt 를 0.05 에 묶어 D_× = C_μ dt/(Δ_hΔz_wall) = 35 — NOB 임계(≈40) 바로 아래라 "같은 D_× 에서 μ̂ 만
  1 이면 안 생긴다"의 직접 증거는 아니지만, NOB 는 ΔT 20 에서 D_× 43·μ̂_top 2 로도 82배 스파이크였고 여기선 D_× 35·μ̂ 1 에서 1.00 이다.
  (dt 고정 초임계 시험 visc44/60 은 사용자 지시로 취소.)

## 5. 결론
1. **윗벽 rms 스파이크는 NOB(가변 μ̂) 경로에서만 생긴다.** 같은 커널(블록 커플링·압력 예측자 포함)·같은 격자·같은 CFL dt 에서 물성만 상수로
   두면 윗/아랫벽 rms 비가 전 항목 1.00 이고, 다른 구현(rbc_ob, 상수 커널)도 같다. 벽 인덱싱·투영·통계의 결함이 아니라
   §2 의 기전(상삼각 증분 커플링 사후보정 이득 D_× ∝ μ̂_top) 그대로 — 10-09 진단과 dt 기준(`viscous_coupling_number = 12`)이 유효.
2. 부수 발견: **legacy `rbc` 앱은 Pr 2547 급 고점성·큰 dt 에서 벽 미끄럼(Nu +45%)** — 머리말의 삭제 예정 결함이 현실화된 것. OB RBC 는
   앞으로 `rbc_ob` 를 쓴다(Kerr 8.63 통과). `apps/rbc` 는 하위호환으로만 남긴다.
3. 가변경로 ΔT=0 과 rbc_ob 의 벽 통계가 u_h 1%·w 15%(첫 면, 투영 예측자 차)·T 3% 안에서 일치, Nu 9.98 vs 9.88 → 상수물성 커널과 가변물성 커널(μ̂≡1)의 등가가 RBC 에서도 확인.

## 6. 파일
- 앱: `apps/rbc_ob/{main.cpp, rbc_settings.cpp, statistics.{hpp,cpp}→../rbc_nob/}`, 덱 `input_rbc_ob_{smoke,pr2547_z256,ra1e6_pr07_kerr}.ini`,
  sbatch `sbatch_gpu01_rbc_ob_pr2547.sh`, `sbatch_gpu02_rbc_ob_kerr_and_varlimit.sh`, 노트북 `plot/plot_rbc_ob.ipynb`.
- rbc_nob: `delta_temperature = 0` 허용(기본 PropertyModel → 정확 OB 극한), 덱 `input_rbc_nob_ob_limit_{restart,cold,cold_full}.ini`,
  `input_rbc_nob_smoke_ob_limit.ini`, `input_rbc_nob_diag_var_pr2547.ini`, sbatch `sbatch_gpu01_rbc_ob_control.sh`.
- rbc(legacy): 통계에 T_rms·u_rms·v_rms, 모니터에 u_bot_max·u_top_max 추가; 덱 `input_rbc_ob_pr2547_z256.ini`, `input_rbc_diag_*.ini`,
  `diag_run_in_alloc.sh`, sbatch `sbatch_gpu01_rbc_ob_pr2547.sh`.
- 통계: `apps/rbc_ob/stats_ob_pr2547_z256/`, `stats_ob_ra1e6_pr07_kerr/`; `apps/rbc_nob/stats_nob_ob_limit_{cold,restart}/`; `apps/rbc/stats_rbc_ob_pr2547_z256/`.

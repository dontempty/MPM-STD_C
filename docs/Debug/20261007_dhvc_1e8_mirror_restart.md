# DHVC Ra=1e8 — mirror 벽 재시작 (2026-10-07)

## 배경
- value 벽(1차 ghost, 참조 Fortran/PaScaL 관례) 4 GPU 런(잡 35641, `sbatch_dhvc_1e8_4gpu.sh`)이
  t≈135/150 에서 **Re_δ* 41.07 vs 참조 43.12 (−4.5%)**. 1e7 은 +0.55% 였으므로 1e8 만 벗어남.
- 의심 원인: ① 통계창이 t 100~150 으로 짧고 u_max 가 저주파로 진동(0.507~0.508), ② 1e7 warm-start 과도,
  ③ 1차 벽(마찰 절반, 벽이 dz/2 안쪽으로 밀림) — Ra 가 커질수록 벽층이 얇아져 ③ 의 비중이 커질 수 있음
  (cavity 에서 같은 기전으로 Ra 와 함께 커지는 양의 편향을 mirror 로 해소한 전례,
  `docs/Design/20261005_boundary_condition_framework.md`).
- 사용자 결정: 가장 정확한 벽(mirror, 2차)으로 ck_00180000(t=121.6578)에서 재시작해 10 time 과도 후
  100 time 통계를 누적하고, 20 time 마다 통계 파일을 쓴다.

## 구현 (커밋 8b7deaf)
- `apps/dhvc/main.cpp`: `sync_velocity(U,V,W,sub,mirror)` — mirror 면 접선 U·V ghost 를
  `fill_z_wall_mirror`(ghost = −V₁), 아니면 `fill_z_wall_value`(ghost = 0). mirror 일 때
  `momentum::set_wall_closure`·`scalar::set_wall_closure(…, Field::T)` 로 벽 행 닫기를 B −= A 로
  바꾼다(mirror ghost + drop 닫기는 dt·ν/dx² ≳ 3 에서 불안정 — cavity 교훈). 덱 `[boundary]
  velocity_ghost = mirror` 로 선택, 기본은 value(기존 결과 불변).
- CPU 스모크(np=2, 32×16×32, 60 스텝): div 1.8e-15, 발산 없음. GPU 는 `build_gpu/bin/dhvc`
  재빌드(gpu02; 실행 중인 35641 의 `build_gpu_d1` 은 건드리지 않음).

## 덱 `input_dhvc_1e8_4gpu_mirror.ini`
| 키 | 값 | 뜻 |
|---|---|---|
| restart_from | restart_1e8/dhvc_ck_00180000.bin | t = 121.6578, value 벽 런의 장 |
| end_time | 231.66 | 121.66 + 과도 10 + 통계 100 |
| stats_start_time | 131.66 | 과도 10 time 뒤부터 누적 |
| stats_interval | 20 | 파일 at t = 151.66, 171.66, 191.66, 211.66, 231.66(최종) |
| stats_dir | out_1e8_mirror/ | 파일 `dhvc_stats_<step>.dat`, 머리글에 running Re_δ* |
| checkpoint | 40000 스텝 → restart_1e8_mirror/ | 16 GB/개, 프루닝 없음(수동 관리) |

통계는 **누적형**(131.66 부터 dt 가중 U(z) 평균) — 20 time 간격 파일은 창이 길어질수록 Re_δ* 가
어디로 수렴하는지 보는 용도. 구간별 평균이 필요하면 인접 파일을 Σdt 가중으로 빼서 만든다
(`sum_dt` 는 머리글에 없으므로 로그의 dt 로 복원).

## 실행
- 잡 **35834** = `sbatch --dependency=afterany:35641 apps/dhvc/sbatch_dhvc_1e8_4gpu_mirror.sh`
  (gpu01+gpu02, npz=4, hpcx 2노드 mpirun). 처음엔 35641 완주(~3 h) 뒤 자동 시작으로 걸었으나
  **사용자 결정으로 35641 을 t=136.65 에서 scancel**(2026-10-07 10:25) → 35834 즉시 시작.
  value 벽 기준선은 t=125 스냅샷 `out_1e8/dhvc_stats_00185448.dat`(창 100~125, Re_δ* 42.33)과
  로그의 누적값(t=136.6 에서 40.98)만 남는다. 마지막 체크포인트 `restart_1e8/dhvc_ck_00200000.bin`.
- 예상 비용: dt ≈ 0.0006, 0.615 s/step → 110 time ≈ 1.8e5 스텝 ≈ **31 h**.
- 재시작 경로 점검: `post::read_checkpoint` 는 루트 fread + 랭크별 Send(MPI-IO 아님) → NFS
  MPI-IO 데드락 무관. 통계 누적기는 체크포인트에 없으므로 value 런 통계가 섞이지 않는다.

## GPU 스케일링 실측 (같은 격자 1536×340×768)
| 구성 | 잡 | s/step(중앙값) | 비 |
|---|---|---|---|
| 2 GPU (gpu02, npz=2) | 35594 | 1.206 | 1.00 |
| 4 GPU (gpu01+gpu02, npz=4) | 35641 | 0.615 | **1.96×** |

z 분할만 쓰는 pencil 구조에서 거의 선형 — 노드 간 통신(hpcx IB)이 병목이 아님.

## 판정 기준(완주 후)
- 5개 파일의 Re_δ* 추이가 창 길이에 따라 수렴하는지(±1% 안 흔들림) → 통계창 가설 판정.
- 최종 Re_δ* 대비 참조 43.12: value 벽 최종(35641 `out_1e8/`)과 mirror 최종을 나란히 두면
  벽 차수 효과를 분리할 수 있다. 1e7(+0.55%, value 벽)과의 일관성도 함께 본다.

## 첫 통계 파일 (t 131.66~151.66, 20 time) — 2026-10-07 18:24
Re_δ* **37.32 (−13.5%)**. value 런 t=125 파일(창 100~125, 42.33)과 분해: u_max 0.5354→0.4918(−8.1%), δ* 0.00661→0.00635(−3.9%).
- u_max −8% 는 저주파 진동 위상: 10 time 블록 평균 120~130 0.513 / 130~140 0.506 / **140~150 0.477(전 구간 최저)** / 150~ 0.488.
  value 런 u_max 궤적은 136.65 까지 mirror 와 0.3% 안에서 일치 → 여기까지는 벽 효과 아님.
- δ* −4% 는 벽 근처 프로파일이 더 찬 형상(첫 6셀 U/u_max 가 mirror 에서 일률 +3~3.5%). 첫 셀 속도는 거의 같은데
  (0.0337 vs 0.0321) 이산 벽 전단은 mirror 가 2배(U₁/z₁ vs U₁/2z₁, z₁=2.5e-4, z₁⁺≈0.3). 흐름이 벽 전단 2배를 첫 셀 속도
  감소로 흡수하지 않은 점은 운동량 수지(벽 전단 vs ∫⟨T⟩ vs 중앙 ⟨uw⟩)로 확인해야 하나 이 런의 통계엔 U(z)뿐.
- 판정 보류: 다음 파일(171.66)에서 u_max 가 0.50 대로 돌아오면 진동, 0.48 아래 지속이면 mirror 효과 → 1e5 value/mirror 짝 비교.

## 통계 확장 (커밋 3826afb) — 이후 런부터
`dhvc_stats_*.dat` 열: `z U V W T urms vrms wrms trms uw uT wT` (dt 가중 면평균; uw·uT·wT 는 슬롯곱 U x-면·W z-면·T 중심,
보간 없음, 평균곱 차감). 헤더: `t_start sum_dt nu alpha Re_dstar u_max delta_star Nu_hot Nu_cold u_tau_hot u_tau_cold
mean_dPdx velocity_ghost`. u_τ 자는 ghost 정책대로(mirror 벽면↔첫 중심, value ghost 중심↔첫 중심). mean_dPdx = 창 평균
**스텝별 적용** 등가 ∂P/∂x(누적 mean_dPdx 아님 — DHVC 는 add_mean_pressure_gradient 를 안 쓴다). 모니터 줄에 순간
Nu_hot Nu_cold u_tau_hot u_tau_cold 추가(헤더 이름으로 노트북이 파싱 — 구형 7열 로그도 읽힘).
노트북 §4: T·rms·플럭스·**열 수지 q=⟨wT⟩−α dT/dz (=Nu·α 평평해야)**·**운동량 수지 τ=ν dU/dz−⟨uw⟩+∫(T−dPdx)dz (=u_τ,hot² 평평해야)**·
창 길이별 Re_δ*. CPU 스모크(32×16×32, mirror/value): Nu 1.000(전도), 열·운동량 열 정상, NaN 없음. GPU 빌드 잡 35906(큐).
⚠ 지금 도는 35834 는 구형 통계 — 확장 통계가 필요하면 ck_00200000(t=134.78) 또는 다음 체크포인트에서 새 바이너리로 재시작(사용자 결정).

## DHVC 참조 신설 — Ng, Ooi, Lohse & Chung (2017) JFM 825 (`docs/Paper/DHVC.pdf`), 2026-10-07
사용자 지시: "기존 PaScaL_TCS 도 유지하고 논문에 나와 있는 상관식을 같이 그려봐". 파일(`apps/dhvc/plot/`):
- `ng2017_correlations.csv` — 본문 상관식: **Re_δ* ≈ 0.23·Ra^0.28** (overall, fig.10; 정의 U=max(ū), δ*=∫(1−ū/U)dz 가 우리 dhvc 와 동일),
  Re_δ*,l ≈ 1.3·Ra^0.20 (low-|Ri_f| 조건평균, 참고용), Nu ~ Ra^0.32 / Re_rms ~ Ra^0.46 (지수 본문; 접두계수 0.0772 / 0.6455 는
  fig.8 디지타이즈에 지수 고정 피팅), 조건평균 지수(Nu_l 0.37·Nu_h 0.34·Re_l 0.41·Re_h 0.50).
- `ng2017_fig10_re_dstar.csv` — fig.10 검은 점 7개(Ra 1e5·5.4e5·2e6·5.4e6·2e7·1e8·1e9 → 5.25·8.03·12.83·18.24·29.4·42.6·58.6).
  300 dpi 렌더를 축 틀로 보정해 디지타이즈(앞 4점 원 중심 ±1.5%, 뒤 3점은 흰 채움 파란 원에 가려져 아랫변에서 복원 ±3%).
- `ng2017_fig8.csv` — fig.8 검은 점(Nu, Re_rms) 7개.
- 발견: **PaScaL_TCS.csv 는 이 fig.10 점들을 로그 보간한 것과 일치**(1e8 43.12 ↔ 42.6, 1e9 62.96 ↔ 58.6)하고, MPM-STD_C RUN_SUMMARY 의
  타깃(1e5 5.78·1e6 11.0·1e7 20.9)은 **상관식 0.23·Ra^0.28 값**이다. 상관식은 점들의 유효 멱법칙이라 1e9 점은 상관식보다 23% 아래,
  1e5 점은 9% 아래 — 상관식과 점 사이에 ±10% 차가 원래 있다.
- 노트북: 검증 그림이 2 패널(Re_δ*, Nu)로, PaScaL(검은 네모)·Ng 점(회색 마름모)·Ng 상관식(실선, low-|Ri_f| 점선)·우리 점. 표에 세 편차 병기.

| case | Re_δ* | vs PaScaL | vs Ng fit 0.23Ra^0.28 | vs Ng DNS 점 |
|---|---|---|---|---|
| 1e5 | 4.790 | −10.9% | −17.1% | −8.8% |
| 1e6 | 10.024 | −3.9% | −8.9% | – |
| 1e7 | 23.080 | +0.6% | +10.0% | – |
| 1e8 value (100~125) | 40.98(로그 누적 136.6) | −5.0% | +2.5% | −3.8% |
| 1e8 mirror (131.7~151.7) | 38.31 | −11.2% | −4.2% | −10.1% |
→ 어느 참조로 보느냐에 따라 1e7 이 +0.6% 또는 +10%, 1e8 이 −5% 또는 +2.5%. 가장 직접적인 비교는 같은 Ra 의 논문 DNS 점(1e5·1e8·1e9).

## 두 번째 통계 파일 (t 131.66~171.66, 40 time) — 2026-10-07 23:10
**Re_δ* 41.24** (PaScaL 43.12 대비 −4.4%, Ng 2017 상관식 39.97 대비 +3.2%, Ng DNS 점 42.6 대비 −3.2%).
20 time 창 37.32 → 40 time 창 41.24. 구간별로 빼면 151.66~171.66 만의 평균은 u_max 0.5605, δ* 0.00676, **Re_δ* 45.27**.
u_max 10 time 블록: 140~150 **0.477(최저)** → 150~160 0.521 → 160~170 **0.586(최고)**. 즉 첫 파일의 −13.5% 는 저주파 진동의
골에서 창을 시작한 탓이고, **mirror 벽이 흐름을 느리게 만드는 것이 아니다**(진동 가설 확인). 판정은 예정대로 100 time 창(231.66)
파일로 — 창이 길어지며 Re_δ* 가 41~42 근처로 수렴하는지 본다.

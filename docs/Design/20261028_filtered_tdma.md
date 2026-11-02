# Filtered_TDMA 백엔드 추가와 채널 통계 검증 (2026-10-28)

## 1. 무엇을 했나

- `/scratch/t639a05/Filtered_TDMA` 의 `libs/filtered_tdma` (커밋 `11a2c2e`, 라이브러리 폴더는 작업 트리도 깨끗)를
  **수정 없이** `src/linear_solver/tdma/Filtered_TDMA/{include,cpu,gpu}` 로 vendor 했다 (`diff -r` 동일).
  상대 include(`../include/...`)가 그대로 맞도록 폴더 구조를 유지했다.
- 백엔드는 **빌드가 정한다** (2026-10-04 결정 "입력 키가 아니라 코드가 정한다"를 따름):
  - 기본 `make GPU=1 …` → PaScaL (지금까지와 같다), `build_gpu/`
  - `make GPU=1 … TDMA=filtered` → 운동량·스칼라 ADI 가 Filtered, `build_gpu_filtered/` (CPU 는 `build_filtered/`)
  - `TDMA` 에 다른 값을 주면 make 가 거부한다 (오타가 조용히 PaScaL 이 되지 않게).
  - 모든 앱의 로그 첫 줄에 `(N ranks, tdma=pascal|filtered)` 가 찍힌다.
- **압력 z TDMA 는 백엔드와 무관하게 PaScaL** (`linear_solver::ExactTdmaPlan`). Poisson 의 (0,0) 모드 시스템은
  대각지배가 아니라(ρ = 1/2, 감쇠 없음) 절단 근사의 전제가 깨진다. Filtered_TDMA 의 채널 앱도 같은 선택이다.
- 절단 허용오차는 라이브러리 README §9-2 의 논문 기준 **eps = dt·Δ̄²** (Δ̄ = 축 전역 길이 / 전역 셀 수,
  스트레칭 축도 평균 간격)을 운동량·스칼라 ADI 가 스윕 직전에 축마다 넣는다 (`set_sweep_tolerances`).

## 2. 코드 변경

| 파일 | 변경 |
|---|---|
| `src/linear_solver/tdma/tdma.hpp` | `ExactTdmaPlan`(PaScaL 고정) 신설. `TdmaPlan` = 기본 빌드에선 `ExactTdmaPlan` + 무시되는 `set_tolerance`, filtered 빌드에선 Filtered 래퍼. `tdma_backend_name()`, `truncation_tolerance(dt, L, N)`. |
| `src/linear_solver/tdma/Filtered_TDMA/` | vendor (무수정) |
| `src/equation/pressure/solve.hpp` | `plan_z` → `ExactTdmaPlan` |
| `src/equation/momentum/solve.cpp`, `src/equation/scalar/solve.cpp` | `set_sweep_tolerances(solver, grid, sub, dt)` — 상수·가변물성 양쪽 |
| `Makefile` | `TDMA=filtered` → `-DMPMSTD_TDMA_FILTERED`, 빌드 디렉터리 `_filtered` |
| `apps/*/main*.cpp` | 로그 첫 줄에 tdma 백엔드 |
| `tdma.hpp` (추가) | J 기록 (`record_truncation`, `truncation`), 진단용 `MPMSTD_FTDMA_EPS_SCALE` |
| `apps/channel_ftdma/` | 검증 전용 앱 (J 모니터) |

Filtered 래퍼의 동작 (`tdma.hpp`):
- Filtered 는 `n_row`·좌우 이웃을 생성자에서 고정한다 → `create(nsys, comm)` 때가 아니라 **첫 풀이에서** 만든다.
  비주기(끝 랭크 이웃 = `MPI_PROC_NULL`)와 주기(랩) 는 이웃이 달라 객체를 따로 둔다.
- 계약 위반은 throw: nsys/nrow 가 첫 풀이와 다름, 분산 풀이(nprocs > 1)에서 랭크당 행 < 3 (커널이 행 J+1·(n_row−2)−J 를 짚음),
  `set_tolerance` 없이 풀이.
- 랭크 안에서 완결되는 축(nprocs = 1)은 Filtered 도 직렬 Thomas 로 폴백한다 → 채널(z 슬랩)에서 차이가 나는 곳은
  **벽 수직 z 운동량 스윕뿐**이다.
- Filtered 의 비계측 solve 는 통신 시간을 재지 않는다 → filtered 빌드의 `comm_seconds` 는 0 (확장성 측정 훅).

## 3. 검증 A — 결정론 대조 (LES 64³, Re_τ 180, vreman 0.12, mirror 벽, GH200)

덱(보관): `apps/channel_ftdma/archive_les64/input_channel_val_ftdma_det_*.ini` (원본 `input_channel_re180_les64_vreman012_mirror_gpu.ini`),
잡: `sbatch_val_ftdma_det.sh`(1154130), `sbatch_val_ftdma_det2.sh`(1154139), 비교: `archive_les64/plot/compare_ftdma.py det`.
채널 초기 섭동은 분할에 따라 달라지므로 비교는 같은 분할끼리만 한다.

| 비교 | 스텝 | max\|ΔU\| (상대) | max\|ΔP\| | 판정 |
|---|---|---|---|---|
| 변경 후 PaScaL vs 변경 전 PaScaL, 1x1x4 | 300 | 1.1e-13 (6e-14) | 1.5e-14 | 같은 바이너리 반복(1.9e-13)과 같은 수준 → **기본 빌드 수치 불변** |
| 같은 바이너리 두 번, 1x1x4 | 300 | 1.9e-13 (1e-13) | 1.6e-14 | GPU 반복 잡음 (통계 축소 atomicAdd 순서) |
| Filtered vs PaScaL, 1x1x1 (직렬 폴백) | 300 | 1.7e-13 (9e-14) | 1.2e-14 | 반복 잡음 수준 |
| Filtered vs PaScaL, 1x1x4 | 1 | 5.1e-7 (3.3e-7) | 7.5e-7 | 첫 스텝 eps = dt·Δz̄² ≈ 3.4e-5 보다 훨씬 작다 |
| 〃 | 10 | 1.8e-5 (1.1e-5) | 2.2e-6 | |
| 〃 | 100 | 7.1e-5 (4.2e-5) | 4.7e-6 | |
| 〃 | 300 | 9.9e-4 (5.3e-4) | 7.1e-5 | 모니터(u_τ, Re_τ, −dP/dx)는 출력 자리수까지 같음 |

→ 절단 오차가 매 스텝 들어가고 난류의 카오스가 키운다. 순간장의 비트 비교는 의미가 없고 **통계로 판정**해야 한다.

## 4. 검증 B — 256³ DNS Re_τ 180, 2x2x2 통계·J·속도 (`apps/channel_ftdma`)

검증 전용 앱 `apps/channel_ftdma` (channel 드라이버 사본 + J 모니터, 설정은 channel 것; README 참조)에서
`input_dns256_{pascal,filtered}_2x2x2.ini` — 박스 4π×2π×2, 256³, Re_b 2857, DNS, **2x2x2 = GH200 8 장 → x·y·z 세 축 모두 분산**.
20 만 스텝, 통계 2 만 스텝부터 1 만 스텝마다 → 구간 17 개. 잡 1154368(PaScaL, node1359-1360)·1154369(Filtered, node1706-1707), 동시 실행.
분석 `apps/channel_ftdma/plot/plot.ipynb` (구간 평균 ± 표준오차, 구간 u_τ 는 코드 정의 그대로 — value ghost 라 dmz = 2·z₁).

### 4.1 통계 — 통계 오차 안에서 같다

| 지표 | PaScaL | Filtered | Δ (F−P) | z |
|---|---|---|---|---|
| Re_τ | 180.886 ± 0.087 | 181.073 ± 0.081 | +0.10 % | +1.58 |
| U_c⁺ | 18.449 ± 0.013 | 18.424 ± 0.009 | −0.13 % | −1.55 |
| u_rms⁺ 정점 | 2.610 ± 0.002 | 2.609 ± 0.003 | −0.05 % | −0.31 |

프로파일 (128 개 z⁺ 점): max|z| / rms|z| — U⁺ 1.72 / 1.41, u_rms⁺ 1.73 / 0.88, v_rms⁺ 1.50 / 1.00, w_rms⁺ 1.96 / 1.22, −u′w′⁺ 1.73 / 0.80.
차이의 크기는 정점 대비 ≤ 0.32 %. **모든 지표 |z| < 2** → 두 백엔드의 통계는 표본 오차 안에서 구별되지 않는다.
MKM 1999 (Re_τ 180.56) 대비는 두 런이 같은 정도로 맞는다 (U⁺ 1.2 %, rms·−u′w′ 2.4~3.3 % — 백엔드 차이보다 10 배 큼).
최종 Re_τ (코드 출력, 누적): PaScaL 180.89, Filtered 181.05.

### 4.2 절단 깊이 J (Filtered, 모니터 2001 스텝 × 8 랭크 × 9 풀이)

| 성분 | x (128 행) | y (128 행) | z (128 행) |
|---|---|---|---|
| dU | 2–4 (평균 2.99) | 2–3 (2.64) | 7–12 (8.81) |
| dV | 2–3 (2.96) | 2–3 (2.65) | 6–11 (8.45) |
| dW | 2–4 (2.96) | 2–3 (2.44) | 8–16 (12.22) |

fallback 0 회. 경계 보정이 블록(128 행) 중 x·y 는 ~3 행, z 는 ~9–12 행만 전파된다. eps = dt·(L/N)²: x ≈ 3e-5, y ≈ 8e-6, z ≈ 8e-7.

### 4.3 속도

통계 파일(1 만 스텝마다) 수정 시각 간격 = 1 만 스텝 벽시계 (0.1 ms/step 해상도, 시작 비용 제외). 2 만 스텝 배수의 체크포인트 구간 제외 8 구간:

| | ms/step | 20 만 스텝 잡 벽시계 |
|---|---|---|
| PaScaL | 13.83 ± 0.05 | 46.0 분 |
| Filtered | 9.50 ± 0.00 | 31.2 분 |

→ **Filtered 가 스텝당 4.33 ms (31.3 %) 빠르다.** 두 런은 다른 노드쌍에서 동시에 돌았으므로 노드 간 차이가 섞인 값이다
(같은 노드 교차 측정은 하지 않음). 백엔드 차이는 운동량 ADI 의 TDMA(성분 3 × 축 3 = 9 풀이/스텝)뿐이다 —
PaScaL 의 축 통신자 Alltoallv ×2 + 전체 블록 후방 대입 → Filtered 의 이웃 1 회 교환 + J 행 보정.

### 4.4 이전 시험 (LES 64³, `apps/channel_ftdma/archive_les64/`, 사용자 지시로 중단)
z 분할 1x1x4·1x1x16 은 15 만 스텝 완주(최종 Re_τ PaScaL 182.23·182.24, Filtered 182.50·182.17), 1x1x8·2x2x2 는 중단.
2x2x2 결정론 대조: 기본 eps 에서 첫 스텝 max|ΔU| 1.0e-4 (x 의 eps 가 z 의 ~40 배), eps 배율 1e-3·1e-6·1e-10 → 2.8e-8·1.1e-10·2.4e-15
(진단용 환경변수 `MPMSTD_FTDMA_EPS_SCALE`; 로그 첫 줄에 표시) — eps 에 비례해 기계정밀도까지 수렴 = 분산 순환 경로 구현 정확.

## 5. 산출물 이름 규칙 (겹침 방지)

검증은 전용 앱 `apps/channel_ftdma` 에서 관리한다. 꼬리표 `<case>_<backend>_<분할>` (예: `dns256_filtered_2x2x2`) 하나로
통계 `stats_<tag>/`(+ `truncation_J.dat`), 체크포인트 `checkpoints_<tag>/`, 로그 `log/<tag>.log` 를 구분한다 (앱의 `.gitignore`).
§3 의 결정론 대조와 LES 64³ 시험 덱·결과는 `apps/channel_ftdma/archive_les64/` 로 옮겼다 (`apps/channel` 은 원래대로).

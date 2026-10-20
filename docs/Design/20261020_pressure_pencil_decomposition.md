# 압력 Poisson 3차원(펜슬) 분할 지원 (2026-10-20)

## 변경
- 이전: `pressure::Solver` 가 npx = npy = 1(z-슬랩)만 허용 → 모든 앱이 z 분할만 가능.
- 지금: npx·npy > 1 이면 같은 z 블록의 xy-평면 그룹(P = npx·npy, `MPI_Comm_split(cart, z.rank)`) 안에서 alltoallv 전치:
  블록 →① x-펜슬(x 전부 × y 를 P 등분) → x FFT(주기 r2c / 벽 거울 DCT) →② y-펜슬(x 모드를 P 등분 × y 전부) → y FFT
  → z 분산 PaScaL_TDMA(원래 z 통신자, 로컬 모드 조각) → 역 y FFT →②⁻¹ → 역 x FFT →①⁻¹ 블록.
  z 는 처음부터 원래 z 분할 그대로. (0,0) 모드 핀은 모드 오프셋 0 인 랭크만, λ_x 는 모드 오프셋 포인터.
- 슬랩(npx = npy = 1)은 기존 코드 경로 그대로(전치 없음).
- 파일: `src/equation/pressure/pencil_transpose.{hpp,cpp}`(호스트 전치, 양 빌드 공용), `solve.{hpp,cpp}`(펜슬 분기),
  `system/system.{hpp,cpp,cu}`(`build_z_modes_system` 에 `pin_mean` 인자).
- 조건: ny ≥ npx·npy, x 모드 수(nx/2+1 또는 벽 nx) ≥ npx·npy — 생성자가 검사. 불균등 분할 허용(split_evenly).
- GPU 빌드: 전치는 호스트에서(장·스펙트럼 다운로드 → MPI → 업로드). 정확성 우선 구현 — 성능 최적화(디바이스 팩·CUDA-aware MPI)는 미착수.

## 검증 (`apps/channel_pencil_test`)
channel 드라이버 사본 + **분할 무관 IC**(잡음 = 고정 시드 + 전역 (성분, i, j, k) SplitMix64 해시). 36 × 30 × 32(4 로 안 나누어떨어지게),
mirror 벽, dt 0.02 고정, 30 스텝 → 마지막 체크포인트(U V W P)를 직렬 1x1x1 과 비교 (`run_compare.sh`, `compare_checkpoints.py`).

| 분할 (npx×npy×npz) | max|ΔU| | max|ΔP| |
|---|---|---|
| CPU 1x1x4 (슬랩) | 2.2e-14 | 1.1e-15 |
| CPU 4x1x1 · 1x4x1 · 2x2x1 · 2x1x2 · 1x2x2 | 4.8~5.1e-14 | 1.3~1.6e-15 |
| CPU 2x2x2 · 4x2x1 (8 랭크) | 1.0e-13 | 2.0e-15 |
| GPU 1x1x2 · 2x1x1 · 1x2x1 (2×A100, vs CPU 직렬) | 7.9e-14 | 1.7~1.8e-15 |
전부 반올림 수준 → 펜슬 경로 정확. 모니터(u_τ, Re_τ, −dP/dx)도 전 케이스 같은 자리수.

## 남은 것
- x 벽(DCT) 펜슬 경로는 코드는 있으나 이 시험(채널, x 주기)으로는 미검증 — cavity 는 ny = 1 이라 xy 분할 불가.
- npy > 1 을 막는 다른 모듈: `post/statistics` 스팬 평균(npy = 1), 경계층 재순환·공간 통계·DHVC 스펙트럼(npx = npy = 1). 채널·RBC·DHVC 본 루프는 영향 없음.
- 성능: 이 시험의 ms/step 은 노드 공유 상태라 비교 무의미 — 스케일링 측정 별도 필요.

## 정정 (2026-10-20 오후) — 10-18 최적화 코드 위로 다시 맞춤
- 첫 구현(38210f5)은 10-18 최적화 커밋(4c95c67, ed1f584)이 없는 클론에서 작성·검증됐다. rebase 때 `solve.cpp` 에 충돌 표시가 커밋됐고,
  펜슬 분기는 지워진 band_d·팩/언팩과 옛 평균 제거를 쓰고 있었다. 위 검증 표는 최적화 이전 코드에 대한 것이었다.
- 수정: 슬랩 분기 = 10-18 최적화 경로 그대로(`build_z_modes_system(..., pin_mean = true)` 만 추가). 펜슬 분기 = 새 레이아웃 —
  TDMA 를 `spectrum_xy` 에서 바로 풀고, `pin_mean_row` 와 DC 모드 합(`dc_mode_sum`)은 모드 오프셋 0 인 랭크만, 평균은 블록으로 되돌릴 때 뺀다
  (`x_lines_to_block(..., scale, subtract, ...)`).
- 함께 고침: 4c95c67 이 GPU `scalar/system.cu` 만 바꾸고 CPU `system.cpp` 의 x 스윕 `build_cell` 호출에 `source_at` 인자(0)를 안 넣어
  **CPU 빌드가 깨져 있었다** — 두 곳에 `0` 추가.
- 재검증 (이 클러스터, 수정 코드):
  - CPU 9 분할(1x1x1·1x1x4·4x1x1·1x4x1·2x2x1·2x1x2·1x2x2·2x2x2·4x2x1): 직렬 대비 max|ΔU| ≤ 1.0e-13, max|ΔP| ≤ 2.0e-15.
  - **슬랩 비트 동일**: 펜슬 이전 최적화 커밋(8a19ba4, 같은 CPU 1줄 수정)으로 빌드한 같은 앱의 1x1x1·1x1x4 체크포인트와 `cmp` 동일.
  - GPU 3 분할(1x1x2·2x1x1·1x2x1, A100 2장): CPU 직렬 대비 max|ΔU| ≤ 8.0e-14.
  - GPU 슬랩은 비트 비교 불가 — **같은 GPU 바이너리를 두 번 돌려도 체크포인트가 다르다**(통계 축소의 atomicAdd 순서, 펜슬 이전 빌드도 같음).
    이전 빌드 대비 차이 3e-15 이고 모니터 출력은 자리수까지 같다.

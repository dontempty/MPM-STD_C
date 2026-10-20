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

# figures/pressure — 압력 솔버 weak scaling 그림의 측정 조건

이 폴더의 그림(P1–P6)과 pressure_table.md 는 `plot/present_pressure.py` 가
`results/` 의 네 결과 폴더에서 만든다. 아래에 그림 → 결과 폴더 → 잡 → 측정 조건 순서로 정리한다.
숫자 자체는 pressure_table.md 에 있다.

## 1. 그림 파일과 데이터 출처

| 파일 | 내용 | np | 결과 폴더 (study = weak_512_pencil) |
|---|---|---|---|
| P1_np1-256_pressure_time.png | pressure_solve ms/step (랭크 max). 참고선 = 전치 4 회 경로 | 1–256 | weak512_zyx_plane (np 1–4) + weak512_zyx_plane256 (np 8–256); 참고 weak512_zyx_selfcopy (np 8–64) |
| P2_np8-256_pressure_efficiency.png | 압력만의 weak 효율 E_p = P(8)/P(np) | 8–256 | 같음 (기준 np 8 = plane256 의 잡 1204174) |
| P3_np1-256_pressure_share.png | pressure_solve / step_total | 1–256 | weak512_zyx_plane + weak512_zyx_plane256 |
| P4_np1-256_pressure_breakdown.png | 압력 내부 누적 막대 (전치 MPI / 팩·언팩 / FFT·재배열 / z TDMA / 기타) | 1–256 | weak512_zyx_plane_phases (np 1–4) + weak512_zyx_plane256_phases (np 8–256) |
| P5_np8-256_pressure_internals.png | 전치 MPI · z TDMA MPI · 계산 선 그래프 | 8–256 | weak512_zyx_plane256_phases |
| P6_np8-256_pressure_stage_growth.png | np 8 → 256 압력 내부 단계별 증가량 | 8–256 | weak512_zyx_plane256_phases |
| pressure_table.md | 위 숫자 표 (잡 번호 포함) | 1–256 | 네 폴더 전부 |
| old_20261102/A*, B* | 이전 그림 (2026-11-02, np ≤ 64, `plot/present_np1_64.py`) | 1–64 | weak512_zyx_selfcopy / packopt / plane 과 각 *_phases |

P1–P3 과 표의 step·pressure 열은 **타이머 없는 런**(solver_phases = false, 반복 2 회 중 step_total 중앙값 런),
P4–P6 과 표의 전치 MPI·z TDMA MPI·계산·대기 열은 **단계 타이머 런**(solver_phases = true, 1 회)에서 온다.
두 종류는 별도 잡이라 같은 np 라도 잡 번호가 다르다 (np 8: 1204174 vs 1204205).

## 2. 결과 폴더별 조건

모든 폴더가 같은 덱 세트(`decks_zyx/`, 단계 타이머 런은 `decks_zyx_phases/`)와 같은 잡 스크립트
(`job_scal.sh`, `env_kisti6.sh`)를 쓴다. 차이는 **코드 버전(압력 솔버 경로)** 과 **측정 일시·노드**다.

| 결과 폴더 | 압력 경로 | 빌드 시각 / git | 측정 일시 | np (잡) | 반복 | 비고 |
|---|---|---|---|---|---|---|
| results/weak512_zyx_selfcopy | **전치 4 회** (pencil: block→lines, x→y, y→x, lines→block), self 블록 D2D 복사, 인덱스 표 팩 | 2026-11-02 21:27 / 47980c1 + 미커밋 8–12 파일 | 2026-11-02 21:27–21:43 | 1·2·4 (1201119), 8 (1201079), 16 (1201080), 32 (1201081), 64 (1201082) | 2 | P1·P2 의 "before" 참고선 (np 8–64 만 그림에 쓴다) |
| results/weak512_zyx_plane | **평면 묶음, 전치 2 회** (block→planes, planes→block) | 2026-11-02 22:40 / 4193820 + 미커밋 18 파일 (= 다음날 커밋 2e892e0 의 내용) | 2026-11-02 22:43–22:50 | 1·2·4·8 (1201468), 16 (1201470), 32 (1201472), 64 (1201486) | 2 | 그림에는 np 1–4 만 쓴다. 커밋 2e892e0 메시지의 "np 64 355.1 → 291.4" 가 이 폴더의 값 |
| results/weak512_zyx_plane_phases | 같음 + solver_phases = true | 같음 | 2026-11-02 22:43–22:50 | 1·2·4·8 (1201469), 16 (1201471), 32 (1201485), 64 (1201487) | 2 | P4 의 np 1–4 |
| results/weak512_zyx_plane256 | 평면 묶음, 전치 2 회 | 2026-11-03 10:11 / 2e892e0 + 미커밋 16 파일 | 2026-11-03 11:14–11:20 | 8 (1204174), 16 (1204160), 32 (1204159), 64 (1204158), 128 (1204157), 256 (1204156) | 2 | P1–P3 의 np 8–256. 같은 평면 경로인데 11-02 빌드보다 np 64 에서 ~11 ms 빠르다 (291 → 280; 원인 미확인, 다른 노드 세트) |
| results/weak512_zyx_plane256_phases | 같음 + solver_phases = true | 같음 | 2026-11-03 11:18–11:23 | 8 (1204205), 16 (1204199), 32 (1204185), 64 (1204184), 128 (1204183), 256 (1204182) | 1 | P4–P6 의 np 8–256. np 8 런의 전치 MPI 42 ms·대기 6.3 ms 는 np 16 보다 크다 (1 회 측정, 잡음 가능) |

관련 폴더 (이 그림에는 쓰지 않음): results/weak512_zyx_phases (selfcopy 의 단계 타이머 런, np 1–8, 잡 1201118),
results/weak512_zyx_packopt(_phases) (전치 4 회 + 구조화 팩, 2026-11-02 22:15, 잡 12013xx).

파일 이름 규칙: `weak_512_pencil_np<np>_n<노드>x<노드당 랭크>_r<반복>_j<잡>.{csv,log}`.
csv = 구간 표 (phase, min_ms, avg_ms, max_ms, share, imbalance; 랭크별 스텝당 평균의 랭크 min/avg/max),
log = 런 표준출력 (설정 에코 + 모니터 줄 + 구간 표). 잡별 환경 기록은 `<폴더>/env/environment_j<잡>.txt`
(노드 목록, git, 모듈, GPU, 바이너리 시각), Slurm 출력은 `log/scal_<잡>.out`, 게이트 로그는 `log/j<잡>/`.

## 3. 공통 측정 조건

### 문제 (덱 `decks_zyx/weak_512_pencil/weak_512_pencil_np<np>.ini`, `gen_decks.py --pencil-order zyx` 생성)

- rbc_nob (NOB 글리세롤 RBC): Ra = 1e6, Pr = 2547, ΔT = 40 K, perturb = 0.05, dt_max = 0.05, max_cfl = 0.3,
  viscous_coupling_number = 12, monitor_interval = 50.
- weak: 랭크(= GPU)당 512³ 격자, 랭크당 도메인 6 × 6 × 1. z 는 비균일 (uniform_z = false, gamma_z = 2.6).
- 분할: 3D cube 블록, z → y → x 순서로 2 씩 나눔. 노드 4 GPU 가 z 이웃이 되는 배치.

| np | 전역 격자 nx×ny×nz | 도메인 lx×ly×lz | npx×npy×npz | 노드 × 노드당 랭크 |
|---|---|---|---|---|
| 1 | 512×512×512 | 6×6×1 | 1×1×1 | 1×1 |
| 2 | 512×512×1024 | 6×6×2 | 1×1×2 | 1×2 |
| 4 | 512×1024×1024 | 6×12×2 | 1×2×2 | 1×4 |
| 8 | 1024×1024×1024 | 12×12×2 | 2×2×2 | 2×4 |
| 16 | 1024×1024×2048 | 12×12×4 | 2×2×4 | 4×4 |
| 32 | 1024×2048×2048 | 12×24×4 | 2×4×4 | 8×4 |
| 64 | 2048×2048×2048 | 24×24×4 | 4×4×4 | 16×4 |
| 128 | 2048×2048×4096 | 24×24×8 | 4×4×8 | 32×4 |
| 256 | 2048×4096×4096 | 24×48×8 | 4×8×8 | 64×4 |

- 시간 적분: max_step = 200, warmup_steps = 50 → **150 스텝 평균**. 단계 타이머 덱(`decks_zyx_phases/`)은
  `[timing] solver_phases = true` 한 줄만 다르다.

### 압력 솔버 (TDMA 백엔드 = PaScaL, 빌드 build_gpu/bin)

- x r2c FFT + y c2c FFT + 모드별 z TDMA (PaScaL_TDMA, 축소 시스템 all-to-all).
- 평면 묶음 경로 (npx·npy > 1 기본, 커밋 2e892e0): 블록 → xy 전체 평면 전치(MPI_Alltoallv, xy 그룹 npx·npy 랭크),
  x·y FFT 는 slab 커널, z 체인 = 전체 랭크, 평면 → 블록 전치. np 1·2 (npx = npy = 1)는 slab 경로.
- 전치 4 회 경로 (selfcopy 폴더): block→lines, x→y, y→x, lines→block, z 체인 = npz.
- 환경변수 (잡 env 기록): MPICH_GPU_SUPPORT_ENABLED=1, PASCAL_TDMA_MPI_MODE=device (디바이스 버퍼 직접 MPI),
  MPMSTD_A2A_SELF=copy (self 블록 cudaMemcpyAsync). MPMSTD_PRESSURE_PENCIL·MPMSTD_PENCIL_PACK 은 설정하지
  않음 = 기본값 (평면 경로, 구조화 팩; selfcopy 폴더의 빌드에는 두 스위치가 아직 없었고 인덱스 표 팩).
- MPI 집합 통신 = 기본 MPI_Alltoallv (pairwise 전송 패치는 apps/FFT_test 의 별도 측정, 여기엔 없음).

### 시스템 / 실행

- KISTI HANGANG GH200 (Grace aarch64 + Hopper, 노드당 GPU 4, 120 GB), Slingshot. 파티션 normal (sacct),
  QOS normal, `--exclusive`, `--ntasks-per-node=4 --cpus-per-task=72 --gres=gpu:4`. 1 랭크 = 1 GPU.
- 모듈: PrgEnv-nvidia 8.7.0, nvidia/26.3, cray-mpich 9.0.1 (GTL), cuda/12.9_26.3, craype-accel-nvidia90,
  libfabric 2.3.1 (`env_kisti6.sh`).
- 잡당 np 하나 (np 1·2·4 는 한 잡에서 노드 하나에 1·2·4 랭크로). 노드 목록은 env 파일에 있다 — 같은 np 라도
  잡마다 노드 세트가 달라 수 ms 차이가 난다.
- 게이트: 매 잡 시작 때 `decks_zyx/gate/` 로 rbc_nob 과 rbc_nob_scalability 를 60 스텝 돌려 모니터 줄이
  동일한지 확인 (`check_gate.py`, 모두 PASS). 측정 앱의 시간 루프 = rbc_nob 과 같은 호출 순서.

### 시간 측정 방식 (`main.cpp`, `common/phase_timer.hpp`)

- 구간 경계에서 cudaDeviceSynchronize 후 랭크별 누적; 런 끝에 랭크별 합을 MPI_Reduce(min/avg/max) 해
  스텝 수로 나눈 ms/step 을 CSV 에 쓴다. 그림·표는 **max_ms (랭크 max)** 를 쓴다.
  share = max / step_total(max), imbalance = (max − min)/avg.
- 반복 2 회면 step_total 중앙값 런 하나를 대표로 쓴다 (`plot/scaling_plot.collect`). 같은 np 에 배치가 여럿이면
  노드를 가장 많이 쓴 것.
- 단계 타이머 런의 압력 내부 항목: pressure.plane.block_to_planes / planes_to_block (전치 전체),
  pressure.comm.* (그 중 MPI_Alltoallv 호출 시간만, 코드 안 MPI_Wtime 누적; self 블록 비동기 복사는 제외),
  pressure.fft_x_fwd·fft_y_fwd·reorder_x_to_y·…·fft_x_bwd,
  pressure.build_z, pressure.tdma_z (안의 MPI = tdma_comm_pressure_z), pressure.mean_remove, pressure.halo.
  그림의 묶음: 전치 MPI = comm 합, 팩·언팩 = plane 합 − comm 합, FFT·재배열, z TDMA = build_z + tdma_z,
  기타 = mean_remove + halo. 항목마다 랭크 max 라 묶음 합이 pressure_solve(랭크 max)와 정확히 같지는 않다.
- 단계 타이머를 켜면 동기화가 늘어 느려질 수 있으나 이 측정에서는 step·pressure 가 타이머 없는 런과 1 % 안에서
  같다 (np 8: 229.0/59.4 vs 230.3/60.3; np 256: 305.4/123.5 vs 307.8/124.3).

## 4. 다시 만들기

```
cd apps/rbc_nob_scalability
/scratch/t525a06/KDH/.venv-plot/bin/python plot/present_pressure.py
```

새 결과(예: np 512)가 생기면 `present_pressure.py` 의 PLANE256 / PH_PLANE256 폴더에 같은 이름 규칙으로 두거나
폴더 상수를 바꾸고, `DECOMP` 에 분할을 추가한다.

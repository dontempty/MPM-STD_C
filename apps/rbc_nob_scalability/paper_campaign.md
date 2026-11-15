# 논문용 확장성 측정 세트 — pencil(cube) vs slab, weak 256³/512³ + strong 1024³/2048³, np 8–256 (2026-11-16 설계, 아직 제출 안 함)

기존 결과(weak 512³/GPU pencil, np 1–256, `plot/figures/pressure/README.md`)를 같은 조건으로 다시 측정하면서
**두 구현(pencil = cube 블록, slab = 1×1×np)** 을 같은 전역 격자에서 비교한다. 덱은 `decks_zyx/` (본 측정)와
`decks_zyx_phases/` (solver_phases = true, 압력 내부 분해)에 있고 `gen_decks.py` 머리의 명령으로 재생성된다.
기존 weak_512_pencil 덱(np ≤ 256)과 게이트 덱은 바뀌지 않았다 (diff 확인). `decks/` (기본 세트)도 바뀌지 않았다.

## 1. 설계

공통: 랭크 = GPU, 노드당 4 랭크, 셀 크기 = weak_512 와 동일 (dx = 6/512, z 비균일 gamma 2.6), 물리 = cv12 ΔT 40
(Ra 1e6, Pr 2547), 200 스텝 중 150 스텝 측정. 메모리: 512³/GPU ≈ 27 GB (GH200 120 GB), 2·512³ ≈ 54 GB 확실, 4·512³ ≈ 108 GB 시험용.

- **pencil**: zyx 순서로 2 씩 나눈 3D 블록 (8: 2×2×2, 16: 2×2×4, 32: 2×4×4, 64: 4×4×4, 128: 4×4×8, 256: 4×8×8).
  압력 = 평면 묶음 경로 (전치 2 회 + 전체 랭크 z 체인). 모멘텀·스칼라 x·y TDMA 가 MPI 를 탄다.
- **slab**: 같은 전역 격자를 1×1×np 로. 압력 = slab 경로 (전치 없음, x·y FFT 국소, z TDMA 체인 = 전체 랭크).
  x·y TDMA 는 MPI 없음, 대신 xy 면 halo 가 크고 z 블록이 얇아진다 (weak 512 np 256: 2048×4096×16).
  `--slab-mesh pencil` 로 만들어 **같은 np 에서 slab/pencil 의 전역 격자·도메인이 같다** (decks/ 의 weak_*_slab 은 z 로만 늘린 다른 문제).

| study | 전역 격자 | 도메인 | np | 랭크당 블록 (pencil / slab) | 비고 |
|---|---|---|---|---|---|
| weak_256_{pencil,slab} | 256³·np (np 8: 512³ … 256: 1024×2048×2048) | 6×6×1 × 분할 | 8–256 | 256³ / 512×512×64 … 1024×2048×8 | 3.4 GB |
| weak_512_{pencil,slab} | 512³·np (np 8: 1024³ … 256: 2048×4096×4096) | 6×6×1 × 분할 | 8–256 | 512³ / 1024×1024×128 … 2048×4096×16 | 27 GB; pencil 은 기존 결과와 같은 덱 |
| strong_1024_{pencil,slab} | 1024³ | 12×12×2 | 8–256 | 512³ → 256×128×128 / 1024²×128 → 1024²×4 | 기준 np 8 (= weak_512 np 8 덱). slab np 256 은 nz_local 4 (하한 8 → 4 완화) |
| strong_2048_{pencil,slab} | 2048³ | 24×24×4 | 16–256 | 1024×1024×512 … / 2048²×128 → 2048²×8 | **np 8 은 1024³/GPU ≈ 216 GB 라 불가**. np 16 (108 GB) 은 시험용, OOM 이면 기준 np 32 |

덱 폴더에는 np 1·2·4 (weak pencil, strong_1024 np 4) 와 np 512 덱도 남아 있다. 이번 범위(8–256) 밖이지만 노드 수로 np 가 정해지므로
-N 2 … 64 잡에서는 돌지 않는다.

효율 정의: weak E(np) = T(8)/T(np); strong E(np) = T(base)·base/(T(np)·np), base = 8 (1024³), 32 또는 16 (2048³).
slab 대 pencil 은 같은 np 의 T 비 (T_slab/T_pencil) 와 구간별(압력 / 모멘텀·스칼라 TDMA MPI / halo) 차이로 본다.

## 2. 잡 구성

job_scal.sh 는 노드 수로 np 를 정하고 STUDIES 의 모든 study 를 그 np 로 돈다 (덱이 없는 조합은 WARN 후 건너뜀 — strong_2048 np 8).
8 study × REPS=3 → 잡당 최대 24 런. 덱 세트 하나에 **6 잡 (N = 2 4 8 16 32 64)**, phases 세트까지 **12 잡**.
모두 **새 잡** → `sbatch --test-only` (통과 확인함, 2026-11-16) → 사용자 확인 → 시험 잡 1 개 (-N 2) → COMPLETED 뒤 나머지.

| -N | np | 런 (REPS=3) | 예상 | 비고 |
|---|---|---|---|---|
| 2 | 8 | 7 study × 3 = 21 | ~20 분 | strong_2048 없음. 시험 잡 |
| 4 | 16 | 24 | ~25 분 | strong_2048 np 16 (108 GB) 여기서 OOM 여부 확인 |
| 8 | 32 | 24 | ~25 분 | |
| 16 | 64 | 24 | ~25 분 | 여기까지 ≤ 64 GPU, 자유 사용 |
| 32 | 128 | 24 | ~25 분 | 64 GPU 초과 — 다른 사용자와 일정 조율 |
| 64 | 256 | 24 | ~25 분 | 조율 필요 |

한 런이 실패해도 잡은 다음 런으로 넘어가고 끝에 실패 목록을 찍는다 (exit code = 실패 수). 실패 CSV 는 쓰지 않는다.
환경변수는 기존과 같은 기본값 (PASCAL_TDMA_MPI_MODE=device, MPMSTD_A2A_SELF=copy, 평면 경로, 구조화 팩). 환경 기록
(`results/<RESULTS>/env/environment_j<잡>.txt`)에 MPMSTD_PRESSURE_PENCIL·MPMSTD_PENCIL_PACK·FI_MR_CACHE_MONITOR 값이 추가로 남는다.

## 3. 제출 명령 (apps/rbc_nob_scalability/ 에서, glogin 노드)

```bash
STUDIES="weak_256_pencil weak_256_slab weak_512_pencil weak_512_slab strong_1024_pencil strong_1024_slab strong_2048_pencil strong_2048_slab"

# 1) 시험 잡 (np 8, 2 노드) — COMPLETED + 실패 0 확인 후 다음
STUDIES="$STUDIES" REPS=3 DECKS=$PWD/decks_zyx RESULTS=$PWD/results/paper_zyx \
sbatch -N 2 --time=01:00:00 --job-name=paper_N2 job_scal.sh

# 2) ≤ 64 GPU: N = 4 8 16 (3 잡)
for n in 4 8 16; do STUDIES="$STUDIES" REPS=3 DECKS=$PWD/decks_zyx RESULTS=$PWD/results/paper_zyx \
  sbatch -N $n --time=01:00:00 --job-name=paper_N$n job_scal.sh; done

# 3) 128·256 GPU: N = 32 64 (2 잡) — 일정 조율 후
# 4) phases 세트 (압력 내부 분해, 1 회): DECKS=$PWD/decks_zyx_phases RESULTS=$PWD/results/paper_zyx_phases REPS=1 로 1)–3) 반복
```

결과: `results/paper_zyx/<study>/<study>_np<np>_n<노드>x4_r<k>_j<잡>.{csv,log}`.

## 4. 추가 측정 (앞 논의, 같은 흐름으로 뒤에)

- 전치 4 회 참고선을 128·256 까지: weak_512_pencil 만, `MPMSTD_PRESSURE_PENCIL=transpose4`, RESULTS=results/paper_zyx_transpose4 (N = 2 … 64, 6 잡).
- 노드 내 vs 노드 간: `NP_LIST="2/1 4/1"` 로 -N 4 잡 1 개 (np 2·4 를 노드당 1 랭크로).
- 단일 GPU nsys 프로파일 (np 1, 512³·256³) 1 잡; 검증 그림은 rbc_nob 본런 결과로.

## 5. 그림 (측정 뒤)

- weak 256/512 × pencil/slab: 시간·효율 (기준 np 8), 압력 비중, slab/pencil 시간 비.
- strong 1024/2048 × pencil/slab: speedup·효율 (기준 8 / 32).
- 구간 분해: 압력 내부 (present_pressure.py 의 internals·loss_chart), 모멘텀·스칼라 TDMA MPI, halo — slab 이 어디서 이득·손실인지.
- 반복 3 회의 중앙값 + min–max 오차 막대.

## 6. 바뀐 파일 (2026-11-16)

- gen_decks.py: `--strong-ref` (strong 도메인 = 6×6×1 × n/REF), `--slab-mesh pencil` (slab 전역 격자 = pencil), `--min-local`.
  기본 옵션 출력은 변하지 않음 (decks/ 와 diff 동일).
- job_scal.sh: 파티션 exclusive02 (+ account bs0242, qos normal), NP_ALL 기본값에 128 256 512, 환경 기록에 압력 경로 스위치 추가.
- decks_zyx/, decks_zyx_phases/: weak_256_pencil, weak_256_slab, weak_512_slab, strong_1024_pencil, strong_1024_slab,
  strong_2048_pencil, strong_2048_slab, weak_512_pencil_np512, gate_np512 + gate_ref_np512 추가.

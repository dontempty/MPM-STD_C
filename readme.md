# MPM-STD

벽 난류·열대류용 **비압축성 Navier–Stokes DNS/LES 코드** (C++17 / MPI / CUDA).
MPM-STD_C(Fortran 참조 코드의 1차 C++ 이식)를 "읽기 쉽게" 다시 쓴 것으로, 같은 수치 방법을 유지하면서
모듈 경계·이름·설정·출력 규약을 정리했다. 최종 목표는 이 코드로 **도심풍(urban wind) 해석**을 돌리는 것이고,
그 전 단계로 채널·열대류·경계층 벤치마크를 하나씩 검증해 가고 있다 (`docs/Plan/Goal.md`).

- 저장소에는 **코드·입력 덱·sbatch 스크립트·노트북·참조 데이터만** 올린다. 런 산출물(통계·로그·체크포인트·그림)과
  빌드 디렉터리는 전부 로컬 전용이다 (`.gitignore`).
- 모든 문서와 주석은 한국어, 식별자는 영어 전체 단어(축약 금지: `domain`, `momentum`, `calculate_*`).

---

## 1. 수치 방법 한눈에

| 항목 | 내용 |
|---|---|
| 격자 | 직교 **staggered(MAC)** 격자. U = x-면, V = y-면, W = z-면, P·T = 셀 중심. x·y 균일, z 는 tanh 스트레칭 가능 (`[stretch] gamma_z`). 2D 는 ny = 1 로 축을 붕괴시켜 푼다. |
| 공간 이산화 | 2차 중심차분. 식의 정본은 `*_stencil.hpp` 하나(CPU/GPU 공용 헤더, `MPMSTD_HD`). |
| 시간 적분 | 운동량: **Beam–Warming δ-form ADI + 속도 블록 결합**(참조 코드 1:1). 스칼라(온도): 융합 ADI 음해. 양해 대류 → CFL 로 dt 결정(`max_cfl`, `dt_max`). |
| 압력 | 증분 투영. Poisson 은 x·y **FFT**(주기) 또는 x **DCT**(벽; 거울 확장 2N FFT) + z **분산 TDMA**(PaScaL_TDMA). |
| 물성 | 상수물성(OB) 전용 경로 `*_constant_property` 와 가변물성(NOB·LES) 경로 `*_variable_property` — 스텐실은 물성 접근자 템플릿으로 하나. |
| 경계조건 | `[boundary]` 입력 → `boundary::BoundaryCondition` 표(필드×축×면). 축 kind = periodic / wall / dirichlet / neumann / inflow_outflow, 면별 override(숫자·adiabatic·slip·continuity·freestream). 접선속도 벽 ghost = `value`(1차, 참조 관례) 또는 `mirror`(2차; 음해 벽 행 닫기 `B −= A` 와 짝). |
| LES | 와점성 모델을 μ̂ = 1 + ν_t/ν 로 가변물성 운동량 커널에 결합. `physics/LES/constant_vreman/`(Vreman 2004), `physics/LES/WALE/`(Nicoud & Ducros 1999). 벽 ghost μ̂ = 1. |
| 병렬 | MPI 만(멀티스레드 금지). 3D Cartesian 토폴로지(`npx·npy·npz`), 현재 앱들은 **z 슬랩 분할**(npz) 위주. halo 폭 1, 면만 교환. |
| GPU | `.cu` 쌍둥이 TU. 배열은 전부 디바이스 상주(`cudaMalloc`), 통계 출력 때만 D2H. cuFFT + GPU TDMA. MPI = CUDA-aware hpcx. |
| 재시작 | `post/checkpoint`(U V W T P + mean_dPdx, 루트 fread → 랭크별 Send). 다른 격자에서 보간해 시작하는 **warm-start**(`warmstart_from`, staggered trilinear) 도 있다. 통계 누적기는 체크포인트에 없다. |

배열 레이아웃·ghost·루프 순서의 정본은 `docs/conventions.md` 다 (`idx = (i·ny + j)·nz + k`, k 가 가장 빠르다, 루프는 i→j→k).

---

## 2. 디렉터리 구조

```
MPM-STD/
├── Makefile, Makefile.inc      빌드 규칙 / 컴파일러·라이브러리 경로 (머신이 바뀌면 .inc 만)
├── readme.md                   이 문서
├── check_mpmc.sh               클러스터 노드별 남은 CPU/GPU 를 한 표로 (./check_mpmc.sh [gpu])
├── src/                        라이브러리 (앱이 공유하는 모든 수치 코드)
├── apps/                       실행파일 = apps/<이름>/main.cpp (테스트도 apps/test_*/)
├── docs/                       계획·설계·디버그 기록·논문 PDF
├── build/   build_gpu/         정본 빌드 디렉터리 (CPU / GPU) — obj 공유, bin 에 전 앱
└── build_gpu_d1/, build_gpu_m11/   과거 보존용 (돌고 있던 잡의 바이너리 보호) — 새로 만들지 말 것
```

### `src/` 모듈

| 폴더 | 역할 | 주요 파일 |
|---|---|---|
| `common/` | 타입(`real_t = double`), `MPMSTD_HD`, CUDA 체크, 필드 크기 검사 | `types.hpp`, `hd.hpp`, `cuda_check.hpp` |
| `config/` | INI 파서 + 스키마 바인딩(모르는 키·타입 불일치·범위 위반은 전부 throw) | `ini_parser.*`, `schema.*` |
| `field/` | `Field`(3D 배열, CPU/GPU 동일 API), `Buffer`(솔버 작업 버퍼), `View` | `cpu/field.hpp`, `gpu/field.hpp`, `buffer.hpp` |
| `grid/` | 축별 면·중심·간격 배열(호스트 + 디바이스 복사), tanh 스트레칭 | `grid.*`, `stretching.*` |
| `parallel/` | MPI 컨텍스트, Cartesian 토폴로지, 서브도메인(halo 교환, 전역↔로컬 변환) | `mpi_context.*`, `topology/`, `subdomain/` |
| `boundary/` | BC 표와 파서, 벽 ghost 채움(value/mirror/zero-gradient), 열린 경계(유입면·대류 유출·자유류), 재순환 유입(Lund 1998) | `boundary_condition.*`, `boundary_settings.*`, `wall.*`, `apply.*`, `open.*`, `recycling.*` |
| `equation/momentum/` | 운동량 RHS(대류·확산·부력 입력)와 δ-form ADI 풀이, 벽 행 닫기 | `rhs/`, `system/`, `solve.*`, `wall_rows.*`, `momentum_stencil.hpp`, `legacy_ob/`(간이 OB, 경고 머리말) |
| `equation/pressure/` | 발산 RHS, FFT/DCT + TDMA Poisson, 속도 보정·압력 누적 | `solve.*`, `solve_kernels.*`, `system/`, `correction.*` |
| `equation/scalar/` | 온도(스칼라) 융합 ADI | `solve.*`, `system/`, `scalar_stencil.hpp` |
| `linear_solver/tdma/` | 분산 TDMA(vendor: `PaScaL_TDMA/`, `Filtered_TDMA/`) 와 중간계층 `tdma.hpp` | `tdma.hpp` |
| `physics/` | 부력(축별), 채널 CFR 강제·벌크 속도 보정, 물성(NOB 글리세롤), LES | `buoyancy.*`, `channel_forcing.*`, `properties.*`, `LES/` |
| `post/` | 평면 평균·제곱·곱·열 가중 평균, 공간 발달 통계, 국소 \|Ri_f\| 분류, 1-D 스펙트럼, 체크포인트(+격자 보간) | `statistics.*`, `spatial_statistics.*`, `wall_richardson*`, `spectra*`, `checkpoint*` |
| `time/` | CFL 기반 dt | `time_step.*` |

CPU/GPU 규칙: 같은 이름의 `.cu` 쌍둥이가 있는 `.cpp` 는 GPU 빌드에서 제외된다. 양쪽에 공통인 오케스트레이션 TU(예: `pressure/solve.cpp`, `post/spectra.cpp`)는 `#ifdef MPMSTD_GPU_FFT` 로 FFT 라이브러리 호출만 가른다. 분기 스위치는 `MPMSTD_GPU` 하나다.

### `apps/` — 앱 하나 = 폴더 하나

각 앱 폴더는 `main.cpp`(+ 앱 소유 `settings.{hpp,cpp}` 또는 다른 앱 설정을 빌려오는 1줄 TU), 입력 덱 `input_*.ini`,
`sbatch_*.sh`, `plot/`(노트북·참조 데이터) 를 갖고, 런 산출물은 **앱 폴더 밑**(`stats_*/`, `out_*/`, `logs/`, `checkpoints*/`)에 쓴다.
`apps/framework.hpp` 는 외부 의존 없는 최소 테스트 프레임워크(`CHECK`, `CHECK_THROWS_MSG`)다.

---

## 3. 빌드

```bash
# CPU (로그인 노드에서 가능) → build/bin/<앱>
make                 # 전체
make dhvc            # 앱 하나 (make channel, make cavity, …)
make test            # apps/test_* 전부 실행, 하나라도 실패하면 exit 1
make clean

# GPU (.cu 가 있으니 GPU 노드에서; sbatch 로 gpu01/gpu02 에 -n 4~8 코어) → build_gpu/bin/<앱>
make GPU=1 -j8 dhvc
```

- 컴파일러·라이브러리 경로는 `Makefile.inc` 한 곳: CPU = 시스템 OpenMPI `mpicxx` + gcc 로 재빌드한 FFTW3(정적),
  GPU = nvhpc 23.7 `mpic++ -cuda -gpu=cc80,cuda12.2` + hpcx-2.15 MPI + cuFFT. **모든 TU 를 한 컴파일러로** 컴파일한다
  (nvcc fat binary 를 섞으면 main 전에 segfault).
- 빌드 디렉터리는 **`build/` 와 `build_gpu/` 둘만** 쓴다. `BUILD=` 변형은 돌고 있는 잡의 바이너리를 보존해야 할 때 말고는 금지.
  ld 재링크는 실행 중인 잡에 안전하다(기존 inode 유지).
- 주의: CPU 빌드 때 Intel oneAPI 환경을 source 하지 말 것(libmpi 혼입). 다른 노드에서 방금 수정한 소스는 NFS 속성 캐시 때문에
  stale `.o` 가 나올 수 있다 — 그 `.o` 를 지우고 다시 빌드한다.

---

## 4. 실행

```bash
# 공통 인터페이스: 실행파일 하나에 입력 덱 하나
./build/bin/<앱> apps/<앱>/input_<케이스>.ini

# CPU 잡 (계산 노드로; 로그인 노드에서 돌리지 않는다)
sbatch apps/channel/sbatch_channel_les64_wale.sh          # 안: srun -n 4 ./build/bin/channel input_….ini
# GPU 잡 (hpcx mpirun; 2 노드면 --map-by ppr:2:node)
sbatch apps/dhvc/sbatch_dhvc_1e8_4gpu_mirror.sh
```

입력 덱은 INI 섹션으로 되어 있고, 생략한 키는 `settings.hpp` 의 멤버 초기값이 기본값이다. 모르는 키는 에러.
런 로그 맨 앞에 설정 전체가 같은 형식으로 에코되므로(재파싱 가능), **그 런이 실제로 쓴 값은 로그를 믿는다**.

| 섹션 | 키(예) |
|---|---|
| `[mesh]` `[domain]` `[stretch]` | `nx ny nz`, `lx ly lz`, `uniform_z gamma_z` |
| `[process]` | `npx npy npz` (곱 = MPI 랭크 수) |
| `[physics]` | 앱별 — `reynolds_bulk`(채널), `rayleigh prandtl delta_temperature gravity_axis`(열대류) |
| `[boundary]` | `x y z = periodic|wall|…`, `velocity_ghost = value|mirror`, `<u|v|w|p|t>_<x|y|z>_<minus|plus>` override, `inflow_profile` |
| `[les]` | `model = none|constant_vreman|wale`, `c_s`, `c_w` (채널) |
| `[time]` | `end_time` / `max_step`, `dt_max`, `max_cfl` |
| `[initial]` | 섭동, `restart_from`(같은 격자), `warmstart_from`(다른 격자 보간) |
| `[output]` | `stats_dir`, `monitor_interval`, `stats_start_time|step`, `stats_interval`, `checkpoint_interval/dir`, (dhvc) `stats_sample_interval conditional_interval spectra_interval` |

출력: 모니터 한 줄(스텝·t·dt·max\|div\|·앱별 진단·sec/step)이 로그로, 프로파일 통계가 `*_stats_<step>.dat`(헤더 `k=v` + 열 이름 줄 + 표)로,
체크포인트가 `*_ck_<step>.bin` 으로 나온다. 노트북은 헤더의 열 이름을 읽으므로 열이 늘어도 구형 파일을 계속 읽는다.

클러스터 운용 메모: 무거운 계산은 전부 `sbatch`(필요하면 `--wait`), `--exclusive`/`--mem=0` 금지(노드 drain),
`srun … | head` 금지(SIGKILL), 로그·스크립트는 NFS 경로에. **GPU 노드는 DHVC 검증 전용, 채널 LES 는 CPU 노드만**
(sbatch 에 `--exclude=gpu01,gpu02`).

---

## 5. 앱과 검증 상태 (2026-10-08)

| 앱 | 문제 | 검증 기준 | 상태 |
|---|---|---|---|
| `channel` | 주기 채널, CFR(일정 유량). DNS + LES(`[les]`) | Kim–Moin–Moser(1987) / MKM(1999) Re_τ 180·395·550; Vreman(2004) §V.B 조건 | DNS 256³ Re_τ 180 +2% PASS. LES 64³ mirror 벽: Vreman C_s 0.12 Re_τ +0.5%, U⁺ 로그영역 RMS 0.09; Re_τ 360(논문 조건) Vreman 0.12 mirror +3.6%(Dean 360 기준), ν_t∝y¹. **WALE 64³·96³ 격자 민감도 런 대기 중**(CPU 큐). |
| `rbc` | Rayleigh–Bénard, 상수물성(OB) | Kerr(1996) Nu, Ra 1e6 | Nu 격자 외삽 8.33 = Kerr PASS, GPU = CPU 일치 |
| `rbc_nob` | RBC 가변물성(글리세롤, NOB) | Horn, Shishkina & Wagner(2013) fig.9 중심온도 θ_c, ΔT 20·40·60 | θ_c 갭 −46% → 약 −13%(부력 면보간 버그 수정 후); 투영 전 물성 재갱신 적용. ΔT 40 참조 재현 +0.05%. |
| `dhvc` | 차등가열 연직공동(VC), OB, x 연직 주기·z 벽 | Ng et al.(2015, 2017) Re_δ* = U δ*/ν; 참조 2종 = `PaScaL_TCS.csv`(fig.10 점 보간) + Ng 2017 상관식 0.23·Ra^0.28 | Ra 1e5 −11%(이상치 미해결), 1e6 −4%, 1e7 +0.6%, **1e8(4 GPU, mirror 벽, 100 time 창) 41.89 = PaScaL −2.9% / Ng DNS 점 −1.7%**(20 time 구간값은 ±10% 진동). 통계는 논문 전 항목(조건평균·\|Ri_f\| PDF·스펙트럼)으로 확장됨. |
| `cavity` | de Vahl Davis 2D 정방 공동(x 벽 Dirichlet/상하 단열), BC 프레임워크의 첫 소비자 | de Vahl Davis(1983) Table V 10 항목, Ra 1e3~1e6 | 전 항목 ≤0.7%(1e6 Nu_max/min 만 −1.4/−1.0%). lid-driven·축 교환·혼합면 덱도 동작. |
| `boundary_layer` | ZPG 평판 경계층, 유입·대류 유출·자유류 상단 BC | A: Blasius 상사해(C_f, δ*, θ, H). B: KTH(Schlatter & Örlü 2010)·Torroja DNS, Re_θ ≤ 1100, Lund 재순환 유입 | A ≤0.07% 완료. B 코드·덱(re4500/re6000) 완료, **GPU 런 미제출**(DHVC 끝나면). |
| `test_pressure_wall` | x 벽 DCT 압력 풀이 단위 테스트 | 이산 참조 | `make test` |

각 앱의 노트북(`apps/<앱>/plot/*.ipynb`)이 로그·통계 파일을 읽어 참조와 비교한다. 채널은 `plot_les.ipynb` 의 `CAMPAIGN` 스위치,
DHVC 는 `plot_dhvc.ipynb` 의 `CASES` 리스트로 그릴 케이스를 지정한다.

---

## 6. 문서 지도

| 문서 | 내용 |
|---|---|
| `docs/Plan/Goal.md` | 최종 목표와 앱별 검증 기준(참조 논문) |
| `docs/Plan/plan_20261003.md` | 재작성 계획 정본: 설계 규칙(D1~D12), 디렉터리 역할, 마일스톤 M0~M10, 이후 진행 로그(§8-1 ~) |
| `docs/conventions.md` | 배열 레이아웃·ghost·루프 순서 관례(정본) |
| `docs/Design/20261005_boundary_condition_framework.md` | `[boundary]` 프레임워크 4단계 설계와 cavity 검증 기록 |
| `docs/Design/20261006_boundary_layer_validation.md` | 경계층 검증 설계(A 층류 → B 난류, 상단 BC 결정 경위) |
| `docs/Design/20261006_les_design.md` | LES 모델 선정·결합·채널 검증 결과(§8~§11: Vreman 64³/360, mirror 벽 발견, WALE) |
| `docs/Design/20261007_dhvc_statistics.md` | DHVC 통계 ↔ Ng 2017 논문 항목 1:1 표, 파일 형식, 표본 간격 |
| `docs/Debug/*.md` | 원인 추적 기록(rbc_nob div 스파이크, DHVC 1e8 mirror 재시작·통계 창 분석) |
| `docs/Paper/` | 참조 논문 PDF(Cavity, Constant_vreman, DHVC) |
| `apps/cavity/README.md` | cavity 앱 사용법과 벤치마크 표 |

새 작업을 시작할 때는 `plan_20261003.md` §8(다음 행동)부터 읽는 것이 빠르다. 중간 검증 결과는 그때그때 `docs/` 에 남기고 커밋한다.

---

## 7. 코드 작성 규칙 요약 (상세는 plan §3)

- 식은 `*_stencil.hpp` 한 곳에만. CPU `.cpp` 와 GPU `.cu` 는 그 헤더를 호출하는 쌍둥이다. `cpu/`·`gpu/` 중복 구현 금지.
- 설정은 타입이다: 앱이 `settings.hpp` 를 소유하고, 입력 키 오타·타입·범위 위반은 전부 throw. 로그 맨 앞 에코는 재파싱 가능해야 한다.
- 식별자는 전체 단어(축약 금지), 시스템을 풀면 `solve_*`, 아니면 `calculate_*`.
- 평면 평균의 나눗셈은 **전역** nx·ny 로(랭크 로컬 크기로 나누면 16배 틀린 전례). 루트 분기 안에서 집합 통신을 부르지 말 것(np ≥ 2 교착).
- 벽 ghost 정책과 음해 벽 행 닫기는 반드시 짝으로(value↔drop, mirror↔B−=A, Neumann↔B+=A). 어기면 dt·κ/dx² ≳ 3 에서 폭주한다.
- 검증 게이트: 리팩터는 스모크 통계의 **수치 비트 일치**(헤더 제외)로 확인하고, 물리 검증은 참조 논문 수치와 비교해 `docs/` 에 표로 남긴다.
- 사용자가 편집 중인 파일(노트북 등)을 `git checkout` 으로 되돌리지 않는다.

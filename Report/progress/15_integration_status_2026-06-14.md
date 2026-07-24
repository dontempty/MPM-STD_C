# MPM-STD_C 통합 진척 현황 보고서 (2026-06-14)

> 작성: 코드베이스 전수 분석 기반 (src/ 라이브러리 + apps/ 드라이버 + 검증 로그 + 실행 작업)
> 목적: **(1) 어디까지 만들어졌나, (2) 무엇이 검증됐나, (3) 어디를 고쳐야 하나, (4) 무엇이 아직 미구현인가**를 코드·로그 근거와 함께 정리.
> 통합 목표: `MPM-STD`(Fortran, channel) + `PaScaL_TCS`(Fortran, RBC)를 **단일 C++/CUDA 듀얼빌드 솔버**로 통합. 검증 흐름: CPU 검증 → GPU 변환 → 대규모 검증.
> ⚠ 이 보고서는 분석 산출물입니다. 소스 코드는 수정하지 않았습니다.

---

## 0. 한눈에 보기 (Executive Summary)

| 영역 | CPU | GPU | 검증 상태 |
|---|---|---|---|
| **Channel** (Re_τ=180 난류) | ✅ 완성 | (드라이버 없음) | ✅ **PASS** (Re_τ=179.8, u_rms⁺ peak 2.84, log-law) |
| **DHVC / Fig 7** (OB, 수직대류) | ✅ 완성 | ✅ 완성 (full-GPU pressure) | ✅ **PASS** (1e5 +1.6%, 1e6 CPU 수렴 −0.4%); 1e7 GPU **실행중** |
| **RBC / Fig 9** (NOB 글리세롤) | ✅ 완성 | ✅ 완성 (full-GPU, 통계 D2H 제거, restart) | ⚠ **물리 불일치 조사중** — Θ_c가 DNS 대비 **~1.6× 낮음**(수렴해도 유지; 코드는 전수 검증 통과 → 형상/해상도 의심) |
| **Cavity** | 스켈레톤만 | 스켈레톤만 | ⛔ SKIP 확정 |

**핵심 수치 라이브러리(core / momentum / pressure / scalar / buoyancy / properties / forcing / TDMA / grid / cfl / restart / statistics)는 CPU·GPU 양쪽 모두 실제 구현 완료**되어 있습니다. NOB 가변물성 경로(var-μ momentum + cross-stress, var-κ energy, var-density projection, glycerol properties)까지 양 백엔드에 포팅됨.

**남은 작업의 본질은 "구현"이 아니라 "검증/수렴"입니다.** 유일하게 물리 검증이 안 끝난 것은 **RBC Fig 9**입니다. 단, 이것은 두 갈래로 봐야 합니다: (a) Θ_c **누적평균**이 아직 순간장을 못 따라잡은 averaging 부족(부분 원인), 그리고 (b) **수렴된 순간 Θ_c 자체가 DNS 대비 ~1.6× 낮은 잔여 물리 불일치**(20K 케이스는 step ~275K부터 순간 Θ_c≈0.047로 사실상 수렴했는데도 1.58× 낮음). **코드 구성요소는 이전 세션에서 전수 검증을 통과**(부력/온도 BC/Nu/물성/NOB 압력투영/대류 스킴 모두 PaScaL_TCS와 일치)했으므로 버그가 아니라 **형상(Γ=6 정사각박스 vs 레퍼런스)·종횡비·열경계층 해상도** 쪽이 유력합니다.

미구현(스텁)은 모두 **비핵심 경로**입니다: IBM(P10), LES(P9), 라이브러리 GPU 통계(앱에는 구현됨), post/diagnostics(statistics로 대체됨), FilteredTDMA(미사용), src/solve(orphan spike).

---

## 1. 빌드 시스템 / 백엔드 구조

- **폴더 기반 듀얼빌드.** 모든 수치 레이어가 `cpu/`(`.cpp`)와 `gpu/`(`.cu`) 하위폴더를 가짐. `make cpu`는 CUDA-free, `make BACKEND=cuda`는 `.cu`까지 컴파일. 호출부 선택은 함수명 접미사 `_cpu`/`_gpu`로 결정.
- 많은 `*_cuda_*.cpp`가 `#ifdef MPMSTD_BACKEND_CUDA`로 실제 CUDA 본문 / `#else`에서 throw 가드를 가짐 — 이 throw들은 **정상적인 듀얼빌드 가드**이며 미구현이 아님.
- GPU 빌드는 **모든 TU를 `nvc++ -cuda`로** 컴파일(공유 `Makefile.inc`는 CPU baseline 보존을 위해 미변경).
- 환경 제약(중요): 컴퓨트 노드(cpu/gpu)는 nvhpc(HPC-X)만 있고 시스템 OpenMPI는 login01에만 존재. 시스템-OpenMPI CPU 바이너리는 login01(48코어) 전용. 컴퓨트 노드 CPU 실행은 nvhpc-CPU 빌드 필요.
- Makefile 앱 타깃: `channel cavity dhvc rbc validate rbc_cpu_ob rbc_cpu_nob` + (cuda) `rbc_gpu_ob rbc_gpu_nob dhvc_gpu`.

---

## 2. 라이브러리(src/) 구현 현황

### 2.1 완성된 핵심 경로 (CPU + GPU 양쪽 실구현)

| 레이어 | CPU | GPU/CUDA | 비고 |
|---|---|---|---|
| **core** (field/halo/boundary_ops) | ✅ | ✅ | GPU halo는 **의도적 host-staging**(D2H→CPU sendrecv→H2D) — A100+OpenMPI device-ptr 경로의 ~200ms/call 회피책 (스텁 아님) |
| **equation/momentum** | ✅ OB(const-visc) + NOB(var-μ) | ✅ 실 device ADI+TDMA+coupling | var-μ + cross/transpose stress + dd* forward-sub 결합 모두 포팅 (`momentum_gpu.cu` ~15 `__global__`) |
| **equation/pressure** (pencil-FFT) | ✅ OB + NOB(var-density) | ✅ all-GPU fast path(np1=np2=1, cuFFT) + host-staged fallback | fallback(np1>1 또는 np2>1)은 **z-TDMA를 CPU로** 수행 (§4 참고) |
| **equation/scalar** (T 수송) | ✅ OB + NOB(var-κ) | ✅ var-κ device 경로 | `scalar_gpu.cu:530` halo-sync TODO 1건(single-GPU 무해) |
| **physics/buoyancy** | ✅ OB+NOB | ✅ `k_add_buoyancy(_nob)` | NOB 비선형 밀도항 `a12pera11·θ²·ΔT` |
| **physics/properties** (glycerol) | ✅ | ✅ `k_update_properties` | μ̂,1/ρ̂,κ̂,dκ̂,1/(ρ̂Ĉp),d[…] 다항식 |
| **physics/forcing** (channel) | ✅ | ✅ (mass-flow) | `apply_body_force_gpu`는 no-op이나 **호출되지 않음**(mass-flow correction이 대체) |
| **post/statistics** | ✅ (Welford, 전역 nx·ny 정규화) | ⚠ 라이브러리는 no-op 스텁 | **실제 GPU 통계는 앱(`main_gpu_nob.cu`)에 device reduction으로 구현** — 라이브러리로 인수분해는 안 됨 |
| **post/io** (restart) | ✅ MPI-IO 글로벌 C-order | n/a(host) | 동결 난류필드 직접 로드 포맷 호환 |
| **parallel** (mpi/cuda/backend) | ✅ | ✅ | `cuda_aware_mpi.cpp`는 미사용 placeholder(실 경로는 halo_gpu.cu) |
| **linear_solver/tdma** (PaScaL) | ✅ | ✅ device 직접 + H↔D 스테이징 | |
| **grid / driver(cfl, restart)** | ✅ | ✅ (cfl) | tanh/uniform stretch |

### 2.2 미구현 / 스텁 / 죽은 코드 (모두 비핵심)

| 항목 | 상태 | 근거 |
|---|---|---|
| **physics/ibm** | ⛔ no-op 스텁 (CPU+GPU) | `ibm_{cpu.cpp,gpu.cu}` `/* TODO(P10) */`; 어떤 앱도 호출 안 함 |
| **physics/les** | ⛔ no-op 스텁 (CPU+GPU) | `les_{cpu.cpp,gpu.cu}` `/* TODO(P9) */`; Smagorinsky/WALE 미작성 |
| **post/gpu/statistics_gpu.cu** | ⛔ no-op `/* TODO(P5) */` | 실 GPU 통계는 앱에만 존재(라이브러리 미인수분해) |
| **post/diagnostics (cpu+gpu)** | ⛔ no-op, **죽은 코드** | Re_δ*/center-temp는 `statistics_cpu.cpp`가 대체 구현 → diagnostics 스텁은 superseded |
| **FilteredTDMACpuBackend** | ⛔ 선언만, 본문/`.cpp` 없음, **미사용** | `filtered_tdma_cpu_backend.hpp`; 선택 시 링크 에러 (현재 inert) |
| **src/solve/** (poisson, banded) | ⛔ orphan spike 잔재 | "P0 skeleton stub", "Spike body" 자기주석; 실제 Poisson은 equation/pressure |
| **boundary/boundary_applier.cpp** | ⛔ 구버전, Wall/Inflow/Outflow throw | 활성 경로는 `core/.../boundary_ops` |
| **scalar_gpu.cu:530** | ⚠ 미세 TODO | GPU scalar solve 후 halo auto-sync 없음(호출부가 sync; single-GPU 무해) |

---

## 3. 앱(apps/) 현황 — 시간전진 레시피

모든 앱 공통 골격: `Config::load(toml)` → topology/subdomain/grid/BC/TDMA → Fields 등록 → 물리상수 → IC → 시간루프. 케이스는 **per-case `.toml`** + sbatch/shell 스크립트로 구성.

### 3.1 channel — `apps/channel/main.cpp` (CPU 전용, live)
등온 난류 채널(§7 레시피): CFL → const-visc momentum(3-sweep ADI + block coupling) → forcing(dPdx body force + mass-flow) → pressure(div+Poisson+project) → statistics. **GPU 드라이버 없음**(§5 참고).

### 3.2 dhvc — `main.cpp`(CPU) + `main_gpu.cu`(GPU), OB만
축: x=수직/중력(주기), y=span(주기), z=벽(가열). OB free-fall ν=√(Pr/Ra), α=1/√(RaPr), 부력 F=θ on U.
레시피: CFL → energy(const-diff) → momentum + **OB buoyancy(main 합성)** → **zero-net-flux 보정**(`apply_mass_flow_correction(target=0)`) → pressure → stats(Re_δ*). GPU는 동일 순서 전부 device, **full-GPU pressure**(`solve_pressure_gpu`).

### 3.3 rbc — Rayleigh-Bénard (Fig 9), 파일 7개 중 정리 필요
축: z=벽/중력, hot bottom +0.5 / cold top −0.5, x·y 주기. 부력 F on W. 중력이 벽방향이라 **zero-net-flux 보정 없음**(dhvc와 차이).

| 파일 | Makefile | 빌드 | 상태 |
|---|---|---|---|
| `main.cpp` | `rbc` | cpu+cuda | 원본 모놀리스(`#ifdef`+런타임 nob), 여전히 빌드됨 |
| `main_cpu_ob.cpp` / `main_cpu_nob.cpp` | `rbc_cpu_*` | cpu+cuda | **활성**(분리된 CPU 드라이버, untracked) |
| `main_gpu_ob.cu` | `rbc_gpu_ob` | cuda | 활성, hybrid CPU pressure |
| **`main_gpu_nob.cu`** | `rbc_gpu_nob` | cuda | **LIVE** — full-GPU pressure, GPU 통계 reduction, restart, steady-state timing |
| `main_gpu_nob.cpp` / `main_gpu_ob.cpp` | — | 없음 | ⛔ **죽은 코드**(Makefile·아티팩트 없음, 구버전 draft) |

NOB-NOB 레시피(`main_cpu_nob.cpp`): CFL → `update_properties`(glycerol) → energy(var-κ, semi-implicit conv) → momentum(var-μ + cross-stress) + **NOB buoyancy** → solve+update → zero_w_wall → **pressure(var-density, inv_rho)** → **T clipping** [−0.5,0.5] → stats(Θ_c, Nu).

### 3.4 cavity — `main.cpp` 스켈레톤만 (SKIP 확정)
MPI init 후 "skeleton" 출력하고 return. 시간루프·솔버 호출 없음. 압력엔진이 X·Y 주기를 하드 요구 → 전벽 cavity는 전-Neumann DCT 솔버 신설 필요 → **DHVC/RBC는 z벽+2주기라 기존 FFT 엔진 재사용 가능하므로 cavity는 의도적 SKIP**.

### 3.5 validate — `main.cpp` (CPU 직렬, PaScaL_TCS 대조 하니스)
DHVC 형상, 고정 dt 200스텝, `physics.faithful` 토글(1=var-μ, 0=const-visc)로 momentum 스킴 비교. 기준 필드를 reference 축 레이아웃으로 덤프 → `compare_pascal_tcs.py`.

---

## 4. 검증 현황 (실측 수치 vs 기준)

### 4.1 Channel Re_τ=180 — ✅ PASS
출처: `apps/channel/turb_out/stats_final.dat` (step=16000, n=1601), 32×24×96 stretch, cold-start.

| 항목 | 측정 | 기준(KMM/DNS) | 오차 |
|---|---|---|---|
| Re_τ | **179.8** | ≈180 | −0.1% |
| u_rms⁺ peak | **2.84** @ Z⁺=14.2 | ≈2.65 @ Z⁺≈15 | +7% (위치 정확) |
| centerline U⁺ | **18.57** | ≈18.3 (log-law) | +1.5% |
| divergence | ~5e-14 (sustain) | — | 기계영점 |

v/w_rms⁺는 작은 박스(32×24)라 다소 낮음. 헤드라인 타깃은 모두 통과.

### 4.2 DHVC Fig 7 — ✅ PASS (1e5, 1e6); 1e7 실행중
기준: Ng `0.23·Ra^0.28` (1e5→5.78, 1e6→11.01, 1e7→20.98), PaScaL_TCS DNS (1e5→5.378, 1e6→10.430, 1e7→22.95).

| Ra | 격자 | 측정 Re_δ* | Ng 대비 | 수렴 | HW |
|---|---|---|---|---|---|
| 1e5 | 384×96×192 | 5.868 | **+1.6%** | 시간제한 컷(9.5K/12K) | GPU |
| **1e6 (CPU)** | 384×96×192 | **10.968** | **−0.4%** | ✅ **완료** | CPU |
| 1e6 (GPU) | 384×96×192 | 10.730 | −2.5% | 시간제한 컷 | GPU |
| 1e7 | 840×192×416 | (10스텝) | — | ⏳ **실행중(job 29421)** | GPU |

가장 깨끗한 검증은 **CPU 1e6 수렴값 −0.4%**(`log/fig7_1e6_29254.out` `done: Re_dstar=10.9684 (Ng=11.0085)`). div ~1e-13. **1e7은 현재 gpu01에서 실행중이며 아직 결과 없음**(840×192×416 ≈ 67M cells, 워밍업 8079 ms/step).

### 4.3 RBC Fig 9 NOB — ⚠ 물리 불일치 ~1.6× (조사중, 버그 아님)
글리세롤, Ra=1e6, Pr=2547, Γ=6, 128³, GPU. 20/40/60은 **ΔT[K]**. Fig-9 양: **Θ_c**(중앙면 온도편차), 플롯 y=ΔT·Θ_c [K].
DNS(`fig9_data/PaScaL_TCS.csv`): 무차원 Θ_c 20K→0.0744, 40K→0.1350, 60K→0.1900 (= 차원 1.49K, 5.40K, 11.40K).

| ΔT | step(최종) | Θ_c 누적평균 | Θ_c 순간(수렴값) | Nu | DNS Θ_c | 순간/DNS | 수렴 |
|---|---|---|---|---|---|---|---|
| 20K | 500,000 | 0.0419 | ≈0.047 | 10.88 | 0.0744 | **1.58× 낮음** | ✅ 순간 수렴(~275K부터) / 누적평균 상승중 |
| 40K | 350,000 | 0.0662 | ≈0.085 | 10.77 | 0.1350 | **1.59× 낮음** | ◐ ~0.87 τ_T |
| 60K | 400,000 | 0.0907 | (미정) | 9.30 | 0.1900 | — | ❌ 평균 부족 |

**진단 (메모리 기록 기반 정밀 조사):** 두 효과가 겹침.
1. **Averaging 부족(부분 원인):** Θ_c **누적평균**(monitor.csv 컬럼)이 단조 증가중이라 순간 수렴값보다 낮게 보임 (20K: 0.042 vs 순간 0.047). τ_T=1/(Ct·Nu)≈4695 ff. 누적평균은 더 길게 모아야 순간값에 수렴.
2. **잔여 물리 불일치(핵심 미해결):** 그런데 **20K는 step ~275K부터 순간 Θ_c≈0.047로 사실상 수렴**했고, 그래도 DNS 0.0744 대비 **1.58× 낮음**. 40K도 동일하게 1.59×. 즉 averaging을 끝까지 해도 ~1.6× 갭이 남음.

**코드는 버그가 아님 (이전 세션 전수 검증 통과):** 부력 `ir·(T+a12pera11·T²·ΔT)` Cmt=1, 온도 antisym ghost(Dirichlet ±0.5), Θ_c 누적(z=H/2 수평평균), Nu=1+⟨WT⟩/Ct, 글리세롤 물성(40K μ비 15.6×), NOB 압력투영, 대류 −0.5(U·∇T) — 모두 PaScaL_TCS와 일치 확인. **유력 원인은 형상/해상도:** ① Γ=6 정사각박스 vs 레퍼런스(PaScaL_TCS DHVC 4×1×2, Horm 실험 원통 Γ~1-2) → 대류셀 조직 차이, ② Ra=1e6,Pr=2547에서 선호파장 λ_c≈2.8H라 Γ=6에 다중셀, ③ 열경계층 해상도(최소셀 ≈2.5 η_B로 hot wall 열BL under-resolve 가능). 갭이 20K/40K에서 1.58 vs 1.59로 **일정**한 점은 BL해상도(2× vs 4× 다름)보다 **형상 효과 지배**를 시사.

출처: `apps/rbc/fig9_nob_gpu_out_{20,40,60}_x/monitor.csv`, `apps/rbc/logs/g25{a,b,c}_*.log`, 메모리 `project_gpu_fig9_status`.

> 참고: 옛 `fig9_data/fig9_thetac_vs_delta.csv`는 다른 CPU-proxy(Pr=7, Ra=1e5, 64³, Γ=2) 캠페인이라 GPU Fig-9와 직접 비교 불가.

### 4.4 성능 (2×A100)
- RBC 128³ steady-state: **~29 ms/step** (g25a 29.3 / g25c 29.4 / g25b 29.0). 워밍업 ~110ms/step.
- 메모리의 "28× vs CPU" 가속과 정합(이전 refactor 벤치 기준).
- DHVC 1e7(67M cells)은 워밍업 8079 ms/step(steady-state 아님).

---

## 5. 수정·정리가 필요한 부분 (우선순위)

1. **[검증 1순위] RBC Fig 9 ~1.6× 갭 규명.** 두 단계: (a) 우선 monitor에 **순간 Θ_c와 누적평균 Θ_c를 분리 출력** + 40/60K를 충분히 길게(τ_T 여러 배) 평균하여 averaging 효과를 제거 → 순간 수렴값 확정. (b) 그래도 남는 ~1.6× 갭(20K에서 이미 확인)은 **형상/해상도 가설 검증**으로: Γ를 레퍼런스에 맞춰(예: DHVC 4×1×2 또는 더 작은 Γ) 재현해 Θ_c가 DNS로 수렴하는지, 그리고 z 해상도(192³ 등)를 높여 열BL을 더 풀었을 때 변화를 확인. 코드 자체는 전수 검증 통과 상태이므로 수정 대상이 아니라 **케이스 셋업/해상도** 점검 대상.
2. **[GPU 스케일링] 압력 fallback의 CPU z-TDMA.** all-GPU fast path는 `np1==1 && np2==1`만. np1>1 또는 np2>1이면 z-TDMA가 CPU로 떨어져 host bounce 발생 → **P8 대규모 multi-GPU 압력 확장의 병목**. 단일노드 2-GPU(1×1×2)는 fast path 활용 가능하나, 다수 GPU 분산 시 device z-TDMA 경로 마무리 필요.
3. **[앱 정리] rbc 드라이버 7개 → 정돈.** `main_gpu_nob.cpp` / `main_gpu_ob.cpp`(죽은 draft) 제거, untracked 활성 드라이버(`main_cpu_*.cpp`, `main_gpu_*.cu`)와 입력 TOML을 git에 커밋. 현재 `git status`에 옛 `input_fig9_*.toml` 다수가 `D`(삭제)로 미커밋 상태.
4. **[라이브러리 위생] GPU 통계 인수분해.** 실 GPU 통계가 앱(`main_gpu_nob.cu`)에만 있고 `post/gpu/statistics_gpu.cu`는 no-op → 라이브러리로 끌어올리면 dhvc/channel GPU에서 재사용 가능.
5. **[미세] `scalar_gpu.cu:530` halo auto-sync TODO.** single-GPU는 무해하나 multi-GPU scalar에서 명시적 sync 누락 주의.
6. **[죽은 코드] 정리 대상**: `src/solve/`(orphan), `boundary/boundary_applier.cpp`(구버전), `post/diagnostics`(superseded), `FilteredTDMACpuBackend`(선언만). 빌드엔 무해하나 혼선 유발.
7. **[충실도 참고] momentum const-visc(channel) 경로의 §3.2 단순화.** Report 14(2026-06-08)가 지적한 델타형 잔차항 `−M·Uⁿ` 누락 등은 **const-visc 채널 경로**에 남아있을 수 있음(채널은 그래도 PASS). 단, 그 이후 추가된 **var-μ 경로는 PaScaL_TCS 충실 포팅**(full stress + cross + `−M·Uⁿ` 포함, NOB_RBC_SPEC §3)이라 Report 14의 "미이식" 다수 항목은 var 경로에서 해소됨. 두 경로의 일관성 점검 권장.

---

## 6. 아직 구현이 덜 된 부분 (미구현)

- **IBM (P10)**: CPU·GPU 모두 no-op. H-masking 미작성.
- **LES (P9)**: CPU·GPU 모두 no-op. Smagorinsky/WALE 미작성.
- **Channel GPU 드라이버**: 없음. 라이브러리 GPU 경로는 완비돼 있어 추가는 직선적이나, 현재 channel은 CPU 전용. (등온 채널 GPU가 필요한지 사용자 결정 사항.)
- **DHVC/validate NOB 드라이버**: 없음(OB만). NOB 물리는 라이브러리에 있으므로 필요시 dhvc NOB도 합성 가능.
- **Cavity**: 의도적 SKIP(전-Neumann DCT 압력솔버 미신설).
- **라이브러리 GPU 통계/diagnostics**: §5-4 참고.

### 마일스톤 매핑 (docs/PROGRESS.md 기준 갱신)
| 단계 | 내용 | 실제 상태 |
|---|---|---|
| P3/P3b | solve 일반화 / cavity | cavity SKIP 확정 |
| P4/P5 | GPU core+멀티GPU 통신 / GPU 커널 | ✅ 완료 (2×A100 실행) |
| **P6** | DHVC Fig 7 | ✅ **검증 완료**(1e5/1e6); 1e7 실행중 |
| **P7** | NOB RBC Fig 9 | ✅ **구현 완료**(코드 전수 검증 통과) / ⚠ Θ_c가 DNS 대비 ~1.6× 낮음 — 형상·해상도 조사중 |
| **P8** | multi-GPU 최종 검증 | ⏳ 단일노드 2-GPU 동작; 압력 multi-GPU z-TDMA 확장 미완(§5-2) |
| P9 | LES | ⛔ 미착수 |
| P10 | IBM | ⛔ 미착수 |

> 주의: `docs/PROGRESS.md`는 P6/P7 초입에서 멈춰 있어 git log(P7 NOB 전체 + GPU 통계/restart까지)보다 뒤처져 있습니다. PROGRESS.md 갱신 권장.

---

## 7. 결론 및 다음 단계 권고

**통합은 본질적으로 완료 단계입니다.** Channel(MPM-STD)과 RBC(PaScaL_TCS)의 수치 방법론이 단일 C++/CUDA 듀얼빌드로 합쳐졌고, NOB 가변물성 전 경로가 CPU·GPU 양쪽에 포팅됐습니다. Channel과 DHVC는 정량 검증 통과, GPU는 2×A100에서 ~29 ms/step로 동작합니다.

**남은 일은 거의 전부 "검증/수렴"이며 "구현"이 아닙니다:**
1. **RBC Fig 9 ~1.6× 갭 규명** → 가장 중요. averaging을 끝내도 순간 Θ_c가 DNS 대비 1.58× 낮음(20K 확인). 코드는 전수 검증 통과 → **형상(Γ)·열BL 해상도** 가설을 케이스 재셋업으로 검증. (averaging만으로는 해결 안 됨.)
2. **DHVC 1e7** GPU 런(job 29421) 완주 → Re_δ*≈21 확인.
3. (선택) **P8 multi-GPU 압력** device z-TDMA 마무리로 대규모 확장.
4. **위생**: rbc 드라이버/입력 TOML 정리·커밋, PROGRESS.md 갱신, 죽은 코드 제거.

미구현(IBM/LES)은 P9/P10 후속이며 현재 검증 목표(Fig 7/9)와 무관합니다.

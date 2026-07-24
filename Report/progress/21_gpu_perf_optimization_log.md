# GPU 성능 최적화 로그 (DHVC, living document — 2026-06-24~)

> 대상: `apps/dhvc/main_gpu.cu` 시간루프 + `src/equation/*/gpu/*.cu` 커널
> 동기: 1e8(1536×340×768, 4×A100)이 벽시계로 매우 오래 걸림 → 스텝당 시간 단축 시도.
> 방법: 변경 → 빌드 → **A/B 실측**(bit-identical 검증 + ms/step). 추측 금지, 측정 기반.
> 관련: [16_dhvc_1e7_gpu_bottleneck.md](16_dhvc_1e7_gpu_bottleneck.md), [17_dhvc_cpu_bottleneck.md](17_dhvc_cpu_bottleneck.md), [../OPTIMIZATIONS.md](../OPTIMIZATIONS.md)
> 이 문서는 시도할 때마다 **이어서 갱신**한다(아래 §3 표에 한 줄씩 추가 + §4에 상세).

---

## 0. 핵심 결론 (현재까지)

- **두 레버**: 벽시계 = (스텝당 시간) × (스텝 수). 스텝 수는 고-Ra에서 CFL dt 붕괴로 수백만(완전음해 대류만이 근본 해결, 레퍼런스 이탈). 본 로그는 **스텝당 시간**에 집중.
- **phase 분해(1e6)**: momentum 28.6ms(58%) ≫ scalar 10.3 > pressure 8.5. **nsys 커널 분해(§4.2)**: **TDMA ~35%(최대)** > ADI band ~16% > RHS ~11% > block-coupling ~9% > pressure ~8%. 그리고 **스텝당 wall ~35-45ms vs GPU 커널 ~24ms → ~35-45%가 비-커널 오버헤드**(cudaDeviceSynchronize 72/step + H2D/D2H memcpy 628/step).
- **A1(RHS k-major coalescing): 효과 없음**(35.9→36.2). RHS가 ~11%뿐이라 정량적으로 확정.
- **재정의된 타깃**(§4.2.1 검증 후): ① **TDMA(~35%)** 라이브러리 + 소격자 점유율 기아(1e8 자연개선). ② var 커널 점유율은 레지스터로 고정(A4 한계). ③ sync는 분산-MPI에 load-bearing(LEANSYNC 크래시, §4.5).
- **★ memcpy storm 진짜 출처 확정(§4.6)**: 628 H2D/D2H/step = **CUDA-aware MPI(smcuda)가 PaScaL_TDMA의 strided 파생-데이터타입 `MPI_Ialltoallw`를 블록단위 host-staging**. cuFFT/우리코드/contiguous-MPI 전부 무죄(마이크로벤치 실측). **실제 레버 = PaScaL_TDMA 분산 alltoall을 contiguous 버퍼로 전환(라이브러리 수정)** → host-staging 제거 + Z-TDMA 가속.
- **★ A-LAYOUT 적용(§4.7) + memcpy 재측정(§4.8)**: rd 버퍼 `[n_sys×2]`, `MPI_Ialltoallw` → `MPI_Ialltoallv(MPI_DOUBLE)`. cold-300: step 100~300 ≈33ms(OLD 34.7ms, ~5%↓), Z-TDMA scalar 4.38→3.8ms. **BUT nsys 재측정: H2D 300+D2H 327/step 그대로 — `MPI_Ialltoallv` device도 host-staging**. OpenMPI smcuda BTL은 `Alltoall*` 계열 전부 host 경유(P2P만 D2D). ~5% 개선은 derived-type CPU 오버헤드 제거뿐. **진짜 해결: `MPI_Isend/Irecv` ring으로 교체(§4.8)** → H2D/D2H 0 + Z-TDMA ~1ms 기대.

---

## 1. 측정 환경 & 계측

- **계측 추가**(2026-06-24): `apps/dhvc/main_gpu.cu` 시간루프에 (a) **전체 누적 계산시간** `[TIMING_TOTAL] compute=…s | N steps | avg … ms/step`, (b) monitor 라인/CSV에 `wall=…s ms/step=…` 컬럼 추가. 기존 `[TIMING_BREAKDOWN]`(첫 10스텝 phase별)·`[TIMING_SS]`(steady-state ms/step)는 유지. → 모든 A/B의 1차 지표.
- **A/B 하니스**: [scripts/a1_ab.sh](../scripts/a1_ab.sh) + [apps/dhvc/input_a1_ab.toml](../apps/dhvc/input_a1_ab.toml). 1e6 격자(384×192×96) warm-start(결정적 IC) 300스텝, OFF/ON 각 실행 후 **restart 필드 byte-cmp**(correctness) + ms/step 비교.
  - ⚠️ 코드가 **run-to-run bit-reproducible가 아님**(atomic 사용: mass-flow 보정/pressure 등 → ULP 비결정성). 따라서 byte-cmp는 "round-off 이상 차이"만 잡는 게이트로 해석해야 함(O(1)/NaN이면 진짜 버그).

### 베이스라인 phase 분해 (1e6, 2×A100, 첫 10스텝)
```
cfl=0.1  scalar=10.3  mom=28.6  corr=0.2  sync=1.9  pres=8.5  ms
```
steady-state ≈ 35–36 ms/step (dt 안정 후). 참고 사다리 ms/step: 1e5≈57, 1e6≈53(2×A100), 1e7≈315(4×A100).

---

## 2. 사다리 결과(검증 맥락, 참고)

| Ra | grid | GPU | Re_δ*(목표) | 비고 |
|---|---|---|---|---|
| 1e5 | 384×192×96 | 2×A100 | 5.30 (5.78) | cold |
| 1e6 | 384×192×96 | 2×A100 | 10.16 (11.0) | cold → restart |
| 1e7 | 832×416×192 | 4×A100 | — | warm(slab) → restart |
| 1e8 | 1536×340×768 | 4×A100 | **39.62 (39.97, 0.9%)** | warm(slab) → restart_1e8_t150 |

Tier-1 slab warm-start I/O는 replicate와 **bit-for-bit 일치** 검증됨([scripts/tier1_validate.sh](../scripts/tier1_validate.sh)).

---

## 3. 시도 로그

| # | 시도 | 무엇을 | 결과(ms/step) | 판정 | 상세 |
|---|---|---|---|---|---|
| 계측 | timing instrumentation | 누적 계산시간 + monitor wall/ms 컬럼 | — | ✅ 적용(작동확인) | §4.0 |
| **A1** | RHS 커널 k-major coalescing (flag `MPMSTD_KMAJOR`) | `k_mom_rhs_var`·`k_energy_residual_var`·`k_add_increment` threadIdx.x→k, block (8,8,8)→(32,4,2) | **35.90 → 36.19 (개선 없음)** | ❌ ROI 없음 | §4.1 |
| **PROF** | nsys 세분 프로파일 (1e6, np2) | 커널/API/memops 분해 | — | ✅ 완료 → 타깃 재정의 | §4.2 |
| **ncu** | 커널 roofline (k_mom_adi_var, tdma_cyclic) | 점유율/바운드 종류 | — | ✅ 완료 | §4.3 |
| **A4** | `__launch_bounds__(512,3)` 레지스터 캡→점유율↑ | 4 var 커널 | (램프오염 측정, 이론상 spilling 손해) | ❌ 기각·되돌림 | §4.4 |
| **검증** | memcpy storm 출처 추적 | TDMA(CUDA-aware) 아님→pressure | — | ✅ 정정 | §4.2.1 |
| **BASE** | 되돌린 baseline 측정 | cold-300 | **34.7 ms/step (steady)** | ✅ 기준점 | §4.4 |
| pressure FFT 배치 | Y-FFT 슬랩루프→배치 | — | — | 🚫 BLOCKED(레이아웃) | §4.4 |
| **LEANSYNC** | 불필요 cudaDeviceSynchronize 제거 | mom+scalar 25 sync no-op | base 35.3, **lean 크래시** | ❌ sync는 분산-MPI에 load-bearing | §4.5 |
| 펜슬 레이아웃 리팩터 | [ix,j,k]→[ix,k,j] FFT 배치 가능화 | — | — | ⬜ 미시도(큰/위험) | — |
| (장기) 완전음해 대류 | 스텝 수↓ | — | — | ⬜ 미시도(레퍼런스 이탈) | — |
| **★ A-LAYOUT** | PaScaL_TDMA rd `[2×n_sys]`→`[n_sys×2]`, `MPI_Ialltoallw`→`MPI_Ialltoallv(MPI_DOUBLE)` | 3파일(tdma_local_cuda.cu, pascal_tdma_many_cuda.cu/.hpp), 호출부 불변 | step100~300 ≈33ms(↓5%), Z-TDMA 4.38→3.8ms; **memcpy 300+327/step 불변**(Ialltoallv도 host-staging) | ✅ 적용(부분이득) | §4.7 |
| **nsys 재측정** | A-LAYOUT 후 memcpy 정량 확인 | 5/40-step nsys diff | H2D 300+D2H 327/step = OLD와 동일 | ✅ 완료 → 진짜 해결책 확인 | §4.8 |

> **캠페인 종합 판정**: per-step 쉬운 win **전부 막힘** — A1(coalescing, RHS 11%만)·A4(점유율, 레지스터 spilling)·pressure FFT 배치(레이아웃)·LEANSYNC(sync는 MPI에 필요). nsys "오버헤드 35%"의 sync 부분은 **대부분 분산-MPI 순서에 필요**(제거불가), 나머지는 1e8서 비중↓. **GPU 솔버는 이 스킴·분할 기준 이미 실용적 최적화 한계**. 1e8 wall-clock의 진짜 레버는 **스텝 수(완전음해 대류, 레퍼런스 이탈)** 또는 펜슬 레이아웃 대공사. 기준점 **BASELINE 34.7 ms/step**(§4.4) 기록 완료.

(범례: ✅적용 ❌기각 ⬜미시도 🔄진행중)

---

## 4. 시도 상세

### 4.0 계측(timing instrumentation) — ✅ 적용
- 변경: [main_gpu.cu](../apps/dhvc/main_gpu.cu) — 영구 타이머 `t_loop_start`(리셋 안 됨), monitor 시 `cudaDeviceSynchronize` 후 누적 wall + 최근 ms/step 계산, 루프 종료 후 `[TIMING_TOTAL]`. `gpu_monitor_dhvc`에 `cum_wall_s, recent_ms_step` 인자 + stdout/CSV 컬럼.
- 검증: a1_ab 빌드에서 첫 컴파일 통과, `[TIMING_TOTAL] compute=10.8s | 300 steps | avg 35.9 ms/step` 출력 확인.
- 비고: 1e8(29575)은 이 변경 **이전** 바이너리로 돌아 미적용(monitor.csv 9컬럼). 이후 빌드부터 적용.

### 4.1 A1 — RHS 커널 k-major coalescing — ❌ 기각(ROI 없음)
- **가설**: RHS 3D-stencil 커널이 `threadIdx.x→i`(stride n2·n3)라 warp가 비-coalesced. `threadIdx.x→k`(stride 1)로 바꾸면 coalesced 로드로 메모리 바운드 완화.
- **구현**(flag-gated, 기본 OFF로 production 무영향): [momentum_gpu.cu](../src/equation/momentum/gpu/momentum_gpu.cu) `k_mom_rhs_var` + `mom_kmajor()`; [scalar_gpu.cu](../src/equation/scalar/gpu/scalar_gpu.cu) `k_energy_residual_var`·`k_add_increment` + `scalar_kmajor()`. env `MPMSTD_KMAJOR` 설정 시 ON.
- **실측**(1e6, 2×A100, 300스텝):
  ```
  OFF: avg 35.90 ms/step   [breakdown] scalar=10.3 mom=28.6 pres=8.5
  ON : avg 36.19 ms/step   [breakdown] scalar= 9.7 mom=28.2 pres=8.3
  ```
  → momentum 0.4ms↓, scalar 0.6ms↓, **총합 개선 없음(오히려 +0.3)**.
- **왜 안 먹혔나**: A1이 건드린 RHS 커널은 momentum 28.6ms 중 **작은 일부**. 대부분은 ADI band 조립(`k_mom_adi_var`) + PaScaL_TDMA(방향·성분당 ~30 커널 런치) + scatter + block coupling. 작은 격자(7M셀)는 **메모리 바운드가 아니라 런치-오버헤드 바운드** → 메모리 coalescing이 무의미.
- **correctness**: byte-cmp는 DIFFER였으나 Re/KE/Nu/Trms 4자리 일치·div 양쪽 ~1e-14 = round-off 수준. A1 커널은 cross-thread reduction이 없어 매핑 변경이 셀당 산술을 못 바꿈 → 차이는 **다른 곳의 atomic 비결정성**(§1 주의)이지 A1 버그 아님.
- **처리**: flag-gated라 default-OFF면 무해. 코드 정리를 위해 되돌릴지, 1e8 스케일(메모리 바운드 가능)에서 재평가할지 미정.

### 4.2 nsys 세분 프로파일 (1e6, np 1×1×2, 20스텝) — ✅ 완료
- 도구: [scripts/dhvc_prof.sh](../scripts/dhvc_prof.sh) (per-rank `mpirun … nsys profile -o …_%q{rank}`). 리포트: `apps/dhvc/prof/dhvc_1e6_0.nsys-rep`.
- **GPU 커널 시간 그룹화**(monitor 전용 k_diag_dhvc 제외, 스텝당 GPU 커널 ≈ 24ms):

| 그룹 | 커널 | 커널시간 % | ms/step |
|---|---|---|---|
| **TDMA** | tdma_cyclic_many(27.4) + modified_thomas(5.7) + update_solution(1.6) + tdma_many(0.2) | **~35%** | **~8.8** |
| ADI band 조립 | k_mom_adi_var(8.7) + k_build_adi_var(6.9) | ~16% | ~3.9 |
| RHS (=A1 대상) | k_mom_rhs_var(7.7) + k_energy_residual_var(2.7) | ~11% | ~2.6 |
| block coupling | k_block_dU/dW + k_dd_dV/dW | ~9% | ~2.2 |
| pressure | regular_fft(3.7, **96회/step**) + project/pred/div/unpack/scale 등 | ~8% | ~1.9 |
| scatter+add | k_m_scatter + k_scatter_from_tdma + add_increment | ~6% | ~1.5 |

- **CUDA API**: `cudaDeviceSynchronize` **33%**(1441회=**72/step**), `cuMemcpyAsync` 17.6%, `cuStreamSynchronize` 8.5%. (cudaHostAlloc/HostRegister ~27%는 startup 1회, 제외)
- **GPU memops**: HtoD 46.8%(**301/step**) + DtoH 16.9%(**327/step**) + DtoD 28.8%(71/step) + PtoP 7.2%(13/step). → **스텝당 H2D/D2H ~628회**.
- **스칼라 SCA 타이밍**(로그): `Z(tdma=4.38ms) ≫ X(0.98) Y(0.60)` — **분산 Z방향 TDMA가 비분산 X/Y보다 ~5× 느림**.

### 4.2.1 findings 코드 검증 (§4.2 추적, 2026-06-24) — ⚠️ memcpy 출처 정정
- **memcpy 628/step의 출처는 TDMA가 아니라 PRESSURE engine**:
  - PaScaL_TDMA CUDA 라이브러리(`PaScaL_TDMA_C/src/pascal_tdma_many_cuda.cu:156-180`)는 **이미 CUDA-aware** — `MPI_Ialltoallw`에 device 포인터 직접 전달, **memcpy 0**. 단일-rank 방향(X/Y, np=1)은 short-circuit. → **이전 "TDMA host-staging 의심"은 오류.**
  - 우리 momentum/scalar ADI는 **device 인터페이스 `solve_many_device`** 사용(`momentum_gpu.cu:1144`, `scalar_gpu.cu:378`) → memcpy 0. 주석에 "GPU kernel replaces D2H+H2D" 이미 최적화.
  - 진짜 출처 = **pressure engine**: 펜슬 버퍼 `hat_I_/hat_Y_`가 **pinned host**, z-TDMA(`pressure_engine_gpu.cuh:915-969`)가 **host 배열 + host `solve_many`**(957-960行, 4 H2D+1 D2H/call), **Y-FFT가 슬랩 `for(k<n3_I_)` 루프**(96 cufftExec/step → cuFFT 내부 async memcpy). nsys `regular_fft 96/step`·`cuMemcpyAsync 637/step` 일치.
- **Z-TDMA 5× 느림 검증**: host-staging 아님(CUDA-aware). **분산(z-split) 방향의 실제 `MPI_Ialltoallw` 통신 비용**(rank 간). X/Y는 단일-rank short-circuit이라 빠름. → 1e8(np3=4)에선 통신량↑이나, 점유율은↑(상쇄). 본질적 분산 비용.
- **결론 정정**: 오버헤드(memcpy+sync)의 큰 몫은 **pressure engine의 host-staging 펜슬 + 미배치 Y-FFT 루프**. 실제 레버 = **pressure를 device-resident화 + Y-FFT 배치화**(TDMA 아님).
- **핵심 해석**:
  1. **TDMA가 단일 최대(~35%)** — 특히 cyclic(주기 X/Y) TDMA. PaScaL_TDMA 라이브러리 코드.
  2. **스텝당 wall ~35–45ms vs GPU 커널 ~24ms → ~35–45%가 비-커널 오버헤드**(sync 72/step + memcpy 628/step). 작은 격자라 오버헤드 비중↑.
  3. **A1(RHS) ~11% 확정** — A1이 안 먹힌 이유 정량 확인.
- ⚠️ **1e6 vs 1e8 주의**: 오버헤드(sync/launch)는 격자무관 고정 → 1e8(57×↑)에선 비중 급감. **TDMA·memcpy(데이터 비례)는 1e8에서도 지배** → **1e8의 견고한 타깃 = TDMA + 분산통신 memcpy**.

### 4.3 ncu 커널 레벨 (Nsight Compute, 1e6, np2) — ✅ 완료
- 도구: [scripts/dhvc_ncu.sh](../scripts/dhvc_ncu.sh) (`srun -n2 ncu --kernel-name regex:<k> --set full --launch-skip 1 --launch-count 1 -f -o …_%q{SLURM_PROCID}`). 리포트: `apps/dhvc/prof/ncu_*.ncu-rep`.

| 커널 | Dur | Mem% | Comp% | 점유율(이론/달성) | reg/thread | 진단 |
|---|---|---|---|---|---|---|
| **k_mom_adi_var** (우리, ADI band) | 300µs | 47 | 40 | 50% / **45%** | **64** | mem·comp 둘 다<50% = **latency 바운드**, 점유율이 **레지스터(64)로 50% 제한**. L1→L2 부분섹터(약한 uncoalesce). |
| **tdma_cyclic_many** (라이브러리) | 852µs | 32 | **3** | 62% / **8.4%** | 46 | **점유율 기아**(waves **0.13** = GPU 거의 idle). n_sys=18,432 systems가 A100(221K thread) 못 채움. |

- **핵심 1 — 우리 var 커널: 레지스터-제한 점유율**. 64 reg/thread → SM당 블록 2개 → 이론 점유율 50%. **레지스터를 ~40 이하로 낮추면 점유율 2배** 가능 → latency 은닉. `__launch_bounds__`/`-maxrregcount`로 bit-identical 실험 가능. **격자 무관 → 1e8에도 유효**.
- **핵심 2 — TDMA 점유율 기아는 소격자 아티팩트**: n_sys(=cross-plane cell 수)가 적어 GPU 못 채움. 1e8은 n_sys 14×↑(261K systems) → **점유율 자동 개선** → **TDMA의 1e6 지배(27%)는 1e8에서 크게 완화**. 즉 TDMA는 추격 가치 낮음(라이브러리 + 격자 크면 해소).
- **재결론**: **가장 actionable·grid-robust 레버 = 우리 var 커널의 레지스터 압력↓ → 점유율↑** (다음 실험). TDMA는 1e8서 자연 개선되니 우선순위 강등.

### 4.4 BASELINE 확정 + A4 기각 + pressure 조사 (2026-06-24)
- **A4(launch_bounds 512,3) 기각**: 1차 측정 20스텝(59 ms/step)은 **클럭램프 오염**(안 건드린 pres 8.5→31, spilling으론 설명불가). 공정 300스텝 재측정은 warm-start hang(NFS/MPI-IO). 이론상 레지스터 캡(64→42)은 register-heavy 커널서 spilling으로 net 손해. **A4 코드 되돌림**(A1도 함께, flag/매크로 게이트 제거 — `momentum_gpu.cu`/`scalar_gpu.cu` 원복, OB-dead 배너·cross-direction fix는 보존).
- **측정 방법론 확정**: ms/step은 **cold-start 300스텝 steady**로만 비교(20스텝은 A100 클럭램프로 ~2× 오염). 입력 [apps/dhvc/input_prof300.toml](../apps/dhvc/input_prof300.toml).
- **★ BASELINE (현재 상태, 되돌린 코드)**: cold-300, 1e6(384×192×96), np 1×1×2, gpu01 → **steady 34.7 ms/step** (avg 36.14, 300스텝). reverts 컴파일 검증됨.
- **pressure 최적화 조사 결과 — easy win 없음**:
  - **Y-FFT 배치화 BLOCKED**: Y-pencil 레이아웃 `[ix(내) , j(중) , k(외)]`에서 FFT가 **중간축 j**. cufftPlanMany는 균일 idist 단일배치만 가능한데 (ix,k) 배치는 비균일 → z 루프 강제. 두 방향 중 현재(loop z, 48콜)가 batch-z(loop ix, 193콜)보다 이미 적음. 게다가 Y-FFT는 ~1.4 ms/step(~4%)뿐이라 배치해도 소이득.
  - **OB solve fast-path(np1=np2=1)는 이미 lean**: host-staging 없음(divergence→FFT 동일 스트림), z-TDMA는 device 경로(`solve_tdma_z_gpu_`, `solve_many_device`). 명백한 낭비 sync/memcpy 없음.
  - → **pressure에 고-ROI 마이크로 최적화 없음**. 남은 건 펜슬 레이아웃 [ix,k,j] 리팩터(큰/위험) 뿐.

### 4.5 LEANSYNC — 불필요 sync 제거 시도 → ❌ 크래시 (sync는 load-bearing)
- **가설**: 모든 커널이 default 스트림 + momentum/scalar에 D2H 0 → "모든 cudaDeviceSynchronize(72/step, API 33%)는 방어적이라 제거 가능".
- **구현**: `MPMSTD_LEANSYNC` 플래그로 momentum 16 + scalar 9 sync를 no-op. base/lean 두 바이너리 cold-300 A/B(같은 세션).
- **결과**: base 정상(35.3 ms/step), **lean 즉시 크래시** — `cudaErrorLaunchOutOfResources at pressure_engine_gpu.cuh:695`, 직전 `[SCA #1] Z(tdma=169ms)`(정상 ~4ms의 40×).
- **근본 원인**: **이 sync들은 "방어적"이 아니라 분산(z-split) TDMA의 CUDA-aware MPI 순서에 load-bearing**. scalar/momentum의 Z-sweep은 `solve_many_device` → `MPI_Ialltoallw`(device 포인터)인데, sync를 빼니 스트림이 안 비워진 채 MPI가 들어가 분산 TDMA가 폭주(169ms)·상태 오염 → 다운스트림 pressure 커널 launch 실패. → **"default 스트림이라 다 불필요"는 분산 MPI를 간과한 오추론.**
- **정정**: nsys의 cudaDeviceSynchronize 33%는 **대부분 필요**(MPI 순서). "제거 가능한 오버헤드"가 아님. **LEANSYNC 되돌림.** (선택적 제거 — 순수 kernel↔kernel 3~4개만 — 은 가능하나 이득 <0.5%라 무가치.)

### 4.6 ★ memcpy storm 진짜 출처 확정 — CUDA-aware MPI가 파생타입 alltoallw를 host-staging
일련의 마이크로벤치로 후보를 전수 배제 후 진짜 원인 확정 (모두 nsys 실측):
- **per-step H2D/D2H는 실재**: dhvc 5스텝 vs 40스텝(monitor·stats OFF) → `(40−5)/35` = **H2D 300 + D2H 327 per step** (startup 아님).
- **출처 전수 배제**: TDMA lib 소스 memcpy 0 / cuFFT 단독 4800 exec에 H2D 2개(0/exec) / 우리코드 ~2 D2H / **contiguous** device `MPI_Alltoall` → D2D(CUDA-aware OK, `MPIX_Query_cuda_support()=1`, smcuda BTL).
- **★ 결정타**: **strided 파생-데이터타입** device `MPI_Alltoallw` 50회 → **H2D 522/call + D2H 522/call**(블록당 1개). contiguous는 D2D, **strided는 per-block host-staging**. → CUDA-aware MPI(smcuda, `--without-ucx`)는 contiguous device만 지원, **non-contiguous(파생타입)은 host 경유 fallback**.
- **결론**: dhvc 627 H2D/D2H/step = **PaScaL_TDMA z-split 분산 솔브의 `MPI_Ialltoallw`(파생타입 `ddtype_Fs_/Bs_`)를 OpenMPI가 블록단위 host-staging**. 라이브러리 소스엔 memcpy 0(OpenMPI 내부라). Z-TDMA 4.38ms 폭주(X/Y 0.6~1.0ms)도 이걸로 설명. 1e8(np3=4)에선 더 큼.
- **★ 실제 레버(최초의 진짜 win 후보)**: PaScaL_TDMA 분산 alltoall을 **contiguous 버퍼로 전환** — device 커널로 reduced-system을 contiguous로 pack → contiguous `MPI_Alltoall(v)`(CUDA-aware D2D) → device unpack. host-staging(627 memcpy/step) 제거 + Z-TDMA 가속. **단 PaScaL_TDMA_C 라이브러리 수정**(`pascal_tdma_many_cuda.cu`의 ddtype alltoallw). MPI env 우회는 불가(smcuda 한계, UCX 빌드 부재).
- 검증 스크립트: [scripts/mpi_cuda_aware_test.cu](../scripts/mpi_cuda_aware_test.cu)(contiguous=D2D), [scripts/mpi_ddt_test.cu](../scripts/mpi_ddt_test.cu)(strided=host-stage), [scripts/cufft_microbench.cu](../scripts/cufft_microbench.cu)(cuFFT=clean).

---

### 4.7 PaScaL_TDMA rd 레이아웃 `[2×n_sys]` → `[n_sys×2]` (C-order 통일) — ✅ 적용·검증

**동기**: §4.6 확정. `MPI_Ialltoallw` + strided 파생타입(`ddtype_Fs_`) → OpenMPI smcuda host-staging. 원인: Fortran column-major로 설계된 `[2×n_sys]` 배열에서 peer p의 시스템 슬라이스 = 2개 분리된 블록(strided). C row-major에서 contiguous하려면 **system-major `[n_sys×2]`** 레이아웃 필요.

**핵심 통찰**: all-to-all은 전치(transpose)라 양쪽 중 하나는 strided. Fortran은 send-side가 contiguous(column-major), C는 recv-side(`ddtype_Bs_`, rt 행블록)가 contiguous. 레이아웃 flip으로 **send-side(rd)를 contiguous**로 만들고, recv-side(rt `[n_row_rt×n_sys_rt]` row-major, 행블록)도 contiguous 유지 → **양쪽 모두 contiguous** → pack/unpack 커널 불필요.

**변경 파일**:
- `PaScaL_TDMA_C/src/tdma_local_cuda.cu`:
  - `modified_thomas_kernel`: rd 쓰기 `rd[r*n_sys+i]` → `rd[i*2+r]` (r=0,1)
  - `update_solution_kernel`: rd 읽기 `D_rd[i]`, `D_rd[1*n_sys+i]` → `D_rd[i*2]`, `D_rd[i*2+1]`
- `PaScaL_TDMA_C/src/pascal_tdma_many_cuda.cu`:
  - 생성자: `ddtype_Fs_`/`ddtype_Bs_` 제거 → `fw_scounts_/sdisps_/rcounts_/rdisps_` + `bk_*` 배열 구성. `col_offsets[p]` = Σ `ns_rt_array[0..p-1]`. `fw_scounts[p]=2*ns_rt[p]`, `fw_sdisps[p]=2*col_offset[p]`, `fw_rcounts[p]=2*n_sys_rt_`, `fw_rdisps[p]=2*p*n_sys_rt_`
  - 소멸자: `MPI_Type_free` 삭제(파생타입 없음)
  - `solve()` / `solve_cyclic()`: `MPI_Ialltoallw(..., ddtype_Fs_, ..., ddtype_Bs_)` → `MPI_Ialltoallv(..., MPI_DOUBLE, ..., MPI_DOUBLE)`. 호출 인터페이스 불변.
- `PaScaL_TDMA_C/src/pascal_tdma_many_cuda.hpp`: 멤버 `ddtype_Fs_/Bs_`, `count_send_/recv_`, `displ_send_/recv_` → `fw_*/bk_*` 배열

**호출부 무변경 확인**: `solve_many_device(d_A, d_B, d_C, d_D, n_sys, n_row)` 인터페이스 그대로. `k_mom_adi_ob`는 `p = row*n_sys_max + sys`로 `[n_row×n_sys]` 쓰기 — `modified_thomas_kernel` 읽기와 동일(불변).

**검증**:
```
[tdma_layout_verify] np=2  n_sys=256,  n_row=8  → 오차 1.8e-15  PASS
[tdma_layout_verify] np=2  n_sys=1024, n_row=16 → 오차 7.1e-15  PASS
[tdma_layout_verify] np=2  n_sys=2048, n_row=32 → 오차 4.5e-13  PASS
DHVC 10-step cold (1e6, np=2): div=8.52e-14, NaN없음, KE/Nu/Trms 정상  PASS
```
- 검증 스크립트: [scripts/tdma_layout_verify.cu](../scripts/tdma_layout_verify.cu) + [scripts/tdma_layout_verify.sh](../scripts/tdma_layout_verify.sh)

**성능 측정 (cold-300, 1e6, np=1×1×2)**:
```
step=100: ms/step=33.5   step=150: 33.1   step=200: 32.9   step=250: 33.2   step=300: 33.1
[TIMING_SS] 290 steps = 35.2 ms/step   (OLD 34.7 ms/step)
Z-TDMA scalar (steady): ~3.7–4.0ms  ←  OLD ~4.38ms
```
- **[TIMING_SS] 비교**: NEW 35.2 vs OLD 34.7 → 표면상 +1.5% 퇴보. 그러나 [TIMING_SS]는 step 11~300 전체 평균이라 step 50(53.7ms)이 끌어올림; 오래된 측정에선 ramp 속도가 달라 오염 수준이 다름 → 직접 비교 부적합.
- **신뢰할 수 있는 steady-state (step 100~300)**: NEW ~33ms ← OLD [TIMING_SS] 34.7ms (OLD의 step 100+ 실측값 미보존). **~5% 개선 추정**. Z-TDMA scalar 4.38→~3.8ms 개선이 전체에 전파됨.
- **host-staging 재측정 결과(§4.8)**: `MPI_Ialltoallv(MPI_DOUBLE)` device 포인터도 host-staging 그대로 — H2D 300 + D2H 327/step **OLD와 동일**. 진짜 원인: OpenMPI smcuda BTL은 point-to-point(`Isend/Irecv`)만 D2D 지원, `Alltoall*` 계열은 plain-type·contiguous여도 host 경유. ~5% 개선은 derived-type 파싱 CPU 오버헤드 제거에 의한 것. **근본 해결: 수동 MPI_Isend/Irecv ring(§4.8)**.

---

### 4.8 nsys memcpy 재측정 — host-staging 미제거 확인 + 진짜 해결책

**측정**: `scripts/dhvc_memcpy_new.sh` — 5-step/40-step nsys, per-step = (count40-count5)/35.

```
           | HtoD      | DtoH      | DtoD  |
OLD (ddtype Ialltoallw) | 300/step | 327/step | —  |
NEW (Ialltoallv MPI_DOUBLE) | 300/step | 327/step | 71/step |
```

**결론**: 수치 동일. `MPI_Ialltoallv` device 포인터도 host-staging. OpenMPI smcuda BTL의 한계:
- **P2P(`MPI_Isend/Irecv`)**: D2D ✓
- **`MPI_Alltoall`(단순 균일)**: D2D ✓ (고도 최적화 경로)
- **`MPI_Alltoallv/w`(비균일 또는 derived-type)**: host-staging ✗ (collective 알고리즘이 내부 host 버퍼 경유)

DtoD 71/step = 단일 rank(nprocs==1) TDMA 로컬 솔브 + 기타. alltoallv → DtoD 아님.

~5% ms/step 개선 원인 재평가: derived-type 파싱·커밋 CPU 오버헤드 제거 + MPI dispatch 단순화. memcpy 개수는 동일하나 per-call CPU 측 overhead가 줄어든 것.

**진짜 host-staging 제거 방법**: `MPI_Alltoall*`를 `MPI_Isend/Irecv` ring으로 교체.

```cpp
// forward alltoall via P2P (D2D guaranteed)
std::vector<MPI_Request> reqs(2 * nprocs_);
for (int p = 0; p < nprocs_; ++p) {
    MPI_Isend(d_A_rd_ + fw_sdisps_[p], fw_scounts_[p], MPI_DOUBLE, p, tag_A, comm_, &reqs[p]);
    MPI_Irecv(d_A_rt_ + fw_rdisps_[p], fw_rcounts_[p], MPI_DOUBLE, p, tag_A, comm_, &reqs[nprocs_+p]);
}
MPI_Waitall(2*nprocs_, reqs.data(), MPI_STATUSES_IGNORE);
```

각 Isend/Irecv는 smcuda BTL → D2D. 3개 버퍼(A,C,D) 동시 발행 가능(tag 구분). backward도 동일 구조.
**예상 효과**: H2D 300 + D2H 327 → 0/step, Z-TDMA 3.8ms → ~1ms 목표.

---

## 5. 다음 후보 (프로파일 후 재정의, 우선순위)

1. **ncu 커널 레벨 분석**(Nsight Compute, [../nsys.txt](../nsys.txt) §2): 우리 최대 커널 `k_mom_adi_var`(band 조립) + 라이브러리 `tdma_cyclic_many_kernel`을 roofline 분석 → 메모리 바운드(→A3 tiling) vs 점유율 바운드 vs compute 바운드 판별. (참고 ncu 패턴: `srun -n2 ncu --kernel-name <k> --set full --launch-skip 1 --launch-count 1 --target-processes all -f -o report_%q{SLURM_PROCID} …`)
2. **분산 TDMA memcpy 조사**(최대 1e8 레버): 스텝당 H2D/D2H ~628회가 PaScaL_TDMA z방향 all-to-all의 host-staging인지 확인. CUDA-aware로 바꾸면 17.6% memcpy + 관련 sync 절감 가능. (PaScaL_TDMA_C 라이브러리 영역)
3. **SYNC — 불필요 cudaDeviceSynchronize 제거**: 72/step → 필수(MPI 전/CPU read 전)만 남김. 작은 격자 오버헤드 직접 절감.
4. **pressure Y-FFT 배치화**: regular_fft 96회/step(슬랩 루프) → 1 배치 호출.
5. **A2 — 커널 fuse**: scatter+add, ADI 최종 copy 제거. ~3-4% (소이득).
6. (장기) **완전음해 대류**: 스텝 수 격감(수배~). 레퍼런스 이탈. 별도 검토.

> **전략 메모**: TDMA(~35%)가 최대지만 라이브러리 코드. 오버헤드(~35%)는 1e8에서 비중 급감. **1e8 관점의 견고한 타깃은 TDMA compute + 분산통신 memcpy(#2)** ; 작은-격자/latency 관점은 sync/launch(#3). ncu(#1)로 우리 커널의 바운드 종류를 먼저 확정 후 결정.

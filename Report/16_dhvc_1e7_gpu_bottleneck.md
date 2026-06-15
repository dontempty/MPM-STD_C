# DHVC Ra=1e7 GPU 성능 병목 분석 보고서 (2026-06-15)

> 대상: `apps/dhvc/main_gpu.cu` (OB DHVC), GPU 빌드, 케이스 Ra=1e7
> 격자: n1m=864 × n2m=192 × n3m=416 (~68.9M 셀), 분할 **np1=1, np2=1, np3=2** (z-split, 2×A100 on gpu02)
> 입력: `apps/dhvc/input_fig7_gpu_1e7.toml`
> 방법: 코드 전수 분석(5개 병렬 프로브) + 적대적 검증 + 실측 로그(`timing_1e7.log`) 대조
> ⚠ 분석 산출물입니다. **소스 코드는 수정하지 않았습니다.**

---

## 0. 결론 (Executive Summary)

**병목은 연산(kernel)이 아니라 데이터 이동이다.** 스텝 시간의 **절반 이상(~55%)이 "필드를 통째로 GPU↔CPU로 복사"하는 host-staging PCIe 전송**이고, 그 유효 대역폭은 **~2.09 GB/s**(이론 pinned PCIe의 1/5~1/6)에 불과하다. 여기에 분산 z-TDMA의 **device-pointer MPI** 호출당 고정 오버헤드가 더해진다.

이 비용은 **RBC 128³(2.1M 셀, 29 ms/step)에서는 보이지 않다가 DHVC 1e7(68.9M 셀)에서 폭발**한다. "셀 33배 → 시간 286배(선형 대비 8.7배)"의 정체는 알고리즘 초선형이 아니라, **느린 데이터-이동 경로(~2 GB/s)가 빠른 on-GPU 커널(~1.5 TB/s)을 대규모에서 추월하는 crossover** 현상이다.

| 지표 | 값 |
|---|---|
| 실측 스텝 시간 (첫 10스텝 평균) | **8,260.6 ms/step** |
| 실측 정상상태 | **7,445.8 ms/step** |
| 50,000 step 예상 소요 | **~103시간** → 현 상태로 1e7 검증 사실상 불가 |
| host-staging PCIe (halo+band) | **~9.5 GB/step @ ~2 GB/s ≈ 4,400–4,600 ms/step (~55%)** |

**1순위 수정 2개(halo 경계면-only + scalar 벽닫힘 on-device)로 8,260 → 약 3,800 ms/step**(절반 이하) 회수 가능.

---

## 1. 실측 분해 (timing_1e7.log — 추정 아님)

개발자가 이미 per-phase 타이머를 넣어두어, 측정값으로 단계별 비용을 확정할 수 있다.

```
[TIMING]    10 steps  total = 8260.6 ms/step
[TIMING_SS] steady-state    = 7445.8 ms/step
[TIMING_BREAKDOWN] cfl=1.0  scalar=3223.6  mom=1555.9  corr=0.9  sync=1352.9  pres=2126.3 ms
```

| 단계 | 실측 ms/step | 비중 | 코드-레벨 원인 (검증) |
|---|---|---|---|
| **scalar** | 3223.6 | 39% | z-TDMA **band 전체 8회 cudaMemcpy**(~1,100ms) + ADI 커널(68.9M×3) + 분산 z-TDMA MPI |
| **pres** | 2126.3 | 26% | **필드 전체 halo 4회**(~1,080ms) + on-device cuFFT + z-TDMA MPI(8 Ialltoallw) |
| **mom** | 1555.9 | 19% | **필드 전체 halo 4회**(~900ms) + ADI 커널 + z-TDMA MPI(12 Ialltoallw) |
| **sync** | 1352.9 | 16% | **필드 전체 halo 5회** = 2,822MB ÷ **2.09 GB/s** ← 대역폭 보정 기준점 |
| cfl+corr | 1.9 | ~0% | 무시 |

`sync=1352.9 ms`가 정확히 halo 5회(2,822MB)에 대응 → **유효 대역폭 = 2.09 GB/s**로 역산된다. 이 값이 모든 host-staging 비용의 보정 상수다.

---

## 2. 근본 원인 — host-staging PCIe (~55%)

### (A) halo 교환이 매 호출마다 "필드 전체"를 복사 — `src/core/gpu/halo_gpu.cu:33-35`

```cpp
const std::size_t n = field.n_elements();          // 필드 전체
cudaMemcpy(h_buf, field.data(), nbytes, D2H);       // 전체 GPU→CPU
sub.exchange_halo(h_buf);                            // CPU halo (실제 교환은 경계면뿐)
cudaMemcpy(field.data(), h_buf, nbytes, H2D);       // 전체 CPU→GPU
```

- 실제로 교환해야 할 것은 z-경계면(864×192 ≈ 1.3MB)뿐인데, 랭크당 필드 **282MB(866×194×210)** 를 통째로 왕복(564MB)한다.
- 원 우회책은 RBC의 "9MB" 필드 기준(주석 `halo_gpu.cu:13-19`)으로 `~2ms`라 무해했다. CUDA-10 OpenMPI + A100에서 device 포인터를 MPI에 넘기면 `cudaPointerGetAttributes`가 ~200ms/call로 느려지는 것을 피하려는 의도였다. 필드가 31배 커지면서 이 우회책 자체가 임계경로가 됨.
- **스텝당 호출 13회** (적대적 검증으로 확정):

| 위치 | 횟수 | codeRef |
|---|---|---|
| main loop (T, U, V, W, W-재동기) | 5 | `apps/dhvc/main_gpu.cu:422,436,437,438,447` |
| momentum (dU, dV, dW, dV-after-couple) | 4 | `momentum_gpu.cu:1354,1355,1356,1368` |
| pressure (P, U, V, W projection 후) | 4 | `pressure_engine_gpu.cuh:1290,1295,1296,1297` |
| **합** | **13** | |

→ **13 × 564MB = 7.34 GB/step @ 2.09 GB/s ≈ 3,300–3,500 ms/step.**

### (B) OB scalar z-TDMA가 band 전체를 CPU로 내렸다 올림 — `src/equation/scalar/gpu/scalar_gpu.cu:341-355`

벽 경계 닫힘을 CPU(`modify_tdma_row_cpu`)에서 처리하려고 A/B/C/D 4개 band(각 ~276MB)를 **8회 cudaMemcpy**(D2H 4 + H2D 4)로 왕복. 게다가 버퍼가 **pageable `std::vector`**.

```cpp
sys.A.resize(sz); ... // sz = n_row × n_sys = 208 × 165,888 = 34.5M
cudaMemcpy(sys.A.data(), d_A_, ..., D2H);   // ×4 (A,B,C,D)
core::modify_tdma_row_cpu(...);             // 실제론 벽 2줄만 수정
cudaMemcpy(d_A_, sys.A.data(), ..., H2D);   // ×4
.solve_many_device(...)                      // TDMA 자체는 on-device
```

- 정정(적대적 검증): `modify_tdma_row_cpu`는 **벽 2줄(n_sys 루프 2회)만** 건드린다. CPU 연산은 사소하고, 비용은 **순수하게 2.21 GB의 band cudaMemcpy**다.
- **이 패턴을 쓰는 건 OB scalar 경로뿐.** momentum(`momentum_gpu.cu:1096`)과 scalar-NOB(`scalar_gpu.cu:484`)는 이미 on-device 커널 `k_modify_tdma_row_gpu`로 처리한다.
- DHVC는 OB 경로(`main_gpu.cu:420-421`)를 타므로 매 스텝 발생.

→ **8 × 276MB = 2.21 GB/step @ 2.09 GB/s ≈ ~1,100 ms/step.**

### (C) 분산 z-TDMA가 device 포인터를 MPI에 직접 전달 — `PaScaL_TDMA_C/src/pascal_tdma_many_cuda.cu:158-180`

`d_A_rd_, d_C_rd_, d_D_rd_, d_D_rt_`(**device 포인터**)를 그대로 `MPI_Ialltoallw`에 넘긴다(solve당 4회 = 3 forward + 1 back, 각 앞에 `cudaDeviceSynchronize`). 이는 (A)의 halo가 일부러 피하려던 바로 그 device-ptr MPI 패턴이다.

- np3=2이므로 z축만 분산 → 스텝당 분산 z-TDMA solve = **9회**(scalar 1 + momentum 6 + pressure 2) × 4 = **~24~36 device-ptr collective/step**.
- reduced system 바이트는 작으므로(2행 단면), 비용이 있다면 **호출당 고정 오버헤드**다. 셀 수와 무관하게 더해지는 항.
- 정정(적대적 검증): 종합본의 "~200ms/call"은 **과대**. 24~36회/step × 200ms = 4,800~7,200ms로 버킷 합과 모순. 실제 호출당 페널티는 그보다 작지만 0은 아니며, scalar/pres/mom 버킷의 "kernel 외 잔여"에 섞여 있다. **정확한 분해는 nsys 필요(§5).**

### 합산

| 항목 | 바이트/step | ms/step (@2.09 GB/s) |
|---|---|---|
| 필드 전체 halo ×13 | 7.34 GB | ~3,300–3,500 |
| scalar band staging ×8 | 2.21 GB | ~1,100 |
| **host-staging 소계** | **~9.5 GB** | **~4,400–4,600 (~55%)** |
| 나머지(ADI 커널, on-device cuFFT, device-ptr MPI) | — | ~2,900–3,800 |

---

## 3. 왜 "선형보다 8.7배" 느린가 — crossover

- 셀 33배(2.1M→68.9M). 모든 전송 바이트는 **선형**으로 증가(필드 31.6×, band 32.9×).
- on-GPU 커널 ~1.5 TB/s vs host-staging ~2 GB/s → **약 750배 속도차**.
- **RBC(2.1M):** halo 9MB·~2ms, MPI ~36회도 29ms 예산의 작은 일부 → 시간 대부분이 진짜 연산. host-staging이 커널 뒤에 가려짐.
- **DHVC(68.9M):** 같은 코드가 halo 282MB·수백ms로 커져 **데이터 이동이 벽시계를 지배**. 바이트는 선형이지만 *느린 경로*로 흘러 전체가 초선형처럼 보임 + device-ptr MPI 고정 오버헤드가 표면화.
- 즉 **알고리즘 O(N²)가 아니라 대역폭 crossover + 호출당 MPI 상수**다.

### 배제된 가설 (코드로 확인)
- ❌ cuFFT 플랜 재생성 — 생성자에서 1회만(`pressure_engine_gpu.cuh:562-577`), 스텝당 `cufftExec*`만.
- ❌ 스텝당 cudaMalloc/Free churn — 워크스페이스 grow-only, step 0 이후 no-op(`momentum_gpu.cu:90-110`).
- ❌ 메모리 용량 한계 — 랭크당 ~9–11 GB / 40 GB A100(~25%), allocator 압박 없음.
- ❌ TDMA 펜슬 전체 staging(`solve_many`) — 존재하지만 **hot-path 호출자 0**. 모든 GPU 호출자는 `*_device` 변형 사용.
- ❌ pressure 파이프라인 — fully on-device fast path(`gpu_fast_path_ = np1==1 && np2==1` = TRUE, `pressure_engine_gpu.cuh:511`).

---

## 4. 수정 우선순위 (효과 큰 순 — 제안만, 코드 미수정)

| 순위 | 수정 | 위치 | 회수 예상 | 비고 |
|---|---|---|---|---|
| **1** | **halo를 "경계면(face)만" pack/unpack** (필드 전체 복사 제거) | `src/core/gpu/halo_gpu.cu:33-35` | **~3,000–3,300 ms/step** | 13회 halo 전부를 한 번에 공격. 564MB→수MB/call |
| **2** | **OB scalar 벽 닫힘을 on-device 커널로** (기존 `k_modify_tdma_row_gpu` 재사용) | `src/equation/scalar/gpu/scalar_gpu.cu:341-355` | **~1,100 ms/step** | 저위험(momentum·scalar-NOB이 이미 사용) |
| **3** | 분산 z-TDMA의 **reduced system(작음)만 pinned 버퍼 경유** 후 alltoallw | `PaScaL_TDMA_C/src/pascal_tdma_many_cuda.cu:158-180` | nsys 확인 필요(수백ms 추정) | halo 우회책과 동일 논리 |
| 4 | pressure y-FFT 배치화(416 launch → batched) | `pressure_engine_gpu.cuh:1023-1030,1097-1105` | 소폭(~수ms) | 1–3 이후 정리용 |

**1+2만으로 8,260 → 약 3,800 ms/step.**

### 빠른 진단/대안 — np3=1 단일 GPU (80GB A100)
z 분할이 없어지면 **랭크간 halo MPI와 분산 z-TDMA의 device-ptr MPI가 통째로 사라진다**. 메모리는 ~12–18 GB로 80GB에 충분히 적재 가능.
- GPU 1장(연산 절반)이 2장보다 **빠르면** → 병목이 데이터 이동/MPI라는 결정적 증거 + 즉시 실용적 이득.
- 단, 멀티-GPU 확장(P8)을 위해서는 결국 1·2번(face-only halo, on-device closure)이 본 해법.

---

## 5. 확실한 것 vs nsys 필요

**코드+로그로 확정:**
- 실측 분해(8,260.6 / 7,445.8 ms/step, `timing_1e7.log`).
- halo 전체-필드 복사 13회/step(`halo_gpu.cu:33-35` + 13개 호출지점).
- OB scalar band 8회 복사 2.21 GB/step(`scalar_gpu.cu:341-355`).
- z-TDMA의 device-ptr `MPI_Ialltoallw`(`pascal_tdma_many_cuda.cu:158-180`).
- 유효 대역폭 ~2.09 GB/s(sync 버킷 역산).
- cuFFT/malloc/메모리/펜슬-staging 모두 배제.

**nsys로 확인할 것(버킷 내부 분해):**
- scalar=3,223.6 중 band-staging(~1,100) 외 **잔여 ~2,000ms**가 ADI 커널인지 device-ptr MPI인지의 정확한 비율.
- 실제 cudaMemcpy 총 GB/s, `cudaPointerGetAttributes`/`MPI_Ialltoallw` 누적시간.

```bash
nsys profile --stats=true -t cuda,mpi,nvtx --cuda-memory-usage=true --force-overwrite=true \
  -o /tmp/dhvc_r%q{OMPI_COMM_WORLD_RANK} \
  mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_fig7_gpu_1e7_n20.toml
# 확인: "CUDA API Summary"의 cudaPointerGetAttributes, "MPI Event Summary"의 MPI_Ialltoallw,
#       "CUDA Memory Operation Statistics"의 D2H/H2D GB/s 및 총 바이트(예상 ~9.5 GB/step)
```

---

## 6. 참조 파일 (절대경로)

- `src/core/gpu/halo_gpu.cu:13-19,33-35` — 전체-필드 halo + 우회책 주석(수정 #1)
- `src/equation/scalar/gpu/scalar_gpu.cu:341-355` — OB scalar band staging(수정 #2); `:484` 기존 on-device 커널
- `src/equation/momentum/gpu/momentum_gpu.cu:1096,1354-1368` — on-device 패턴 레퍼런스 + momentum halo
- `src/equation/pressure/gpu/pressure_engine_gpu.cuh:511,1290-1297` — fast path 조건 + projection 후 halo 4회
- `../PaScaL_TDMA_C/src/pascal_tdma_many_cuda.cu:158-180` — device-ptr Ialltoallw(수정 #3)
- `apps/dhvc/main_gpu.cu:420-422,436-447` — 시간루프 + halo 호출지점
- `apps/dhvc/logs/timing_1e7.log` — 실측 per-phase 분해

---

## 7. 한 줄 요약

> DHVC 1e7이 느린 건 솔버 알고리즘 때문이 아니라, **halo와 scalar 벽-닫힘이 매 스텝 ~9.5GB의 필드/밴드 전체를 ~2GB/s의 host-staging 경로로 왕복**하기 때문이다. **halo를 경계면만 교환하도록 바꾸고(#1) OB scalar 벽닫힘을 on-device로 옮기면(#2)** 스텝 시간이 절반 이하로 떨어져 1e7 GPU 검증이 비로소 현실화된다.

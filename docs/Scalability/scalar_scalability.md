# scalar_solve(에너지 방정식) 분해 측정과 최적화 (2026-10-17)

`scalar::solve_variable_property`(가변 κ̂ 융합 ADI: x → y → z 스윕 + 증분 적용)가 부르는 호출 10개를 구간으로 잰다
(`scalar::Solver::timer`, 이름 `scalar.<build|tdma|scatter>_<x|y|z>`, `scalar.add_increment`). momentum 과 같은 틀.
단위 ms/step(랭크 max). 노트북 `apps/rbc_nob/plot/scalar_scalability.ipynb`. 기종은 꼬리표마다 환경 파일에서 읽어 그림 제목에 그대로 쓴다.

| 꼬리표 | 기종 | np |
|---|---|---|
| `h200k` → `h200z` (전 → 후) | NVIDIA H200 ×4 @ gpu55 | 1·2·4 |
| `v100e` → `v100z` (전 → 후) | Tesla V100-SXM2-32GB ×8 @ gpu04 | 1·2·4·8 (512³ 는 2·4·8 — np 1 은 32 GB 에 안 들어감) |

## 1. 분해 결과 — np 1 (변경 전)

| 호출 | H200 256³ (scalar 7.0) | H200 512³ (62.7) | V100 256³ (41.7) | momentum 성분 하나 대비 (H200 256³) |
|---|---|---|---|---|
| build_x (첫 스윕: 세 축 R 융합 조립) | 1.92 (27 %) | 16.0 (26 %) | 5.7 (14 %) | 3.4 배 — **의도된 것**: 우변 R 을 세 축 모두 여기서 만든다 |
| build_y | 0.80 (11 %) | 6.8 (11 %) | 3.6 (9 %) | 1.4 배 |
| **build_z** | **2.01 (29 %)** | **24.5 (39 %)** | **23.1 (55 %)** | 3.3 배 → momentum 과 같은 전치 문제 |
| tdma_x, tdma_y | 0.67 ×2 | 4.1 ×2 | 2.6 ×2 | 1.0 (같은 커널) |
| tdma_z | 0.45 | 2.6 | 1.6 | 1.0 |
| scatter ×3, add_increment | 0.5 | 4.5 | 2.6 | 1.0 |

- **비정상은 build_z 하나**: build_y 의 2.5 배(H200 256³)·3.6 배(512³)·**6.4 배(V100)**. 같은 식인데 z 만 느리다.
- build_x 가 큰 것은 정상: 델타형 3-스윕 ADI 의 우변 R(세 축의 확산·이류·dκ̂ 선형화)을 첫 스윕에 융합해 두었기 때문(`scalar_stencil.hpp` 머리말). 세 축 몫 ≈ 3 배.
- 확장 손실은 momentum 과 같이 tdma_x·y 의 np 4 바닥(0.67 → 0.46, 이상 0.17)과 weak 의 tdma_z(0.45 → 0.90)뿐.

## 2. build_z 의 원인과 수정

원인은 momentum 과 동일 (`momentum_scalability.md` §2): z 밴드 `[k·(nx·ny) + i·ny + j]` 와 필드 `[(i·ny + j)·nz + k]` 가 (j,k) 전치 관계인데
공용 1차원 커널이 밴드 순서(j 연속)로 스레드를 놓아 `build_cell` 의 필드 로드가 전부 nz 보폭으로 흩어졌다.

**수정** (`src/equation/scalar/system/system.cu` `build_z_kernel`): momentum 의 `build_kernel_z` 와 같은 (32×8) 블록 + 공유 메모리 32×33 타일 4장.
k 연속으로 읽어 계수를 타일에 두고, 전치해 j 연속으로 밴드에 쓴다. 식(`build_cell`)은 그대로 → **비트 동일** (게이트: 스모크 FINAL·200 스텝 모니터 전 열 일치, `check_gate.py` PASS).
상수물성 판도 같은 커널.

## 3. 전 → 후

| 조합 | build_z | scalar_solve | step_total |
|---|---|---|---|
| H200 256³ np 1 | 2.01 → 0.65 (×3.1) | 7.00 → 5.65 | 37.6 → 36.2 (−4 %) |
| H200 512³ np 1 | 24.5 → 5.1 (×4.8) | 62.7 → 43.3 | 284.0 → 264.6 (**−7 %**) |
| H200 512³ np 4 | 6.0 → 1.3 | 16.2 → 11.6 | 78.1 → 73.5 (−6 %) |
| V100 256³ np 1 | 23.1 → 2.3 (×10) | 41.7 → 20.9 | 140.0 → 119.2 (**−15 %**) |
| V100 256³ np 4 | 3.8 → 0.6 | 9.1 → 5.9 | 42.6 → 39.4 (−8 %) |
| V100 weak np 4 | 23.2 → 2.3 | 43.7 → 22.9 | 163.1 → 142.3 (−13 %) |

build_z 가 build_y 의 0.8 배(H200)·0.6 배(V100)로 내려와 x·y 조립과 같은 수준이 됐다. scalar_solve 가 V100 에서 절반.
**원본 대비 누적**(momentum 2회차 + scalar): H200 256³ np 1 41.7 → 36.2 (−13 %), 512³ 329 → 265 (**−20 %**), V100 256³ 198 → 119 (**−40 %**).

## 4. np 8 (V100 gpu04 8장) — 통신이 전부였고, 원인은 코드 안의 "자기 자신에게 보내는 MPI"

| 256³ (halo 수정 전) | strong np 4 → 8 | weak np 4 → 8 |
|---|---|---|
| step_total | 42.6 → 42.3 (효율 82 → **41 %**) | 163 → 230 (86 → **61 %**) |
| halo 동기화 전체(main 4 + momentum 4) | 4.9 → 12.1 | 11.5 → 71.4 |
| TDMA Alltoallv (3 풀이) | 2.8 → 9.9 | 3.6 → 10.9 |

**진단** (`apps/rbc_nob/scalability/diag/`, gpu04):
1. 순수 MPI: 0.5 MB 디바이스↔디바이스 Sendrecv 는 NVLink 쌍 16–24 GB/s, 소켓 간(SYS) 쌍 5.5 GB/s, 8랭크 링 교환 0.14 ms — **하드웨어·UCX 는 정상**. 모든 쌍 P2P 가능.
2. 솔버 없이 `Subdomain::exchange_halo` 만 반복(`apps/bench_halo`): 솔버 안 수치를 그대로 재현 (weak np 8 교환 1회 6.0 ms, strong 1.0 ms). Barrier 로 맞춰도 같음(기다림 아님),
   GPU 순서·바인딩·`UCX_TLS` 무효. 비용 ∝ np² × (x·y 면 크기).
3. **원인**: x·y 는 주기 축에 npx = npy = 1 이라 `MPI_Cart_shift` 가 이웃을 **자기 자신**으로 준다 → 교환마다 x·y 면 4장을 자기 자신에게 `MPI_Sendrecv`.
   UCX 가 디바이스 메모리의 자기 전송을 호스트 경유로 처리하고 노드 안 랭크가 늘수록 급격히 느려진다 (np 1 0.1 ms → np 8 6 ms). H200 노드는 이 경로가 빨라 티가 안 났다.

**수정** (`src/parallel/subdomain/subdomain_halo.cu`, 2026-10-17): 이웃이 자기 자신이면 MPI 대신 디바이스 복사(x: `cudaMemcpy` D2D, y: 팩 버퍼를 바로 반대쪽 ghost 에 언팩).
값이 같은 복사라 **비트 동일** (스모크 FINAL·모니터 일치). CPU 경로(`subdomain.cpp`)는 호스트 self-send 라 손대지 않음.

| V100 gpu04, 256³ (전 → 후) | exchange_halo 1회 | step_total | 효율 |
|---|---|---|---|
| strong np 4 | 0.46 → 0.20 ms | 39.4 → 36.1 (−8 %) | 76 → 82 % |
| strong np 8 | 1.00 → 0.25 ms | 41.0 → 31.5 (**−23 %**) | 36 → 47 % |
| weak np 4 | 1.25 → 0.24 ms | 142 → 130 (−9 %) | 84 → 92 % |
| weak np 8 | 5.89 → 0.29 ms | 210 → 137 (**−35 %**) | 57 → **86 %** |

- weak 는 이제 86 % 로 정상 범위. strong 256³ np 8 의 47 % 는 남은 통신 = **분산 TDMA 의 Alltoallv**(momentum 3 풀이 6.0 ms + scalar·pressure 각 2 ms, 스텝 31.5 ms 의 1/3)
  와 랭크당 격자가 작은(nz_local 32) 데서 오는 x·y TDMA 바닥. 이것이 다음 통신 항목.
- np 1 에서도 교환 1회 0.18 → 0.14 ms (자기 전송 4개가 복사로) — H200 포함 모든 조합에서 스텝 1–4 % 이득.

## 5. 랭크→GPU 순서 실험 — 불필요해짐
원인이 토폴로지가 아니었으므로(순수 MPI 링은 기본 순서로도 0.14 ms) GPU 순서 재배치는 의미가 없다. 첫 시도는 Slurm 의 `CUDA_VISIBLE_DEVICES` 덮어쓰기로 무효였고 재실행하지 않았다.

## 노트북 양식 (2026-10-17 밤)
`apps/rbc_nob/plot/scalar_scalability.ipynb`: §A 최적화 전(`h200k`·`v100e`) 이벤트별 막대 → 이벤트별 scaling, §B 최적화 후(최신 바이너리 `h200h`·`v100h`) 같은 그림,
§C 전 vs 후 — 이벤트별 전/후 막대 + 감소량(전 − 후) 막대 + 요약표. 기종·노드·꼬리표·문제 크기를 제목·범례에, 그림 저장 없음.

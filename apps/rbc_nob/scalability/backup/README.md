# 백업 — 조각 분할 순환 Thomas (3회차, 2026-10-17, 본 코드에서 제외)

`pascal_tdma_cuda_segmented_20261017.cu` = 조각 분할 커널(`tdma_cycl_segmented_kernel`, `choose_segments`, `launch_tdma_cycl_segmented`)이
들어 있던 시점의 `src/linear_solver/tdma/PaScaL_TDMA/pascal_tdma_cuda.cu` 전체. `segmented_cyclic_tdma_20261017.patch` = 본 코드(2회차) 대비 diff.

제외 사유 (사용자 결정): 본런 격자(512³+)에서 이득이 2~3 % 뿐이고 코드만 복잡해진다. 작은 격자를 여러 GPU 로 돌릴 때(H200 256³ np 4: 스텝 −18 %, 효율 61 → 75 %)만
효과가 있다. 측정·해석은 `docs/Scalability/momentum_scalability.md` §6. 되살리려면 patch 를 적용하고 `PASCAL_TDMA_SEGMENTS` 로 p 를 고른다.

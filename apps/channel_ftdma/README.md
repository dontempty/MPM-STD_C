# channel_ftdma — Filtered_TDMA 검증용 채널 앱

운동량 ADI 의 TDMA 를 PaScaL(정확해) ↔ Filtered_TDMA(절단 근사)로 바꿔 같은 채널 DNS 를 돌리고
**통계·시간·절단 깊이 J** 를 비교하는 앱 (시간은 통계 런 자체의 모니터 sec/step·잡 벽시계) (2026-10-28). 설계·결과 정리는 `docs/Design/20261028_filtered_tdma.md`.

- `main.cpp` = `apps/channel/main.cpp` 사본 + **J 모니터**. 설정은 channel 의 것을 그대로 쓴다(`settings.cpp` 1줄 TU).
- 백엔드는 빌드가 정한다 (로그 첫 줄 `tdma=pascal|filtered`). 압력 z TDMA 는 두 빌드 모두 PaScaL.
- J 모니터: 모니터 스텝마다 운동량 x·y·z 플랜의 J·fallback 을 성분(dU·dV·dW)·랭크별로 `stats_<tag>/truncation_J.dat` 에 쓴다
  (filtered 빌드에서만; J 읽기가 디바이스 동기화라 모니터 스텝에만 켠다. 랭크 안에서 완결되는 축은 J = −1).

## 빌드 (BUILD_HANGANG.md 의 환경)

```bash
A=(GPU=1 -j32 CXX="CC --gcc-toolchain=/usr/bin/gcc-14" NVHPC_DIR=$NVIDIA_PATH CUDA_VER=12.9 GPU_ARCH=cc90 \
   EXT_INC=-I$NVIDIA_PATH/math_libs/12.9/include)
make "${A[@]}" channel_ftdma channel_ftdma_scale                # PaScaL  → build_gpu/bin/
make "${A[@]}" TDMA=filtered channel_ftdma channel_ftdma_scale  # Filtered → build_gpu_filtered/bin/
```

`main_scale.cpp` 는 `python3 tools/scale/instrument_main.py apps/channel_ftdma/main.cpp -o apps/channel_ftdma/main_scale.cpp`
로 만든다 (main.cpp 를 고치면 다시 생성).

## 런

| 파일 | 내용 |
|---|---|
| `input_dns256_{pascal,filtered}_2x2x2.ini` | 256³ DNS Re_τ 180 (박스 4π×2π×2, Re_b 2857), 2x2x2 = GH200 8 장, 20 만 스텝, 통계 2 만 스텝부터 1 만 스텝마다 (구간 18 개), 2 만 스텝마다 체크포인트 |
| `sbatch_stat.sh` | 통계 런 하나 (`--export=ALL,TAG=dns256_<backend>_2x2x2`), 2 노드 `--exclusive` |
| `plot/plot.ipynb` | 통계(구간 평균 ± σ, z 값, MKM 180 대비)·J·시간 비교 (kernel `channel-plot`) |

```bash
sbatch -J ftdma_pascal   --export=ALL,TAG=dns256_pascal_2x2x2   sbatch_stat.sh
sbatch -J ftdma_filtered --export=ALL,TAG=dns256_filtered_2x2x2 sbatch_stat.sh
```

## 산출물 이름 (런끼리 겹치지 않게)

꼬리표 `<case>_<backend>_<분할>` 하나로 모든 것을 구분한다: 통계 `stats_<tag>/`, 체크포인트 `checkpoints_<tag>/`,
로그 `log/<tag>.log`, 잡 출력 `log/ftdma_<backend>_<job>.out`(노드·시작·끝). 전부 `.gitignore`.
`archive_les64/` = 이 앱을 만들기 전 `apps/channel` 에서 돌린 LES 64³ 결정론·통계 시험(덱·로그·결과, 커밋 안 함).

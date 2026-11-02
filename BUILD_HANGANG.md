# MPM-STD 빌드·실행 — 한강(KISTI GH200) 클러스터

이 클러스터에서 MPM-STD 를 **GH200 GPU 로** 빌드하고 돌리는 방법이다.
여기 적은 모듈 조합과 명령은 2026-10-27 에 그대로 실행해 확인했다. 전 앱 10종이 빌드됐고,
natStr 의 GH200 노드(node1254)에서 `test_pressure_wall` PASS, `dhvc` 스모크(2 랭크·2 GPU)가 끝까지 정상 종료했다.

`Makefile.inc` 는 **구 클러스터(x86_64, gpu01/gpu02 A100, hpcx mpirun) 기준**이라 그대로 `make GPU=1` 하면 안 된다.
`Makefile.inc` 는 고치지 않았다. 아래처럼 **명령행 변수로 덮어써서** 빌드한다.

---

## 1. 이 클러스터에서 달라지는 점

| 항목 | 구 클러스터 (`Makefile.inc` 기본값) | 한강 (GH200) |
|---|---|---|
| CPU 아키텍처 | x86_64 | **aarch64** (Grace, neoverse-v2) — 로그인 노드 `glogin01` 도 Grace |
| GPU | A100 (`cc80`) | **GH200 120GB (`cc90`)**, 노드당 4개, 노드당 288 코어 |
| 컴파일러 | nvhpc 23.7 `mpic++` (hpcx) | Cray PE `PrgEnv-nvidia` → **`CC` 래퍼** (내부적으로 nvc++ 26.3 실행) |
| MPI | hpcx-2.15 OpenMPI, `mpirun` | **cray-mpich 9.0.1** (GPU-aware, GTL), 실행은 **`srun`** |
| CUDA | 12.2 | **12.9** (`cuda/12.9_26.3`) |
| cuFFT 헤더 경로 | `math_libs/<ver>/targets/x86_64-linux/include` | `$NVIDIA_PATH/math_libs/12.9/include` (aarch64 는 `sbsa-linux`) |
| GPU 드라이버 | — | 570.124.06 (CUDA 12.x 까지 지원; **CUDA 13 은 동작하지 않음**) |

---

## 2. 환경 (빌드 때와 실행 때 같은 것을 쓴다)

```bash
module purge
module load craype-arm-grace PrgEnv-nvidia          # nvidia/26.3 + cray-mpich/9.1.0 이 같이 올라온다
module swap cray-mpich cray-mpich/9.0.1             # ← 반드시 바꾼다 (아래 함정 ①)
module load cuda/12.9_26.3 craype-accel-nvidia90
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH   # ← 실행 때 필수 (함정 ②)
export MPICH_GPU_SUPPORT_ENABLED=1                  # ← 실행 때 필수: 디바이스 포인터를 MPI 로 직접 보낸다
```

확인: `module -t list` 결과가
`craype-arm-grace nvidia/26.3 craype/2.7.36 cray-dsmml/0.3.1 libfabric/2.3.1 craype-network-ofi cray-mpich/9.0.1 cray-libsci/26.03.0 PrgEnv-nvidia/8.7.0 cuda/12.9_26.3 craype-accel-nvidia90`
이면 된다.

---

## 3. GPU 빌드

로그인 노드에서 빌드한다(Grace 라 아키텍처가 같다. `-j32` 로 전체 빌드 약 20초).

```bash
cd /scratch/t639a05/MPM-STD_C
# (2절의 module 명령을 먼저 실행)

make GPU=1 -j32 \
  CXX="CC --gcc-toolchain=/usr/bin/gcc-14" \
  NVHPC_DIR=$NVIDIA_PATH \
  CUDA_VER=12.9 \
  GPU_ARCH=cc90 \
  EXT_INC=-I$NVIDIA_PATH/math_libs/12.9/include \
  dhvc                      # 앱 이름을 빼면 전체 빌드 → build_gpu/bin/*
```

각 덮어쓰기가 필요한 이유:

| 변수 | 값 | 이유 |
|---|---|---|
| `CXX` | `CC --gcc-toolchain=/usr/bin/gcc-14` | 원래 값 `$(MPI_DIR)/bin/mpic++` 는 이 클러스터에 없다. `CC` 래퍼가 cray-mpich·`-tp=neoverse-v2 -gpu=cc90` 를 붙여 준다. `--gcc-toolchain` 은 함정 ③ 때문에 필요하다. |
| `NVHPC_DIR` | `$NVIDIA_PATH` | `nvidia` 모듈이 `NVHPC_ROOT` 를 주지 않아서 기본값이 구 클러스터 경로(`/opt/nvidia/...23.7`)가 된다. |
| `CUDA_VER` | `12.9` | `KISTI_CUDA_VER` 가 없으니 기본값 12.2 가 된다. 12.2 는 이 SDK 에 없다. |
| `GPU_ARCH` | `cc90` | GH200 = Hopper. 기본값 `cc80` 으로 빌드하면 GH200 용 코드가 아니다. |
| `EXT_INC` | `-I$NVIDIA_PATH/math_libs/12.9/include` | 기본값은 `targets/x86_64-linux/include` 를 하드코딩하고 있다. |

명령행 변수는 `Makefile.inc` 안의 `:=` 대입보다 우선하므로 파일은 수정하지 않아도 된다.
빌드 결과는 정본 디렉터리 `build_gpu/` 로 들어간다.

빌드 로그에 `warning: CUDA C++ compilation is unsupported in nvc++; nvcc is the recommended CUDA compiler`
가 TU 마다 나오지만 무시해도 된다. 모든 TU 를 한 컴파일러로 빌드한다는 기존 설계(readme §3)에 따른 경고다.

---

## 4. 실행 (sbatch)

### 파티션과 제출 규칙 (그룹 `bs0356` 기준, 2026-10-27 확인)

- 쓸 수 있는 GPU 파티션은 **`natStr`** 하나다. `normal`·`5thEx`·`intTest` 는
  "your group is not authorized" 로 거부된다. 네 파티션 모두 GH200 노드이고 MaxTime 은 1일이다.
- `--comment="field=<분야>;appl=<프로그램>"` 를 붙인다(사이트 규정; 값 목록은 `showappl`). 이 코드는 `field=mae;appl=in_house` 를 쓴다.
- **단일 노드 GPU 잡**에서는 `--cpus-per-task`·`--mem`·`-n` 을 쓸 수 없다. `--ntasks-per-node` 와 `--gres=gpu:N` 만 쓴다
  (CPU·메모리는 GPU 수에 비례해 자동 배정). 다른 옵션까지 쓰려면 `--exclusive` 를 붙여야 한다.
- 실행은 `mpirun` 이 아니라 **`srun`** 이다.
- GPU 바인딩은 코드가 직접 한다(`cudaSetDevice(node_rank % n_devices)`). **1 랭크 = 1 GPU** 가 되도록 `--ntasks-per-node` 와 `--gres=gpu:` 를 같게 맞춘다.

### 예: 1 노드 4 GPU

```bash
#!/bin/bash
#SBATCH -J dhvc
#SBATCH -p natStr
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH -o /scratch/t639a05/MPM-STD_C/apps/dhvc/logs/%x_%j.log

module purge
module load craype-arm-grace PrgEnv-nvidia
module swap cray-mpich cray-mpich/9.0.1
module load cuda/12.9_26.3 craype-accel-nvidia90
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1

cd /scratch/t639a05/MPM-STD_C
srun ./build_gpu/bin/dhvc apps/dhvc/input_<케이스>.ini     # 랭크 수 = npx·npy·npz = 4
```

여러 노드를 쓸 때는 `-N` 만 늘리고 `--ntasks-per-node=4 --gres=gpu:4` 는 그대로 둔다. 멀티노드는 아직 돌려 보지 않았다.

- 입력 덱의 `stats_dir`·`checkpoint_dir` 상대 경로는 **덱 파일이 있는 위치 기준**으로 풀린다(스모크에서 확인).
- `apps/*/sbatch_*.sh` 는 전부 구 클러스터용이다(`-p batch`, `-w gpu01,gpu02`, `/shared/home/...`, hpcx `mpirun`).
  이 클러스터에서는 위 템플릿으로 다시 써야 한다.

### 검증 잡

`build_gpu/smoke/smoke.sh` 에 이번에 쓴 스모크 잡을 남겨 두었다. 2 GPU 에서 `test_pressure_wall` 와 `dhvc` 를 돌린다.
로그는 `build_gpu/smoke/smoke_1153104.log` 이다.

```bash
sbatch --wait build_gpu/smoke/smoke.sh
```

---

## 5. 함정 (실제로 겪은 것)

① **cray-mpich 9.1.0(기본값) + CUDA 13 조합은 실행되지 않는다.**
  9.1.0 의 GTL(`libmpi_gtl_cuda.so`)은 `libcudart.so.13` 에 링크돼 있다. 그래서 CUDA 12.9 로 링크하면
  `undefined reference to cudaHostRegister@libcudart.so.13` 이 난다. 그렇다고 CUDA 13.2(`nvidia/26.5` + `cuda/13.2_26.5`,
  사이트 예제 `~/job_examples/job_gpu.sh` 의 조합)로 바꾸면 빌드는 되지만, 계산 노드 드라이버(570)가 CUDA 13 을 지원하지 않는다.
  결과는 `cudaMalloc -> CUDA driver version is insufficient for CUDA runtime version` 이다.
  → GTL 이 `libcudart.so.12` 에 링크된 **cray-mpich 9.0.1** 과 CUDA 12.9 를 함께 쓴다(8.1.33 도 cudart 12).

② **`/opt/cray/pe/lib64` 는 항상 9.1.0 을 가리킨다.** 9.0.1 로 빌드해도 실행 때는 9.1.0 의 GTL 이 먼저 잡혀
  `libcudart.so.13 => not found` 가 난다. `export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH` 를 넣으면
  9.0.1 쪽이 먼저 잡힌다(Cray PE 표준 관례). `ldd build_gpu/bin/dhvc | grep -E 'gtl|libmpi'` 결과가 `/opt/cray/pe/mpich/9.0.1/...` 이면 정상이다.

③ **nvc++ 는 시스템 GCC 7.5 의 libstdc++ 헤더를 쓴다**(SDK `localrc` 에 고정). GCC 7 에는 `<filesystem>` 이 없어서
  `catastrophic error: cannot open source file "filesystem"` 이 난다. `--gcc-toolchain=/usr/bin/gcc-14` 로 GCC 14 헤더를 쓰게 한다.
  실행 때 쓰는 `/usr/lib64/libstdc++.so.6` 은 그대로 동작한다.

④ `CC` 래퍼는 `-acc` 도 붙이지만, 코드에 OpenACC 가 없으므로 영향이 없다.

---

## 6. CPU 빌드는?

이번에는 확인하지 않았다. 기본 `make` 도 이 클러스터에서는 동작하지 않는다.
`/usr/bin/mpicxx` 가 없고, FFTW 경로(`/shared/home/wel1come1234/local/fftw3-gcc`)도 없다.
필요하면 `PrgEnv-gnu` + `cray-fftw` 를 쓰고
`CXX=CC EXT_INC=-I$FFTW_INC EXT_LIBS="-L$FFTW_DIR -lfftw3 -lm"` 로 덮어쓰는 방향이 자연스럽다.
CPU 전용 노드(`normal_cpu` 등)는 x86 Turin 이라 **로그인 노드(aarch64)에서 빌드한 바이너리는 거기서 돌지 않는다**는 점에 주의한다.

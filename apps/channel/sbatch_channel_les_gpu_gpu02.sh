#!/bin/bash
# 채널 LES GPU 검증 (2026-10-11, 사용자: GPU 비어 있으면 LES 를 GPU 로) — gpu02 2 GPU, 케이스 순차: les96_vreman012_mirror les96_none_mirror
# 같은 덱의 CPU 런(stats_<case>/)과 통계 비교. gpu02 첫 단계는 결정론 대조(300 스텝, CPU np2 와 모니터 비교용).
#SBATCH -J les_gpu_gpu02
#SBATCH -p batch
#SBATCH -w gpu02
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/channel/log/les_gpu_gpu02_%j.log
set -u
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
cd $ROOT/apps/channel
"$HPCX/ompi/bin/mpirun" -np 2 $ROOT/build_gpu/bin/channel input_channel_re180_les96_vreman012_mirror_gpu.ini > log/les96_vreman012_mirror_gpu.log 2>&1; echo "== les96_vreman012_mirror: $(grep FINAL log/les96_vreman012_mirror_gpu.log)"
"$HPCX/ompi/bin/mpirun" -np 2 $ROOT/build_gpu/bin/channel input_channel_re180_les96_none_mirror_gpu.ini > log/les96_none_mirror_gpu.log 2>&1; echo "== les96_none_mirror: $(grep FINAL log/les96_none_mirror_gpu.log)"

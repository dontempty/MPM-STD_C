#!/bin/bash
# 잡 할당 안에서 작은 진단 런 — srun --jobid=<id> --overlap -n1 bash diag_run_in_alloc.sh
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
run() { local app=$1 deck=$2; echo "=== $app $deck $(date '+%T')"; ( cd $ROOT/apps/$app && $HPCX/ompi/bin/mpirun --oversubscribe -np 2 $ROOT/build_gpu/bin/$app $deck > log/diag_${deck%.ini}.log 2>&1 ); grep -E "FINAL|error \(rank|blew up" $ROOT/apps/$app/log/diag_${deck%.ini}.log | tail -2; }
run rbc input_rbc_diag_const_pr2547.ini
run rbc_nob input_rbc_nob_diag_var_pr2547.ini
run rbc input_rbc_diag_const_pr07.ini
run rbc input_rbc_diag_const_pr2547_dt005.ini
echo "=== diag done $(date '+%T')"

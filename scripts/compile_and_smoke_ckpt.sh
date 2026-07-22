#!/bin/bash
#SBATCH -J ckpt_smoke
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:40:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/ckpt_smoke.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== build ==="
make BACKEND=cuda apps 2>&1 | tail -4
echo "=== smoke: 3000 steps with nrestart_write=1000 ==="
mpirun -np 1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_verify_ckpt.toml 2>&1 | grep -E "restart|step=3000|nan|error" 
echo "=== ckpt dirs ==="
ls -la apps/rbc/restart_verify_ckpt/ apps/rbc/restart_verify_ckpt/ckpt_a apps/rbc/restart_verify_ckpt/ckpt_b 2>&1 | grep -E "ckpt|U.bin|stats"
cat apps/rbc/restart_verify_ckpt/ckpt_a/stats.txt apps/rbc/restart_verify_ckpt/ckpt_b/stats.txt 2>/dev/null | grep -E "step|tc_n"
echo "=== resume test: restart from ckpt_b (step 2000) for 500 steps ==="
sed -e 's|dir_out = "apps/rbc/restart_verify_ckpt/"|dir_in  = "apps/rbc/restart_verify_ckpt/ckpt_b/"\ndir_out = ""|' \
    -e 's/n_steps  = 3000/n_steps  = 2500/' \
    -e 's/nrestart_write = 1000/nrestart_write = 0/' \
    apps/rbc/input_verify_ckpt.toml > apps/rbc/input_verify_ckpt_resume.toml
mpirun -np 1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_verify_ckpt_resume.toml 2>&1 | grep -E "restart|step=2[0-5]00|nan|error" | head -8
echo "=== done ==="

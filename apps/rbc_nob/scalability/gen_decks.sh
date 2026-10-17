#!/bin/bash
# 확장성 벤치마크 덱 생성 — docs/Scalability/plan.md §3·§4.
#   ./gen_decks.sh            → input_scal_strong_{256,512}_np{N}.ini, input_scal_weak_{256,512}_np{N}.ini
# 물리는 cv12 ΔT 40 덱(input_rbc_nob_cv12_dt40.ini)과 같고 격자·npz·lz·출력만 다르다.
# strong: 전역 n³ 고정, npz = N.   weak: 랭크당 n³ 고정 → 전역 nz = n·N, lz = N (셀 종횡비 유지).
# np 상한 8 = 1노드 (다노드 잡은 이 큐에서 돌지 않는다 — 사용자 결정 2026-10-17).
# 200 스텝, 워밍업 50 제외(phase_timing_start_step 50), 통계·체크포인트 끔, 모니터 50 스텝.
set -eu
cd "$(dirname "$0")"
emit() {  # mode n np nz lz
  local mode=$1 n=$2 np=$3 nz=$4 lz=$5
  local name="scal_${mode}_${n}_np${np}"
  cat > "input_${name}.ini" <<INI
; 확장성 벤치마크 ${mode} — 전역 ${n}x${n}x${nz}, npz ${np} (gen_decks.sh 생성; docs/Scalability/plan.md)
[mesh]
nx = ${n}
ny = ${n}
nz = ${nz}

[domain]
lx = 6.0
ly = 6.0
lz = ${lz}

[stretch]
uniform_z = false
gamma_z = 2.6

[process]
npz = ${np}

[physics]
rayleigh = 1.0e6
prandtl = 2547.0
delta_temperature = 40.0

[initial]
perturb = 0.05

[time]
end_time = 1.0e9
max_step = 200
dt_max = 0.05
max_cfl = 0.3
viscous_coupling_number = 12

[output]
stats_dir = "out/${name}/"
monitor_interval = 50
stats_start_time = 1.0e9
checkpoint_interval = 0
phase_timing_start_step = 50
INI
}
for np in 1 2 4 8;    do emit strong 256 $np 256 1.0; done
for np in 1 2 4 8;    do emit strong 512 $np 512 1.0; done
for np in 1 2 4 8;    do emit weak 256 $np $((256*np)) ${np}.0; done
for np in 1 2 4 8;          do emit weak 512 $np $((512*np)) ${np}.0; done
ls input_scal_*.ini | wc -l

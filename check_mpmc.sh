#!/bin/bash
# 클러스터의 노드별 파티션과 남은 CPU / GPU 개수를 터미널에 바로 출력한다.
#   ./check_mpmc.sh            전체 노드
#   ./check_mpmc.sh gpu        이름 또는 파티션에 'gpu' 가 들어간 노드만
#   ./check_mpmc.sh a100       파티션 이름으로도 거를 수 있다 (amd_a100nv_8 등)
#
# 셈법: scontrol show node 의 CfgTRES(보유) − AllocTRES(할당) = 남은 자원.
# sinfo 의 CPUS(A/I/O/T) 와 같은 수치지만 GPU(gres)까지 한 표로 본다.

filter="${1:-}"

scontrol show node --oneliner 2>/dev/null | awk -v filter="$filter" '
function tres_gpu(s,   m) {           # "...,gres/gpu=2,..." 에서 2 를 뽑는다
  m = match(s, /gres\/gpu=[0-9]+/)
  return m ? substr(s, RSTART + 9, RLENGTH - 9) + 0 : 0
}
{
  node = ""; part = "-"; state = ""; cpu_total = 0; cpu_alloc = 0; cfg = ""; alloc = ""
  for (f = 1; f <= NF; ++f) {
    if      ($f ~ /^NodeName=/)  { node = substr($f, 10) }
    else if ($f ~ /^Partitions=/) { part = substr($f, 12) }
    else if ($f ~ /^State=/)     { state = substr($f, 7) }
    else if ($f ~ /^CPUTot=/)    { cpu_total = substr($f, 8) + 0 }
    else if ($f ~ /^CPUAlloc=/)  { cpu_alloc = substr($f, 10) + 0 }
    else if ($f ~ /^CfgTRES=/)   { cfg = $f }
    else if ($f ~ /^AllocTRES=/) { alloc = $f }
  }
  if (filter != "" && index(node, filter) == 0 && index(part, filter) == 0) next

  gpu_total = tres_gpu(cfg)
  gpu_alloc = tres_gpu(alloc)
  cpu_free = cpu_total - cpu_alloc
  gpu_free = gpu_total - gpu_alloc

  if (NR == 1 || !header_printed) {
    printf "%-8s %-16s %14s %14s   %s\n", "node", "partition", "CPU free/tot", "GPU free/tot", "state"
    header_printed = 1
  }
  gpu_text = (gpu_total > 0) ? sprintf("%d/%d", gpu_free, gpu_total) : "-"
  printf "%-8s %-16s %14s %14s   %s\n", node, part, sprintf("%d/%d", cpu_free, cpu_total), gpu_text, state

  sum_cpu_free += cpu_free; sum_cpu_total += cpu_total
  sum_gpu_free += gpu_free; sum_gpu_total += gpu_total
}
END {
  if (header_printed) {
    printf "%-8s %-16s %14s %14s\n", "TOTAL", "",
           sprintf("%d/%d", sum_cpu_free, sum_cpu_total),
           sprintf("%d/%d", sum_gpu_free, sum_gpu_total)
  } else {
    print "노드 정보를 읽지 못했다 — Slurm(scontrol) 이 있는 곳에서 실행할 것"
    exit 1
  }
}'

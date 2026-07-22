#!/bin/bash
# Queries sinfo/squeue/scontrol directly on the login node (no compute
# allocation needed) so the result prints immediately.

echo "===== $(date) on $(hostname) ====="
echo

for P in $(sinfo -h -o "%P" | sort -u | sed 's/\*$//'); do
    echo "############################################"
    echo "# Partition: $P"
    echo "############################################"

    echo "-- node summary --"
    sinfo -p "$P" -N -o "%12N %10T %12C %10m %15G" -h \
        | awk 'BEGIN{printf "%-12s %-10s %-18s %-10s %-15s\n","NODE","STATE","CPU(A/I/O/T)","MEM(MB)","GRES"}
               {printf "%-12s %-10s %-18s %-10s %-15s\n",$1,$2,$3,$4,$5}'

    echo
    echo "-- per-node free resources (scontrol) --"
    printf "%-10s %-10s %10s %12s %10s\n" "NODE" "STATE" "CPU(used/tot)" "MEM_FREE(MB)" "GPU(used/tot)"
    for N in $(sinfo -p "$P" -N -h -o "%N" | sort -u); do
        INFO=$(scontrol show node "$N")
        STATE=$(echo "$INFO" | grep -oP 'State=\K\S+')
        CPU_TOT=$(echo "$INFO" | grep -oP 'CPUTot=\K[0-9]+')
        CPU_ALLOC=$(echo "$INFO" | grep -oP 'CPUAlloc=\K[0-9]+')
        MEM_FREE=$(echo "$INFO" | grep -oP 'FreeMem=\K[0-9]+')
        GPU_TOT=$(echo "$INFO" | grep -oP 'CfgTRES=\S*gres/gpu=\K[0-9]+')
        GPU_ALLOC=$(echo "$INFO" | grep -oP 'AllocTRES=\S*gres/gpu=\K[0-9]+')
        [ -z "$GPU_TOT" ] && GPU_TOT=0
        [ -z "$GPU_ALLOC" ] && GPU_ALLOC=0
        printf "%-10s %-10s %10s %12s %10s\n" "$N" "$STATE" "${CPU_ALLOC}/${CPU_TOT}" "${MEM_FREE:-N/A}" "${GPU_ALLOC}/${GPU_TOT}"
    done

    echo
    echo "-- pending jobs (queue) --"
    squeue -p "$P" -h -t PD -o "%.10i %.10u %.8j %.6D %R" | head -20
    echo
done

echo "===== done ====="

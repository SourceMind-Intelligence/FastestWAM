#!/usr/bin/env bash
# Periodic GPU / host snapshots for scaling plots (CSV, no secrets).
set -euo pipefail
OUT="${1:?usage: monitor_gpus.sh <csv_path>}"
INTERVAL="${2:-10}"
mkdir -p "$(dirname "$OUT")"
if [[ ! -f "$OUT" ]]; then
    echo "timestamp,gpu_index,util_gpu,mem_used_mib,mem_total_mib,power_w,cpu_load1,ram_used_gib,ram_total_gib" >"$OUT"
fi
while true; do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    load1="$(cut -d' ' -f1 /proc/loadavg)"
    # MemTotal / MemAvailable in kB
    ram="$(awk '/MemTotal:/{t=$2} /MemAvailable:/{a=$2} END{printf "%.1f,%.1f", (t-a)/1048576, t/1048576}' /proc/meminfo)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits \
        | while IFS=',' read -r idx util mem tot pwr; do
            echo "${ts},${idx// /},${util// /},${mem// /},${tot// /},${pwr// /},${load1},${ram}"
        done >>"$OUT"
    sleep "${INTERVAL}"
done

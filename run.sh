#!/usr/bin/env bash
set -euo pipefail

cd /workspace/HOV-SG

SCENES=(
    "00824-Dd4bFSTQ8gi"
    "00829-QaLdnwvtxbs"
    "00843-DYehNKdT76V"
    "00847-bCPU9suPUw9"
    "00849-a8BtkwhxdRV"
    "00861-GLAQ4DNUx5U"
    "00873-bxsVRursffK"
    "00877-4ok3usBNeis"
    "00890-6s7QHgap2fW"
)

DATASET_PATH=/workspace/HOV-SG/data/hm3dsem_walks
SAVE_PATH=/workspace/HOV-SG/data/scene_graphs
LOG_DIR=/workspace/HOV-SG/logs/create_graph
mkdir -p "$LOG_DIR"

for scene_id in "${SCENES[@]}"; do
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting scene: ${scene_id} ==="
    log_file="${LOG_DIR}/${scene_id}.log"

    if python application/create_graph.py \
        main.dataset=hm3dsem \
        main.dataset_path="${DATASET_PATH}" \
        main.split=val \
        main.scene_id="${scene_id}" \
        main.save_path="${SAVE_PATH}" \
        2>&1 | tee "${log_file}"; then
        echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished scene: ${scene_id} ==="
    else
        echo "!!! [$(date '+%Y-%m-%d %H:%M:%S')] FAILED scene: ${scene_id} — see ${log_file} — continuing to next scene"
    fi
done

echo "All scenes processed."

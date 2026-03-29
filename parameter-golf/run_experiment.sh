#!/bin/bash
# run_experiment.sh — Run parameter-golf training on a single H100
# Usage: bash run_experiment.sh [baseline|cnn_hybrid]
#
# This script:
# 1. Installs dependencies
# 2. Downloads the FineWeb dataset (sp1024 variant, 10 shards for speed)
# 3. Runs the training script with a wallclock cap
# 4. Copies outputs to /workspace/outputs/

set -e

EXPERIMENT="${1:-baseline}"
echo "============================================================"
echo "Parameter Golf Experiment: $EXPERIMENT"
echo "============================================================"

cd /workspace/code/parameter-golf

# Install dependencies
echo "[1/4] Installing dependencies..."
pip install -q sentencepiece huggingface-hub datasets tqdm numpy 2>&1 | tail -3

# Download dataset (10 train shards = ~1B tokens, enough for a demo)
echo "[2/4] Downloading FineWeb dataset..."
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10 2>&1 | tail -5

# Prepare output directory
mkdir -p /workspace/outputs

# Choose training script
if [ "$EXPERIMENT" = "baseline" ]; then
    SCRIPT="records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py"
elif [ "$EXPERIMENT" = "cnn_hybrid" ]; then
    SCRIPT="records/cnn_hybrid_experiment/train_gpt.py"
else
    echo "Unknown experiment: $EXPERIMENT"
    exit 1
fi

echo "[3/4] Running training: $SCRIPT"
echo "  Wallclock cap: 1800s (30 min, single GPU)"
echo "  Iterations: 3000 (reduced for 1-GPU demo)"

# Run single-GPU training with reduced iterations and 30-min wallclock cap
# The challenge uses 8xH100 for 10 min; we use 1xH100 so we give it more time
# but cap at 30 min to stay safe under the 40-min pod kill limit.
MAX_WALLCLOCK_SECONDS=1800 \
ITERATIONS=3000 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
TRAIN_BATCH_TOKENS=65536 \
WARMUP_STEPS=5 \
python3 "$SCRIPT" 2>&1 | tee /workspace/outputs/${EXPERIMENT}_train.log

# Copy model artifacts to outputs
echo "[4/4] Collecting outputs..."
cp -f final_model.pt /workspace/outputs/${EXPERIMENT}_model.pt 2>/dev/null || true
cp -f final_model.int8.ptz /workspace/outputs/${EXPERIMENT}_model.int8.ptz 2>/dev/null || true

echo "============================================================"
echo "Experiment $EXPERIMENT complete!"
echo "============================================================"

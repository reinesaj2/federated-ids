#!/bin/bash

# Sprint 1: Baseline with PerDatasetEncoderNet for Edge-IIoTset
# Expected: Macro-F1 improves from 55% to 75-85%

export DATASET="edge-iiotset-full"
export MODEL_ARCH="auto"
export NUM_CLIENTS=10
export ALPHA=0.5
export NUM_ROUNDS=15
export LOCAL_EPOCHS=1
export LR=0.001
export SEED=42

python server.py \
    --dataset "$DATASET" \
    --model_arch "$MODEL_ARCH" \
    --aggregation_method "fedavg" \
    --num_clients $NUM_CLIENTS \
    --alpha $ALPHA \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --lr $LR \
    --seed $SEED \
    --run_id "sprint1_baseline_$(date +%Y%m%d_%H%M%S)"

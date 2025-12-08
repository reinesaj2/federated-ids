#!/bin/bash

# Sprint 2: PerDatasetEncoderNet + FocalLoss
# Expected: Macro-F1 improves to 85-95%

export DATASET="edge-iiotset-full"
export MODEL_ARCH="auto"
export NUM_CLIENTS=10
export ALPHA=0.5
export NUM_ROUNDS=15
export LOCAL_EPOCHS=2
export LR=0.001
export SEED=42
export USE_FOCAL_LOSS=1
export FOCAL_GAMMA=2.0

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
    --use_focal_loss \
    --focal_gamma $FOCAL_GAMMA \
    --run_id "sprint2_focal_$(date +%Y%m%d_%H%M%S)"

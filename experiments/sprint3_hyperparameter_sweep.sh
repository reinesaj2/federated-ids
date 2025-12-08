#!/bin/bash

# Sprint 3: Hyperparameter Tuning for >90% Macro-F1
# Grid search: LR x LOCAL_EPOCHS x FOCAL_GAMMA

DATASET="edge-iiotset-full"
MODEL_ARCH="auto"
NUM_CLIENTS=10
ALPHA=0.5
NUM_ROUNDS=15
SEED=42

LRS=(0.0005 0.001 0.002)
LOCAL_EPOCHS_LIST=(1 2 3)
FOCAL_GAMMAS=(1.0 2.0 3.0)

for lr in "${LRS[@]}"; do
    for epochs in "${LOCAL_EPOCHS_LIST[@]}"; do
        for gamma in "${FOCAL_GAMMAS[@]}"; do
            echo "========================================"
            echo "Testing: LR=$lr, Epochs=$epochs, Gamma=$gamma"
            echo "========================================"

            python server.py \
                --dataset "$DATASET" \
                --model_arch "$MODEL_ARCH" \
                --aggregation_method "fedavg" \
                --num_clients $NUM_CLIENTS \
                --alpha $ALPHA \
                --num_rounds $NUM_ROUNDS \
                --local_epochs $epochs \
                --lr $lr \
                --seed $SEED \
                --use_focal_loss \
                --focal_gamma $gamma \
                --run_id "sprint3_lr${lr}_e${epochs}_g${gamma}_$(date +%Y%m%d_%H%M%S)"

            echo ""
            echo "Completed: LR=$lr, Epochs=$epochs, Gamma=$gamma"
            echo ""
        done
    done
done

echo "Hyperparameter sweep complete!"
echo "Analyze results in runs/ directory"

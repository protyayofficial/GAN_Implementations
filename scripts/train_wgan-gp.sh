#!/bin/bash

# Make sure the script exits if any command fails
set -e

# Set up the training parameters
BATCH_SIZE=64
G_LR=1e-4
C_LR=1e-4
EPOCHS=100
IMG_SIZE=64
FEATURES_C=64
FEATURES_G=64
IMG_CHANNELS=1
SAMPLE_SIZE=100
CRITIC_ITER=5
LMBDA=10
SEED=9
OUTPUT_DIR="results/WGAN-GP"

# Run the training script with the specified arguments
python3 models/WGAN-GP.py \
    --batch_size $BATCH_SIZE \
    --g_lr $G_LR \
    --c_lr $C_LR \
    --epochs $EPOCHS \
    --img_size $IMG_SIZE \
    --features_c $FEATURES_C \
    --features_g $FEATURES_G \
    --img_channels $IMG_CHANNELS \
    --sample_size $SAMPLE_SIZE \
    --critic_iter $CRITIC_ITER \
    --lmbda $LMBDA \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

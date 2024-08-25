#!/bin/bash

# Make sure the script exits if any command fails
set -e

# Set up the training parameters
BATCH_SIZE=128
G_LR=0.0002
D_LR=0.0002
EPOCHS=100
IMG_SIZE=64
FEATURES_D=64
FEATURES_G=64
IMG_CHANNELS=1
SAMPLE_SIZE=100
SEED=9
OUTPUT_DIR="results/DCGAN"

# Run the training script with the specified arguments
python3 models/DCGAN.py \
    --batch_size $BATCH_SIZE \
    --g_lr $G_LR \
    --d_lr $D_LR \
    --epochs $EPOCHS \
    --img_size $IMG_SIZE \
    --features_d $FEATURES_D \
    --features_g $FEATURES_G \
    --img_channels $IMG_CHANNELS \
    --sample_size $SAMPLE_SIZE \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

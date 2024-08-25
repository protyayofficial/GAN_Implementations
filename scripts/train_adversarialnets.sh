#!/bin/bash

# Bash script to run the GAN training script with specified arguments

# Default parameters
BATCH_SIZE=64
GENERATOR_LR=0.0001
DISCRIMINATOR_LR=0.0001
EPOCHS=100
SAMPLE_SIZE=100
SEED=9
OUTPUT_DIR='results/AdversarialNets'

# Execute the training script with arguments
python3 models/AdversarialNets.py \
    --batch_size $BATCH_SIZE \
    --g_lr $GENERATOR_LR \
    --d_lr $DISCRIMINATOR_LR \
    --epochs $EPOCHS \
    --sample_size $SAMPLE_SIZE \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

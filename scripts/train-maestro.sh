#!/bin/bash
# Adjust devices and hyperparameters for the available training environment.

uv run maestro-train \
    --project maestro-allof \
    --devices 0 \
    --data_dirs data/processed/cyto-diffusion-bridge/allof/allof-allof \
    --number_cells_subset 40000 \
    --num_outputs 5000 \
    --num_inds 16 \
    --dim_hidden 384 \
    --dim_latent 256 \
    --num_heads 1 \
    --epochs 1000 \
    --mode Train \
    --student_temperature 0.11 \
    --teacher_temperature 0.04 \
    --center_momentum 0.99 \
    --teacher_beta 0.9995

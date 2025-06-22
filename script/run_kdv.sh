#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

DATA_PATH="./data/kdv_nx256_t0.1_IMEXRK2.h5"

echo "Starting experiment run..."
echo "Using dataset: $DATA_PATH"
echo "GPU Device: $CUDA_VISIBLE_DEVICES"
echo "-------------------------------------"

echo "Running experiment for FNO..."
python train_eval.py --model_type fno --data_path ${DATA_PATH}
echo "-------------------------------------"

echo "Running experiment for SNO (Chebyshev)..."
python train_eval.py --model_type sno_chebyshev --data_path ${DATA_PATH}
echo "-------------------------------------"

echo "Running experiment for SNO (Fourier)..."
python train_eval.py --model_type sno_fourier --data_path ${DATA_PATH}
echo "-------------------------------------"

echo "All experiments completed."
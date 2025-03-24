#!/bin/bash

# source /irip/zhangjinjin_2023/anaconda3/etc/profile.d/conda.sh
# conda activate diffusion-4k

export INSTANCE_DIR="./Aesthetic-4K/train"
export OUTPUT_DIR="./checkpoint/flux_wavelet"

export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=8000000
# export TORCH_NCCL_ENABLE_MONITORING=0
export MODEL_NAME="./pretrain/FLUX.1-dev"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --config_file ds_config.yaml train_flux_4k_wavelet.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --resolution=4096  \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=20000 \
  --checkpointing_steps=5000 \
  --seed="0" \
  # --resume_from_checkpoint="latest"
  # --max_sequence_length=512

# srun --gres=gpu:a800:8 -c 64 --mem=512G train_flux.sh
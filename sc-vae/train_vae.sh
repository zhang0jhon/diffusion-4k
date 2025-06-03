#!/bin/bash

# source /irip/zhangjinjin_2023/anaconda3/etc/profile.d/conda.sh
# conda activate diffusion-4k

export INSTANCE_DIR="./data"
export OUTPUT_DIR="./checkpoint/flux_sc_vae_sa1b_512x512"

export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=8000000
# export TORCH_NCCL_ENABLE_MONITORING=0
# export MODEL_NAME="../pretrain/stable-diffusion-3-medium-diffusers"
export MODEL_NAME="../pretrain/FLUX.1-dev"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export TOKENIZERS_PARALLELISM=false

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --config_file config.yaml train_vae.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="no" \
  --image_column="file_name" \
  --resolution=512 \
  --lpips_scale=1e-1 \
  --gen_scale=5e-2 \
  --sc_scale=1 \
  --kl_scale=1e-6 \
  --train_batch_size=32 \
  --num_train_epochs=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --learning_rate=1e-5 \
  --discr_learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --discr_lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed="0" \
  --use_ema \
  --adam_weight_decay=1e-4 \
  --checkpointing_steps=10000 


# # srun --gres=gpu:a800:4 -c 32 --mem=160G train_vae.sh
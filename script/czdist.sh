#!/bin/bash
#PROCESS_NUM=8
#
#CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=2 \
#  train_dist.py \
#  --root "" \
#  --batch-size 96 \
#  --image_size 384 192 \
#  --num-workers 8 \
#  --nce-k 18000 \
#  --neck NonLinearNeckV1 \
#  --model swin_resnet \
#  --alpha 0.996 \
#  --with_ibn True \
#  --aug v1 \
#  --memory_func MemoryMoCo \
#  --nce-t 0.07 \
#  --epochs 600 \
#  --output-dir "./output" \
#  --base-lr 0.1

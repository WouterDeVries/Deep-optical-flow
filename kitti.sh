#!/usr/bin/env bash
set -x

python3 main.py \
    --maxfx 512 \
    --maxfy 512 \
    --fea_c 32 24 24 16 16 \
\
    --dataset kitti \
    --datapath datasets/KITTI/ \
    --trainlist ./filenames/kitti_train.txt \
    --testlist ./filenames/shuffled_kitti12_val.txt \
\
    --lr 0.0004 \
    --batch_size 1 \
    --test_batch_size 1 \
    --epochs 700 \
    --lrepochs "500,600:8,5" \
    --ckpt_start_epoch 0 \
\
    --logdir logs/ \
    --seed 1 \
    --save_freq 1 \
\
    --loadfeat logs/experiment_0/checkpoint_004709.ckpt \
#   --resume logs/experiment_0/checkpoint_000001.ckpt \
#   --loadckpt logs/experiment_0/checkpoint_000001.ckpt \
#   --test_img img/current_model/ \
#   --init_flow_gt \
#   --init_slant_gt \

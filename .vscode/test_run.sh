#!/bin/bash
#
# 测试程序能否运行，在项目根目录下运行该脚本

set -e # 严格模式，任何命令的失败都将导致脚本退出
cd tools 

# ST3D
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:43210 train.py --launcher pytorch \
    --cfg_file cfgs/kitti_models/secondiou_orcale.yaml --debug --epochs 3 --batch_size 4
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:43210 train.py --launcher pytorch \
    --cfg_file cfgs/nuscenes_models/secondiou_car_oracle.yaml --debug --epochs 3 --batch_size 4

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:43210 train.py --launcher pytorch \
    --cfg_file cfgs/da-nuscenes-kitti_models/secondiou/secondiou_old_anchor.yaml --debug --epochs 3 --batch_size 4
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:43210 train.py --launcher pytorch \
    --cfg_file cfgs/da-nuscenes-kitti_models/secondiou/secondiou_old_anchor_ros.yaml --debug --epochs 3 --batch_size 4

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:43211 train.py --launcher pytorch \
    --cfg_file cfgs/da-nuscenes-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml --debug --epochs 3 --batch_size 4 \
    --pretrained_model ../model_zoo/download/st3d/da-nus-kitti/ros/secondiou/ckpt.pth

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:43211 train.py --launcher pytorch \
    --cfg_file cfgs/da-nuscenes-kitti_models/secondiou_st3d/secondiou_st3d++_ros_car.yaml --debug --epochs 3 --batch_size 4 \
    --pretrained_model ../model_zoo/download/st3d/da-nus-kitti/ros/secondiou/ckpt.pth

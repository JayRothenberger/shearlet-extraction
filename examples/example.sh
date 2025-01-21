#!/bin/bash

conda activate shearlets
wandb login 6ac799cb76304b17ce74f5161bc27f7a80b6ecee

torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "localhost:64425" \
example.py --dataset_path ./
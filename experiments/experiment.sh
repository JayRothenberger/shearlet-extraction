torchrun --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=1 \
        experiment.py \
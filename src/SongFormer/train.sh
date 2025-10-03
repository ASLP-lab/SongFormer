export CUDA_VISIBLE_DEVICES=0

export WANDB_API_KEY="Please paste your secret key here"
export PYTHONPATH=$(realpath .):$PYTHONPATH

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# export TORCH_LOGS=attention
# export WANDB_MODE=disabled

gpustat --id $CUDA_VISIBLE_DEVICES

yaml_names=(
    SongFormer
)

init_seed_lists=(
    "42"
    # "8988"
    # "1331"
)

for init_seed in "${init_seed_lists[@]}"; do
    for yaml_name in "${yaml_names[@]}"; do
        accelerate launch --config_file train/accelerate_config/single_gpu.yaml \
            train/train.py \
            --config "configs/${yaml_name}.yaml" \
            --log_interval 5 \
            --init_seed ${init_seed}
    done
done

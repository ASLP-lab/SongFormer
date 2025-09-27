export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=${PWD}:$PYTHONPATH

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


EST_DIR=
ANN_DIR=
OUTPUT_DIR=
echo "$EST_DIR --> $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

python evaluation/eval_infer_results.py \
    --ann_dir $ANN_DIR \
    --est_dir $EST_DIR \
    --output_dir $OUTPUT_DIR \
    --prechorus2what verse
    # --armerge_continuous_segments
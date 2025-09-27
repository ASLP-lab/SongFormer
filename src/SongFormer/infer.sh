
export CUDA_VISIBLE_DEVICES=
echo "use gpu ${CUDA_VISIBLE_DEVICES}"

export PYTHONPATH=../third_party:$PYTHONPATH

export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python infer/infer.py \
-i XXX.scp \
-o XXX_dir \
--model SongFormer \
--checkpoint SongFormer.pt \
--config_path SongFormer.yaml \
-gn 1 \
-tn 1
# --debug
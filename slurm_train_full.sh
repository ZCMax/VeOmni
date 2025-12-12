#!/bin/bash
#SBATCH -J veomni_train
#SBATCH -p efm_t
#SBATCH -N 4
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00 

wandb login 9e37f762624801dfc332b03f2ecefbf87153ed8f # chenming's wandb

export HF_HOME=/mnt/petrelfs/zhuchenming
export WANDB_DIR=/mnt/inspurfs/efm_t/zhuchenming
export TRITON_CACHE_DIR=/tmp/zhuchenming/.triton
export TOKENIZERS_PARALLELISM=false
MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))

srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    tasks/omni/train_qwen2_vl_intern.py configs/multimodal/qwen2_vl/qwen2_vl.yaml \
    --model.model_path /mnt/inspurfs/efm_t/maoxiaohan/LLM/Qwen2.5-VL-7B-Instruct \
    --data.train_path configs/multimodal/data/internvl3_5_tiny_data_new.yaml \
    --train.global_batch_size 32 \
    --train.save_steps 2000 \
    --train.output_dir results/Qwen2.5-VL-7B-InternVL-Tiny-SFT-Full \
    --train.ulysses_parallel_size 4 \
    --train.data_parallel_replicate_size 4 \
    --train.use_wandb False 2>&1 | tee log_full.txt
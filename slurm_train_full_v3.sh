sbatch -J veomni_v2 \
    --preempt \
    -p efm_t \
    --nodes=4 \
    --ntasks-per-node=1 \
    --gres=gpu:8 \
    --quotatype=reserved \
    --cpus-per-task=128 \
    slurm/run.sh tasks/omni/train_qwen2_vl_intern.py configs/multimodal/qwen2_vl/qwen2_vl.yaml \
    --model.model_path /mnt/inspurfs/efm_t/maoxiaohan/LLM/Qwen2.5-VL-7B-Instruct \
    --data.train_path configs/multimodal/data/internvl3_5_tiny_data_new.yaml \
    --data.max_seq_len 60000 \
    --train.global_batch_size 32 \
    --train.save_steps 2000 \
    --train.output_dir results/Qwen2.5-VL-7B-InternVL-Tiny-SFT-full-v3 \
    --train.ulysses_parallel_size 4 \
    --train.data_parallel_replicate_size 4 \
    2>&1 | tee log_full_v3.txt
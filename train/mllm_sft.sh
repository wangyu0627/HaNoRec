#!/bin/bash
source /home/share/anaconda3/bin/activate llama_factory
export PYTHONPATH="/data/wy/MLLMRec/LLaMA-Factory:$PYTHONPATH"

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/wy/MLLMRec/Qwen-2.5-VL-3B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir /data/wy/MLLMRec/LLaMA-Factory/data \
    --dataset microlens_vl_train_sft_1 \
    --cutoff_len 4096 \
    --learning_rate 1e-04 \
    --num_train_epochs 5.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 50 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir /data/wy/MLLMRec/checkpoints/microlens/sft_1_lora_qwenvl \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target all
#!/bin/bash

source /home/share/anaconda/bin/activate llama_factory
export PYTHONPATH="/data/wy/MLLMRec/LLaMA-Factory:$PYTHONPATH"

llamafactory-cli train \
    --stage sft \
    --model_name_or_path /data/wy/MLLMRec/Qwen-2.5-VL-3B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bnb \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir /data/wy/MLLMRec/LLaMA-Factory/data \
    --eval_dataset microlens_vl_test_sft_3 \
    --cutoff_len 4096 \
    --max_samples 3000 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate True \
    --max_new_tokens 512 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir /data/wy/MLLMRec/results/microlens/mllm_3 \
    --trust_remote_code True \
    --do_predict True \
    --adapter_name_or_path /data/wy/MLLMRec/checkpoints/microlens/sft_3_lora_qwenvl
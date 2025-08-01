import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora(base_model_name, sft_lora_path, save_path):
    print(f"🔵 Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print(f"🟢 Loading SFT LoRA from: {sft_lora_path}")
    model = PeftModel.from_pretrained(model, sft_lora_path, is_trainable=False)

    print("🧩 Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"💾 Saving merged model to: {save_path}")
    model.save_pretrained(save_path, safe_serialization=False)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)

    print("✅ Merge and save completed.")

if __name__ == "__main__":
    # 👉 根据你的实际路径调整以下参数
    base_model_name = "/data/wy/MLLMRec/Qwen-2.5-3B-Instruct"  # or absolute path to HF model
    sft_lora_path = "/data/wy/MLLMRec/checkpoints/netflix/sft_3_lora_qwen"
    save_path = "/data/wy/MLLMRec/Qwen-2.5-3B-Instruct-SFT-Netflix_3"

    merge_lora(base_model_name, sft_lora_path, save_path)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import Dataset as HFDataset
from transformers import GenerationConfig
import os

class DPOTrainerWrapper:
    def __init__(self,
                 model_name,
                 output_dir,
                 lora_path,
                 train_file,
                 val_file,
                 train_sample_size=8192,
                 seed=2025,
                 max_seq_length=512):
        self.model_name = model_name
        self.output_dir = output_dir
        self.lora_path = lora_path
        self.train_file = train_file
        self.val_file = val_file
        self.train_sample_size = train_sample_size
        self.seed = seed
        self.max_seq_length = max_seq_length

        self._prepare_model()
        self._prepare_trainer()

    def _prepare_model(self):
        print("🔵 Loading model and tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.tokenizer.pad_token = self.tokenizer.eos_token = "<|endoftext|>"

        # 加载基础模型（非量化，使用 bf16）
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        )

        self.peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            target_modules=["q_proj", "v_proj"]
        )

        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()  # 可选：打印 LoRA 层参数数量

        # 加载参考模型（冻结 LoRA，或直接加载 base）
        self.model_ref = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        )
        self.model_ref.eval()

    def _prepare_trainer(self):
        print("🟠 Preparing DPO trainer...")

        hf_train_dataset = HFDataset.from_list(self.train_file)
        hf_val_dataset = HFDataset.from_list(self.val_file)

        self.training_args = DPOConfig(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            warmup_steps=20,
            num_train_epochs=5,
            learning_rate=1e-4,
            bf16=True,
            logging_steps=50,
            optim="adamw_torch",
            save_strategy="epoch",
            eval_strategy="epoch",
            output_dir=self.output_dir,
            save_total_limit=20,
            max_length=self.max_seq_length,
            beta=0.1,
            label_names=["chosen", "rejected"],
            report_to="none",
            ddp_find_unused_parameters=False,
            max_prompt_length=512,
            loss_type="sigmoid"
            
        )

        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.model_ref,
            args=self.training_args,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_val_dataset,
            processing_class=self.tokenizer,
        )

    def train(self):
        print("🟢 Starting DPO training...")
        self.trainer.train()
        self.trainer.save_model()
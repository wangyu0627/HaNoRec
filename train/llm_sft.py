import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import os
from datasets import Dataset as HFDataset

class SFTTrainerWrapper:
    def __init__(self,
                 model_name,
                 output_dir,
                 train_file,
                 val_file,
                 train_sample_size=8192,
                 seed=2025,
                 max_seq_length=512):
        self.model_name = model_name
        self.output_dir = output_dir
        self.train_file = train_file
        self.val_file = val_file
        self.train_sample_size = train_sample_size
        self.seed = seed
        self.max_seq_length = max_seq_length

        self._prepare_model()
        self._prepare_trainer()

    def _prepare_model(self):
        print("🔵 Loading model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            # target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()

    def _prepare_trainer(self):
        print("🟠 Preparing trainer...")

        hf_train_dataset = HFDataset.from_list(self.train_file)
        hf_val_dataset = HFDataset.from_list(self.val_file)

        self.training_args = SFTConfig(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=8,
            warmup_steps=20,
            num_train_epochs=20,
            learning_rate=1e-4,
            bf16=True,
            logging_steps=50,
            optim="adamw_torch",
            save_strategy="epoch",
            eval_strategy="epoch",
            output_dir=self.output_dir,
            dataset_text_field="text",
            save_total_limit=20,
            load_best_model_at_end=True,
            max_seq_length=self.max_seq_length,
            ddp_find_unused_parameters=False,
            report_to="none",
            label_names=["labels"],
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_val_dataset,
        )

    def train(self):
        print("🟢 Starting training...")
        self.trainer.train()
        self.trainer.save_model()


import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from datasets import Dataset as HFDataset
import os
from tqdm import tqdm
import json
import random
import math
from sklearn.metrics import roc_auc_score

# 🔥 peft UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="peft")

class SFTInferWrapper:
    def __init__(self,
                 model_name,
                 lora_path,
                 test_file,
                 max_seq_length=2048,
                 max_new_tokens=1024,
                 test_batch=None,
                 hit=1):
        self.model_name = model_name
        self.lora_path = lora_path
        self.test_file = test_file
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.test_batch = test_batch
        self.hit = hit

        self._prepare_model()
        self._prepare_dataset()

    def _prepare_model(self):
        print("🔵 Loading base model and LoRA weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # quantization_config=self.bnb_config
        )

        self.model = PeftModel.from_pretrained(self.model, self.lora_path)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _prepare_dataset(self):
        self.dataset = HFDataset.from_list(list(self.test_file))

    def extract_ground_truth(self, prompt_text):
        if "### Output:" in prompt_text:
            return prompt_text.split("### Output:")[1].strip()
        else:
            return None

    def extract_prediction(self, prediction_text):
        if "### Output:" in prediction_text:
            return prediction_text.split("### Output:")[1].strip()
        else:
            return prediction_text.strip()  

    def evaluate_auc(self, outputs):
        y_true = []
        y_score = []

        for item in outputs:
            gt = item["ground_truth"].strip().lower()
            pred = item["prediction"].strip().lower()

            if gt not in {"yes", "no"} or pred not in {"yes", "no"}:
                continue

            # 将 yes 视为正例 (1)，no 视为负例 (0)
            label = 1 if gt == "yes" else 0
            score = 1 if pred == "yes" else 0

            y_true.append(label)
            y_score.append(score)

        auc = roc_auc_score(y_true, y_score) if y_true else 0
        print(f"🎯 AUC: {auc:.4f}")
        return {"AUC": auc}

    def evaluate_metrics(self, outputs, k=3):
        hit = 0
        ndcg = 0
        total = 0

        for item in outputs:
            gt = item["ground_truth"]
            pred = item["prediction"]

            if gt is None or pred is None:
                continue

            gt_list = [s.strip('" ') for s in gt.strip().split('\n') if s.strip()]
            pred_list = [s.strip('" ') for s in pred.strip().split('\n') if s.strip()]
            pred_topk = pred_list[:k]

            if not gt_list:
                continue
            true_item = gt_list[0]

            # HR@K
            hit += int(true_item in pred_topk)

            # NDCG@K
            dcg = 0.0
            for i, p in enumerate(pred_topk):
                if p == true_item:
                    dcg = 1.0 / math.log2(i + 2)
                    break
            idcg = 1.0
            ndcg += dcg / idcg

            total += 1

        metrics = {
            "HR": hit / total if total else 0,
            "NDCG": ndcg / total if total else 0
        }

        print(f"🎯 HR@{k}: {metrics['HR']:.4f}, NDCG@{k}: {metrics['NDCG']:.4f}")
        return metrics

    def infer(self):
        print("🟢 Starting inference...")
        outputs = []

        # 🔥 
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
        )

        # 🔥
        dataset_to_infer = self.dataset
        if self.test_batch is not None:
            n_samples = min(self.test_batch, len(self.dataset))
            indices = random.sample(range(len(self.dataset)), n_samples) 
            dataset_to_infer = self.dataset.select(indices)

        for sample in tqdm(dataset_to_infer, desc="Inferencing"):
            raw_text = sample["text"]
            gt = self.extract_ground_truth(raw_text)
            prompt_text = raw_text.split("### Output:")[0].strip()

            inputs = self.tokenizer(
                prompt_text, return_tensors="pt",
                truncation=True, max_length=self.max_seq_length,
                padding="longest" 
            ).to(self.model.device)

            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],  
                    generation_config=generation_config,
                )

            output_text = self.tokenizer.decode(generation_output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            output_text = self.extract_prediction(output_text)
            outputs.append({
                "ground_truth": gt.strip(),
                "prediction": output_text.strip(),
            })
        # 🔥 
        if self.hit == 1:
            metrics = self.evaluate_auc(outputs)
        elif self.hit ==3:
            metrics = self.evaluate_metrics(outputs, k=self.hit)

        return outputs

    def save(self, outputs, output_path):
        print(f"🟣 Saving outputs to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in outputs:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("✅ Inference completed and saved.")

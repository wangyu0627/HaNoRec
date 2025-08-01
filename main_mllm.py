import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils.data_loader import MultimodalInteractionSFTDataset
import argparse
import subprocess
import json
import math
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT for Microvideo Recommendation")

    parser.add_argument("--do_train", action="store_true", help="是否进行训练")
    parser.add_argument("--do_predict", action="store_true", help="是否进行推理")
    parser.add_argument("--training_mode", choices=["sft", "dpo"], default="sft", help="选择训练模式：sft 或 dpo")

    parser.add_argument("--dataset", type=str, default="microlens", help="指定数据集名称，例如 microlens 或 other_dataset")

    parser.add_argument("--train_path", type=str)
    parser.add_argument("--val_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--title_path", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--save_train_path", type=str)
    parser.add_argument("--save_val_path", type=str)
    parser.add_argument("--save_test_path", type=str)
    parser.add_argument("--template_path", type=str)
    parser.add_argument("--predict_output_path", type=str)
    parser.add_argument("--lora_name", type=str)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--max_item_id", type=int, default=None, help="数据集中物品ID最大值")
    parser.add_argument("--num_negatives", type=int, default=4)
    parser.add_argument("--hit", type=int, default=1)

    parser.add_argument("--model_name", type=str, default="Qwen-2.5-VL-3B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=4096)

    parser.add_argument("--train_sample_size", type=int, default=None, help="limit training samples")
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--test_batch", type=int, default=1000)

    args = parser.parse_args()

    base = args.dataset
    hit = args.hit
    training_mode = args.training_mode
    if args.train_path is None:
        args.train_path = f"data/{base}/train.tsv"
    if args.val_path is None:
        args.val_path = f"data/{base}/val.tsv"
    if args.test_path is None:
        args.test_path = f"data/{base}/test.tsv"
    if args.title_path is None:
        args.title_path = f"data/{base}/{base.capitalize()}_titles.csv"
    if args.image_path is None:
        args.image_path = f"/data/my_name/MLLMRec/data/{base}/images"
    if args.save_train_path is None:
        args.save_train_path = f"data/{base}/train-mllm_{training_mode}_{hit}.json"
    if args.save_val_path is None:
        args.save_val_path = f"data/{base}/val-mllm_{training_mode}_{hit}.json"
    if args.save_test_path is None:
        args.save_test_path = f"data/{base}/test-mllm_{training_mode}_{hit}.json"
    if args.template_path is None:
        args.template_path = f"prompt/{base}.txt"
    if args.predict_output_path is None:
        args.predict_output_path = f"results/{base}/mllm_{hit}/generated_predictions.jsonl"
    if args.lora_name is None:
        args.lora_name = f"checkpoints/{base}/{training_mode}_{hit}_lora_qwen/"
    if args.output_dir is None:
        args.output_dir = f"checkpoints/{base}/{training_mode}_{hit}_lora_qwen/"
    if args.max_item_id is None:
        if base == "microlens":
            args.max_item_id = 19379
        elif base == "netflix":
            args.max_item_id = 17700
        elif base == "movielens":
            args.max_item_id = 3953

    return args

if __name__ == '__main__':
    args = parse_args()

    if args.do_train:

        if args.save_train_path and os.path.exists(args.save_train_path):
            print(f"📂 Loading cached train dataset from {args.save_train_path}...")
            with open(args.save_train_path, 'r', encoding='utf-8') as f:
                train_dataset = json.load(f)
        else:
            print("🧪 Building training dataset...")
            train_dataset = MultimodalInteractionSFTDataset(
                tsv_path=args.train_path,
                title_path=args.title_path,
                image_root=args.image_path,
                max_item_id=args.max_item_id,
                template_path=args.template_path,
                save_path=args.save_train_path,
                num_negatives=args.num_negatives,
                hit=args.hit
            )

        if args.save_val_path and os.path.exists(args.save_val_path):
            print(f"📂 Loading cached val dataset from {args.save_val_path}...")
            with open(args.save_val_path, 'r', encoding='utf-8') as f:
                val_dataset = json.load(f)
        else:
            print("🧪 Building validation dataset...")
            val_dataset = MultimodalInteractionSFTDataset(
                tsv_path=args.val_path,
                title_path=args.title_path,
                image_root=args.image_path,
                max_item_id=args.max_item_id,
                template_path=args.template_path,
                save_path=args.save_val_path,
                num_negatives=args.num_negatives,
                hit=args.hit
            )

        print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
        del train_dataset
        del val_dataset

        print("🟢 Start training...")
        subprocess.run(["bash", "train/mllm_sft.sh"])
        print("🟣 Finished!")

    if args.do_predict:
        print("🧪 Loading test dataset...")
        if args.save_test_path and os.path.exists(args.save_test_path):
            print(f"📂 Loading cached test dataset from {args.save_test_path}...")
            with open(args.save_test_path, 'r', encoding='utf-8') as f:
                test_dataset = json.load(f)
        else:
            test_dataset = MultimodalInteractionSFTDataset(
                tsv_path=args.test_path,
                title_path=args.title_path,
                image_root=args.image_path,
                max_item_id=args.max_item_id,
                template_path=args.template_path,
                save_path=args.save_test_path,
                num_negatives=args.num_negatives,
                hit=args.hit
            )
        print(f"📊 Test {len(test_dataset)}")
        del test_dataset

        print("🟢 Start testing...")
        subprocess.run(["bash", "validation/mllm_infer.sh"])
        print("🟣 Finished!")

        # 🔍 HR
        print(f"📥 Loading predictions from {args.predict_output_path}...")

        if args.hit == 1:
            y_true = []
            y_score = []
            with open(args.predict_output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    gt = item.get("label")
                    pred = item.get("predict")
                    if gt is None or pred is None:
                        continue
                    gt = gt.strip().lower()
                    pred = pred.strip().lower()
                    if gt not in {"yes", "no"} or pred not in {"yes", "no"}:
                        continue
                    y_true.append(1 if gt == "yes" else 0)
                    y_score.append(1 if pred == "yes" else 0)
            auc = roc_auc_score(y_true, y_score) if y_true else 0
            print(f"🎯 Final AUC: {auc:.4f}")

        elif args.hit == 3:
            hits = 0
            ndcg = 0
            total = 0
            k = 3
            with open(args.predict_output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    gt = item.get("label")
                    pred = item.get("predict")
                    if gt is None or pred is None:
                        continue
                    gt_list = [s.strip('" ') for s in gt.strip().split('\n') if s.strip()]
                    pred_list = [s.strip('" ') for s in pred.strip().split('\n') if s.strip()]
                    pred_topk = pred_list[:k]
                    if not gt_list:
                        continue
                    true_item = gt_list[0]
                    if true_item in pred_topk:
                        hits += 1
                    dcg = 0.0
                    for i, p in enumerate(pred_topk):
                        if p == true_item:
                            dcg = 1.0 / math.log2(i + 2)
                            break
                    ndcg += dcg
                    total += 1
            hr = hits / total if total > 0 else 0.0
            ndcg_score = ndcg / total if total > 0 else 0.0
            print(f"🎯 Final Hit Rate (HR@{k}): {hr:.4f}")
            print(f"📈 Final NDCG@{k}: {ndcg_score:.4f}")

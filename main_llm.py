import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from utils.data_loader import InteractionSFTDataset, InteractionDPODataset
from train.llm_sft import SFTTrainerWrapper
from train.llm_dpo import DPOTrainerWrapper
from validation.llm_infer import SFTInferWrapper
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT for Microvideo Recommendation")

    # 控制流程
    parser.add_argument("--do_train", action="store_true", help="是否进行训练")
    parser.add_argument("--do_predict", action="store_true", help="是否进行推理")
    parser.add_argument("--training_mode", choices=["sft", "dpo"], default="sft", help="选择训练模式：sft 或 dpo")

    # 数据集名称参数
    parser.add_argument("--dataset", type=str, default="microlens", help="指定数据集名称，例如 microlens 或 other_dataset")

    # 预设路径模板（会在解析后动态替换）
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--val_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--title_path", type=str)
    parser.add_argument("--template_path", type=str)
    parser.add_argument("--output_json_dir", type=str)
    parser.add_argument("--predict_output_path", type=str)
    parser.add_argument("--lora_name", type=str)
    parser.add_argument("--output_dir", type=str)

    # 其他数据参数
    parser.add_argument("--max_item_id", type=int, default=None, help="数据集中物品ID最大值")
    parser.add_argument("--num_negatives", type=int, default=7)
    parser.add_argument("--hit", type=int, default=1)

    # 模型相关
    parser.add_argument("--model_name", type=str, default="Qwen-2.5-3B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=1024)

    # 训练参数
    parser.add_argument("--train_sample_size", type=int, default=None, help="limit training samples")
    parser.add_argument("--seed", type=int, default=2025)

    # 推理参数
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--test_batch", type=int, default=100)

    args = parser.parse_args()

    # 根据 dataset 参数动态构造路径
    base = args.dataset
    training_mode = args.training_mode
    hit = args.hit
    if args.train_path is None:
        args.train_path = f"data/{base}/train.tsv"
    if args.val_path is None:
        args.val_path = f"data/{base}/val.tsv"
    if args.test_path is None:
        args.test_path = f"data/{base}/test.tsv"
    if args.title_path is None:
        args.title_path = f"data/{base}/{base.capitalize()}_titles.csv"
    if args.template_path is None:
        args.template_path = f"prompt/{base}.txt"
    if args.predict_output_path is None:
        args.predict_output_path = f"results/{base}/llm_{training_mode}_{hit}_test_predictions.jsonl"
    if args.lora_name is None:
        args.lora_name = f"checkpoints/{base}/{training_mode}_{hit}_lora_qwen/"
    if args.output_dir is None:
        args.output_dir = f"checkpoints/{base}/{training_mode}_{hit}_lora_qwen/"
    if args.max_item_id is None:
        if base == "microlens":
            args.max_item_id = 19379
        elif base == "netflix":
            args.max_item_id = 17771
        elif base == "movielens":
            args.max_item_id = 3953

    return args

if __name__ == '__main__':
    args = parse_args()

    if args.do_train:
        print("🧪 Loading training dataset...")
        if args.training_mode == "sft":
            train_dataset = InteractionSFTDataset(
                tsv_path=args.train_path,
                template_path=args.template_path,
                title_path=args.title_path,
                max_item_id=args.max_item_id,
                num_negatives=args.num_negatives,
                hit=args.hit
            )

            val_dataset = InteractionSFTDataset(
                tsv_path=args.val_path,
                template_path=args.template_path,
                title_path=args.title_path,
                max_item_id=args.max_item_id,
                num_negatives=args.num_negatives,
                hit=args.hit
            )
            print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")

            trainer = SFTTrainerWrapper(
                model_name=args.model_name,
                output_dir=args.output_dir,
                train_file=train_dataset,
                val_file=val_dataset,
                train_sample_size=args.train_sample_size or len(train_dataset),
                max_seq_length=args.max_tokens,
                seed=args.seed
            )
            trainer.train()

        elif args.training_mode == "dpo":
            train_dataset = InteractionDPODataset(
                tsv_path=args.train_path,
                template_path=args.template_path,
                title_path=args.title_path,
                max_item_id=args.max_item_id,
                num_negatives=args.num_negatives
            )

            val_dataset = InteractionDPODataset(
                tsv_path=args.val_path,
                template_path=args.template_path,
                title_path=args.title_path,
                max_item_id=args.max_item_id,
                num_negatives=args.num_negatives
            )
            print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")

            trainer = DPOTrainerWrapper(
                model_name=args.model_name,
                output_dir=args.output_dir,
                lora_path=args.lora_name,
                train_file=train_dataset,
                val_file=val_dataset,
                train_sample_size=args.train_sample_size or len(train_dataset),
                seed=args.seed
            )
            trainer.train()


    if args.do_predict:
        print("🧪 Loading test dataset...")
        test_dataset = InteractionSFTDataset(
            tsv_path=args.test_path,
            template_path=args.template_path,
            title_path=args.title_path,
            max_item_id=args.max_item_id,
            num_negatives=args.num_negatives,
            hit=args.hit
        )
        print(f"📊 Test {len(test_dataset)}")

        inferencer = SFTInferWrapper(
            model_name=args.model_name,
            lora_path=args.lora_name,
            test_file=test_dataset,
            max_new_tokens=args.max_new_tokens,
            test_batch=args.test_batch,
            hit=args.hit
        )

        outputs = inferencer.infer()
        inferencer.save(outputs, args.predict_output_path)
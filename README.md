# HaNoRec
Multimodal Large Language Models with Adaptive Preference Optimization for Sequential Recommendation

## Model Architecture
<img src='model.png' />

## 🧰 Environment Setup for HaNoRec

You can run the following command to download the codes faster:

```bash
conda create -y -n hanorec python=3.10
conda activate hanorec
pip install -r requirement.txt
```

## 💡 Dataset Structure and Download

You can download multimodal recommendation files in the following datasets:

**Microlens**/ **Netflix**/ **Movielens** [[Google Drive]](https://drive.google.com/file/d/1hYlIN6qt7vnzCd4ZwBSFkvMBeqo63d4b/view)

Each dataset consists of a training set, a validation set, and a test set. During the training process, we utilize the validation set to determine when to stop the training in order to prevent overfitting.

```plaintext
- microlens (netflix/movielens)
|--- dataset_pairs #  user-item pairs
|--- dataset_titles #  item titles
|--- train/val/test.tsv # training/validation/test set
|--- train/val/test_sft_1 # training/validation/test for MLLM SFT (HR@1)
|--- train/val/test_sft_3 # training/validation/test for MLLM SFT (HR@3 && NDCG@3)
```

> [!NOTE]
> The architecture tag above expects `model.png` at the repository root. Add the paper architecture figure with that filename when preparing a release; it is not included in the current working tree.

## Overview

HaNoRec improves multimodal sequential recommendation in two stages. First, Qwen2.5-VL is supervised on the current recommendation data to obtain a task-specific LoRA adapter. Second, the SFT samples are converted into deterministic preference pairs and optimized with HaNoRec, which combines Hardness-aware Reward Scaling (HaRS) and Noise-Driven Optimization (NoDO).

```text
Current TSV, titles, images, and SFT JSON
    -> Qwen2.5-VL supervised fine-tuning
    -> chosen/rejected preference construction
    -> multimodal catalog encoding and Top-K retrieval
    -> offline HaRS sample hardness
    -> HaNoRec DPO with dynamic per-sample beta and NoDO
```

The implementation supports two tasks:

| Argument | Task | Evaluation |
|---|---|---|
| `--hit 1` | Binary Yes/No next-item preference | AUC |
| `--hit 3` | Top-3 next-item ranking | HR@3 and NDCG@3 |

Run all commands from the repository root.

## Repository Layout

The relevant files are organized as follows:

```text
.
|--- Qwen-2.5-VL-3B-Instruct/        # local multimodal base model
|--- data/<dataset>/                 # current source data; treated as read-only
|--- checkpoints/<dataset>/
|    |--- sft_1_lora_qwenvl/         # SFT adapter for the binary task
|    `--- sft_3_lora_qwenvl/         # SFT adapter for the ranking task
|--- artifacts/hanorec/<dataset>/    # generated preferences, embeddings, and hardness
|--- configs/hanorec/                # six HaNoRec training configurations
|--- hanorec/                        # HaRS, NoDO, data, and trainer implementation
|--- scripts/prepare_hanorec.py      # offline preference and hardness preparation
`--- scripts/train_hanorec.py        # end-to-end HaNoRec entry point
```

The current repository already contains the six SFT adapter directories expected by the HaNoRec configurations. If you use these adapters, you may go directly to [HaNoRec data preparation](#stage-2-prepare-hanorec-preference-data-and-hars-hardness).

## Stage 1: Supervised Fine-Tuning

### 1.1 Prepare SFT data

For each dataset and task, the multimodal SFT files used by this repository follow these names:

```text
data/<dataset>/train-mllm_sft_<hit>.json
data/<dataset>/val-mllm_sft_<hit>.json
data/<dataset>/test-mllm_sft_<hit>.json
```

Each JSON sample contains a user/assistant conversation and the associated item image paths. Keep the TSV files, title CSV files, image directories, and SFT JSON files together under `data/<dataset>/`.

### 1.2 Train an SFT adapter

HaNoRec expects the SFT adapter at:

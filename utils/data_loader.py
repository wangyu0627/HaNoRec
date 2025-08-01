from torch.utils.data import Dataset
import pandas as pd
import random
import json
import os

class InteractionSFTDataset(Dataset):
    def __init__(self,
                 tsv_path,
                 title_path,
                 max_item_id,
                 template_path,
                 num_negatives=7,
                 hit=1):
        self.samples = []
        self.item2title = self._load_title_map(title_path)
        self.max_item_id = max_item_id
        self.num_negatives = num_negatives
        self.hit = hit
        self.max_seq_len = 6


        with open(template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        df = pd.read_csv(tsv_path, sep='\t', header=None)

        if self.hit == 1:
            for _, row in df.iterrows():
                user_id = int(row[0])
                item_ids = list(map(int, str(row[1]).strip().split()))
                if len(item_ids) < self.max_seq_len:
                    continue

                truncated = item_ids[-self.max_seq_len:]

                next_item_true = truncated[-1]
                seq = truncated[:-1]

                from_candidates = random.random() < 0.5

                negatives = set()
                target_num_negatives = num_negatives + 1 if from_candidates else num_negatives

                seen = set(item_ids)
                while len(negatives) < target_num_negatives:
                    neg = random.randint(1, self.max_item_id)
                    if neg not in seen:
                        negatives.add(neg)

                candidates = list(negatives)
                random.shuffle(candidates)

                if from_candidates:
                    next_item = random.choice(candidates)
                    candidates.remove(next_item)
                    answer = "No"
                else:
                    next_item = next_item_true
                    answer = "Yes"

                seq_titles = [self.item2title.get(i, f"[unknown {i}]") for i in seq]
                cand_titles = [self.item2title.get(i, f"[unknown {i}]") for i in candidates]
                next_title = self.item2title.get(next_item, f"[unknown {next_item}]")

                seq_str = '\n'.join([f'"{t}"' for t in seq_titles])
                cand_str = '\n'.join([f'"{t}"' for t in cand_titles])

                user_prompt = (f"The user has watched the following movies:\n{seq_str}\n\n"
                               f"The movies the user has not watched, which may be those they do not prefer:\n{cand_str}\n\n"
                               f"Whether the user will like the target movie:\n\"{next_title}\"")

                final_text = f"### Instruction:\nPlease answer Yes or No.\n### Input:\n{user_prompt.strip()}### Output:\n{answer}"

                self.samples.append({"text": final_text})

        else:
            for _, row in df.iterrows():
                user_id = int(row[0])
                item_ids = list(map(int, str(row[1]).strip().split()))
                if len(item_ids) < self.max_seq_len:
                    continue

                truncated = item_ids[-self.max_seq_len:]
                seq = truncated[:-self.hit]
                next_item = truncated[-self.hit:]
                next_item.reverse()

                seen = set(item_ids)
                negatives = set()
                while len(negatives) < num_negatives:
                    neg = random.randint(1, self.max_item_id)
                    if neg not in seen:
                        negatives.add(neg)

                candidates = list(negatives) + next_item
                random.shuffle(candidates)

                seq_titles = [self.item2title.get(i, f"[unknown {i}]") for i in seq]
                cand_titles = [self.item2title.get(i, f"[unknown {i}]") for i in candidates]
                next_titles = [self.item2title.get(i, f"[unknown {i}]") for i in next_item]

                seq_str  = '\n'.join([f'"{t}"' for t in seq_titles])
                cand_str = '\n'.join([f'"{t}"' for t in cand_titles])
                next_str = '\n'.join([f'"{t}"' for t in next_titles])

                user_prompt = self.prompt_template.replace("[seq]", seq_str).replace("[candidates]", cand_str)
                model_response = f'"{next_str}"'

                instruction = ("Given the title of a list of the user's recently enjoyed, please recommend a new item that the user may like. "
                           "Only output the title of the selected candidate item, without any additional explanation or description.")

                final_text = f"### Instruction:\n{instruction}\n### Input:\n{user_prompt.strip()}### Output:\n{model_response.strip()}"

                self.samples.append({"text": final_text})

    def _load_title_map(self, title_path):
        df = pd.read_csv(title_path)
        return dict(zip(df['item'], df['title']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class InteractionDPODataset(Dataset):
    def __init__(self,
                 tsv_path,
                 title_path,
                 max_item_id,
                 template_path,
                 num_negatives=4,
                 max_seq_len=5,
                 num_train_samples=500):
        self.samples = []
        self.item2title = self._load_title_map(title_path)
        self.max_item_id = max_item_id
        self.num_negatives = num_negatives
        self.num_train_samples = num_train_samples

        with open(template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        df = pd.read_csv(tsv_path, sep='\t', header=None)
        df = df.iloc[:self.num_train_samples]

        for _, row in df.iterrows():
            item_ids = list(map(int, str(row[1]).strip().split()))
            if len(item_ids) < max_seq_len:
                continue

            truncated = item_ids[-max_seq_len:]
            seq = truncated[:-1]
            next_item = truncated[-1]

            seen = set(item_ids)
            negatives = set()
            while len(negatives) < num_negatives:
                neg = random.randint(1, self.max_item_id)
                if neg not in seen:
                    negatives.add(neg)

            candidates = list(negatives) + [next_item]
            random.shuffle(candidates)

            # 
            seq_titles = [self.item2title.get(i, f"[unknown {i}]") for i in seq]
            cand_titles = [self.item2title.get(i, f"[unknown {i}]") for i in candidates]
            next_title = self.item2title.get(next_item, f"[unknown {next_item}]")

            # 
            rejected_item = random.choice(list(negatives))
            rejected_title = self.item2title.get(rejected_item, f"[unknown {rejected_item}]")

            instruction = (
                "Given the title of a list of the user's recently enjoyed, please recommend a new item that the user may like. "
                "Only output the title of the selected candidate item, without any additional explanation or description.")

            seq_str = '\n'.join([f'"{t}"' for t in seq_titles])
            cand_str = '\n'.join([f'"{t}"' for t in cand_titles])
            user_prompt = instruction.strip() + "\n\n" + self.prompt_template.replace("[seq]", seq_str).replace("[candidates]", cand_str)

            self.samples.append({
                "prompt": user_prompt.strip(),
                "chosen": next_title + "<|endoftext|>",
                "rejected": rejected_title + "<|endoftext|>"
            })

    def _load_title_map(self, title_path):
        df = pd.read_csv(title_path)
        return dict(zip(df['item'], df['title']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class MultimodalInteractionSFTDataset(Dataset):
    def __init__(self,
                 tsv_path,
                 title_path,
                 max_item_id,
                 template_path,
                 image_root,
                 save_path,
                 num_negatives=7,
                 hit=1):
        self.samples = []
        self.item2title = self._load_title_map(title_path)
        self.max_item_id = max_item_id
        self.num_negatives = num_negatives
        self.image_path = image_root
        self.save_path = save_path
        self.max_seq_len = 6
        self.hit = hit

        with open(template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        df = pd.read_csv(tsv_path, sep='\t', header=None)

        if self.hit == 1:
            for _, row in df.iterrows():
                user_id = int(row[0])
                item_ids = list(map(int, str(row[1]).strip().split()))
                if len(item_ids) < self.max_seq_len:
                    continue

                truncated = item_ids[-self.max_seq_len:]

                next_item_true = truncated[-1]
                seq = truncated[:-1]

                from_candidates = random.random() < 0.5

                negatives = set()
                target_num_negatives = num_negatives + 1 if from_candidates else num_negatives

                seen = set(item_ids)
                while len(negatives) < target_num_negatives:
                    neg = random.randint(1, self.max_item_id)
                    if neg not in seen:
                        negatives.add(neg)

                candidates = list(negatives)
                random.shuffle(candidates)

                if from_candidates:
                    next_item = random.choice(candidates)
                    candidates.remove(next_item)
                    answer = "No"
                else:
                    next_item = next_item_true
                    answer = "Yes"

                seq_titles = [self.item2title.get(i, f"[unknown {i}]") for i in seq]
                cand_titles = [self.item2title.get(i, f"[unknown {i}]") for i in candidates]
                next_title = self.item2title.get(next_item, f"[unknown {next_item}]")

                seq_str = '\n'.join([f'<image> "{t}"' for t in seq_titles])
                cand_str = '\n'.join([f'"{t}"' for t in cand_titles])

                user_prompt = (f"The user has watched the following movies:\n{seq_str}\n\n"
                               f"The movies the user has not watched, which may be those they do not prefer:\n{cand_str}\n\n"
                               f"Whether the user will like the target movie:\n\"{next_title}\"")

                final_text = f"### Instruction:\nPlease answer Yes or No.\n### Input:\n{user_prompt.strip()}### Output:\n{answer}"

                images = []
                for item_id in seq:
                    img_path = os.path.join(self.image_path, f"{item_id}.jpg")
                    if os.path.exists(img_path):
                        images.append(img_path)
                    else:
                        images.append(os.path.join(self.image_path, "627.jpg"))

                self.samples.append({
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": answer}
                    ],
                    "images": images
                })

        elif self.hit == 3:
            for _, row in df.iterrows():
                user_id = int(row[0])
                item_ids = list(map(int, str(row[1]).strip().split()))
                if len(item_ids) < self.max_seq_len:
                    continue

                truncated = item_ids[-self.max_seq_len:]
                seq = truncated[:-self.hit]
                next_item = truncated[-self.hit:]
                next_item.reverse()

                seen = set(item_ids)
                negatives = set()
                while len(negatives) < num_negatives:
                    neg = random.randint(1, self.max_item_id)
                    if neg not in seen:
                        negatives.add(neg)

                candidates = list(negatives) + next_item
                random.shuffle(candidates)

                seq_titles = [self.item2title.get(i, f"[unknown {i}]") for i in seq]
                cand_titles = [self.item2title.get(i, f"[unknown {i}]") for i in candidates]
                next_titles = [self.item2title.get(i, f"[unknown {i}]") for i in next_item]

                seq_str = '\n'.join([f'<image> "{t}"' for t in seq_titles])
                cand_str = '\n'.join([f'"{t}"' for t in cand_titles])
                next_str = '\n'.join([f'"{t}"' for t in next_titles])

                user_prompt = self.prompt_template.replace("[seq]", seq_str).replace("[candidates]", cand_str)
                model_response = f'"{next_str}"'

                # 
                instruction = (
                    "Given the title of a list of the user's recently enjoyed, please recommend a new item that the user may like. "
                    "Only output the title of the selected candidate item, without any additional explanation or description.")
                user_prompt = instruction + user_prompt

                images = []
                for item_id in seq:
                    img_path = os.path.join(self.image_path, f"{item_id}.jpg")

                    if os.path.exists(img_path):
                        images.append(img_path)
                    else:
                        empty_img_path = os.path.join(self.image_path, "627.jpg")
                        images.append(empty_img_path)


                sample = {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_prompt
                        },
                        {
                            "role": "assistant",
                            "content": model_response
                        }
                    ],
                    "images": images  # 
                }

                self.samples.append(sample)

        self.save_dataset(self.samples, self.save_path)

    def _load_title_map(self, title_path):
        df = pd.read_csv(title_path)
        return dict(zip(df['item'], df['title']))

    def save_dataset(self, dataset, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        data_to_save = dataset.samples if hasattr(dataset, "samples") else dataset

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        print(f"🟢 Saved dataset to {save_path} (total {len(data_to_save)} samples)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


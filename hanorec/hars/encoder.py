"""Lazy Qwen2.5-VL catalog encoder used by HaRS preprocessing."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from hanorec.data.preference import TitleIndex

from .cache import atomic_npz_dump
from .hardness import fuse_embeddings


class QwenVLItemEncoder:
    """Encode title and image modalities with the paper's MLLM backbone."""

    def __init__(self, model_name_or_path: str, *, device_map: str = "auto"):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Catalog encoding requires torch and transformers with Qwen2.5-VL support"
            ) from error

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map=device_map,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def encode_title(self, title: str) -> np.ndarray:
        tokenizer = self.processor.tokenizer
        inputs = tokenizer(title, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with self.torch.no_grad():
            states = self.model.model.embed_tokens(input_ids)
            mask = attention_mask.unsqueeze(-1).to(states.dtype)
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return pooled[0].float().cpu().numpy()

    def encode_image(self, image_path: Path) -> np.ndarray:
        from PIL import Image

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            inputs = self.processor.image_processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        grid_thw = inputs["image_grid_thw"].to(self.device)
        visual_dtype = next(self.model.visual.parameters()).dtype
        pixel_values = pixel_values.to(visual_dtype)
        with self.torch.no_grad():
            features = self.model.visual(pixel_values, grid_thw)
        return features.mean(dim=0).float().cpu().numpy()


def encode_catalog(
    *,
    title_path: str | Path,
    image_dir: str | Path,
    model_name_or_path: str,
    output_path: str | Path,
    extra_item_ids: Iterable[int] = (),
    device_map: str = "auto",
) -> dict[str, Any]:
    """Encode the catalog and write a compressed, self-contained embedding cache."""

    title_index = TitleIndex.from_csv(title_path)
    items = dict(title_index.items())
    for item_id in extra_item_ids:
        items.setdefault(int(item_id), f"[unknown {int(item_id)}]")
    ordered = sorted(items.items())
    if len(ordered) < 2:
        raise ValueError("Catalog must contain at least two items")

    encoder = QwenVLItemEncoder(model_name_or_path, device_map=device_map)
    text_vectors: list[np.ndarray] = []
    visual_vectors: list[np.ndarray | None] = []
    missing_images = 0
    expected_dimension: int | None = None
    root = Path(image_dir)

    for item_id, title in ordered:
        text = encoder.encode_title(title)
        if expected_dimension is None:
            expected_dimension = int(text.shape[0])
        if text.ndim != 1 or text.shape[0] != expected_dimension:
            raise ValueError(f"Unexpected text embedding shape for item {item_id}: {text.shape}")
        text_vectors.append(text)

        image_path = root / f"{item_id}.jpg"
        if image_path.exists():
            visual = encoder.encode_image(image_path)
            if visual.ndim != 1 or visual.shape[0] != expected_dimension:
                raise ValueError(f"Unexpected visual embedding shape for item {item_id}: {visual.shape}")
            visual_vectors.append(visual)
        else:
            visual_vectors.append(None)
            missing_images += 1

    text_matrix = np.stack(text_vectors).astype(np.float32)
    visual_matrix = np.stack(
        [np.zeros(expected_dimension, dtype=np.float32) if value is None else value for value in visual_vectors]
    ).astype(np.float32)
    fused = fuse_embeddings(text_matrix, visual_matrix).astype(np.float32)
    atomic_npz_dump(
        output_path,
        item_ids=np.asarray([item_id for item_id, _ in ordered], dtype=np.int64),
        text_embeddings=text_matrix,
        visual_embeddings=visual_matrix,
        fused_embeddings=fused,
        missing_images=np.asarray([missing_images], dtype=np.int64),
    )
    return {
        "items": len(ordered),
        "dimension": expected_dimension,
        "missing_images": missing_images,
        "output": str(Path(output_path)),
    }

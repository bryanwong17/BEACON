from typing import List, Union

import numpy as np
import torch
import PIL
from datasets import Dataset, Image as HFImage
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

# CLIP/PLIP's standard image normalization constants (ViT-B/32 checkpoint).
PLIP_IMAGE_MEAN = (0.481, 0.457, 0.408)
PLIP_IMAGE_STD = (0.268, 0.261, 0.275)


class PLIPEncoder:

    def __init__(self, ckpt_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = f"plip:{ckpt_path}"

        self.model = CLIPModel.from_pretrained(ckpt_path).to(self.device)
        self.preprocess = CLIPProcessor.from_pretrained(ckpt_path)
        self.embed_dim = self.model.config.projection_dim

        # For fast tensor-batch inference in the offline feature-extraction script.
        self.fast_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=PLIP_IMAGE_MEAN, std=PLIP_IMAGE_STD),
        ])

    def encode_images(self, images: Union[List[str], List["PIL.Image.Image"]], batch_size: int) -> np.ndarray:
        def transform_fn(el):
            imgs = (
                el["image"] if isinstance(el["image"][0], PIL.Image.Image)
                else [HFImage().decode_example(_) for _ in el["image"]]
            )
            return self.preprocess(images=imgs, return_tensors="pt")

        dataset = Dataset.from_dict({"image": images})
        if isinstance(images[0], str):
            dataset = dataset.cast_column("image", HFImage(decode=False))
        dataset.set_format("torch")
        dataset.set_transform(transform_fn)

        dataloader = DataLoader(dataset, batch_size=batch_size)
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                embeddings.extend(self.model.get_image_features(**batch).detach().cpu().numpy())
        return np.stack(embeddings)

    def encode_text(self, text: List[str], batch_size: int) -> np.ndarray:
        dataset = Dataset.from_dict({"text": text})
        dataset = dataset.map(
            lambda el: self.preprocess(
                text=el["text"], return_tensors="pt",
                max_length=77, padding="max_length", truncation=True,
            ),
            batched=True,
            remove_columns=["text"],
        )
        dataset.set_format("torch")

        dataloader = DataLoader(dataset, batch_size=batch_size)
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                embeddings.extend(self.model.get_text_features(**batch).detach().cpu().numpy())
        return np.stack(embeddings)

    @torch.no_grad()
    def encode_image_tensor_batch(self, pixel_values: torch.Tensor) -> np.ndarray:
        pixel_values = pixel_values.to(self.device)
        features = self.model.get_image_features(pixel_values=pixel_values)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

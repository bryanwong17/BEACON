import numpy as np
import torch

try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
except ImportError as e:
    raise ImportError(
        "The 'conch' package is required for --vlm_backend conch. Install the CONCH "
        "package (see its official repository for instructions) and retry.\n"
        f"Original import error: {e}"
    ) from e

# CONCH's shared image/text embedding dim (conch_ViT-B-16 model config).
CONCH_EMBED_DIM = 512
CONCH_MODEL_CFG = "conch_ViT-B-16"


class CONCHEncoder:

    def __init__(self, ckpt_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = f"conch:{ckpt_path}"
        self.embed_dim = CONCH_EMBED_DIM

        self.model, self.fast_preprocess = create_model_from_pretrained(
            CONCH_MODEL_CFG, ckpt_path, device=self.device
        )
        self.tokenizer = get_tokenizer()

    @torch.no_grad()
    def encode_image_tensor_batch(self, pixel_values: torch.Tensor) -> np.ndarray:
        pixel_values = pixel_values.to(self.device)
        features = self.model.encode_image(pixel_values, proj_contrast=True, normalize=True)
        return features.cpu().numpy()

    def encode_images(self, images, batch_size: int) -> np.ndarray:
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            pixel_values = torch.stack([self.fast_preprocess(img) for img in batch], dim=0)
            all_feats.append(self.encode_image_tensor_batch(pixel_values))
        return np.concatenate(all_feats, axis=0)

    @torch.no_grad()
    def encode_text(self, text, batch_size: int) -> np.ndarray:
        all_feats = []
        for i in range(0, len(text), batch_size):
            batch = list(text[i:i + batch_size])
            tokens = tokenize(self.tokenizer, batch).to(self.device)
            features = self.model.encode_text(tokens, normalize=True)
            all_feats.append(features.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

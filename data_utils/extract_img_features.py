import argparse
import os
import sys
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DISCERN_ROOT = os.path.dirname(_THIS_DIR)
if _DISCERN_ROOT not in sys.path:
    sys.path.insert(0, _DISCERN_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract patch-level VLM features for DISCERN")

    parser.add_argument("--vlm_backend", type=str, default="plip", choices=["plip", "conch"],
                        help="Which VLM to encode patches with.")
    parser.add_argument("--vlm_ckpt", type=str, required=True,
                        help="Path or HF id for the VLM checkpoint.")

    parser.add_argument("--task_file", type=str, required=True,
                        help="Text file listing slide folder names (one per line) to process.")
    parser.add_argument("--img_root", type=str, required=True,
                        help="Root directory containing per-slide patch folders.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save one combined {slide_id}.npy per slide.")

    parser.add_argument("--batch_size", type=int, default=256,
                        help="Patches per encode batch.")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="DataLoader worker processes for image decode/preprocess.")

    return parser.parse_args()


class PatchTensorDataset(Dataset):

    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), os.path.basename(path)


def run_processing(args: argparse.Namespace) -> None:
    from vlm.factory import build_vlm_encoder

    print(f"Loading VLM backend '{args.vlm_backend}' from: {args.vlm_ckpt}")
    vlm_encoder = build_vlm_encoder(args.vlm_backend, args.vlm_ckpt)
    vlm_encoder.model.eval()
    print(f"VLM encoder ready: {vlm_encoder.model_name}")

    with open(args.task_file, "r") as f:
        slide_ids = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Task file: {args.task_file}, {len(slide_ids)} slides to process")
    print(f"Image root: {args.img_root}")
    print(f"Output dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    for slide_id in slide_ids:
        save_path = os.path.join(args.output_dir, f"{slide_id}.npy")
        if os.path.exists(save_path):
            print(f"Skipping {slide_id}, already extracted.")
            continue

        img_dir = os.path.join(args.img_root, slide_id)
        image_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
        if not image_paths:
            print(f"{slide_id}: no .jpg patches found, skipping")
            continue

        dataset = PatchTensorDataset(image_paths, vlm_encoder.fast_preprocess)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        all_embs = []
        all_names = []
        print(f"Processing {slide_id} ({len(image_paths)} patches)...")
        for pixel_values, names in dataloader:
            embs = vlm_encoder.encode_image_tensor_batch(pixel_values)
            all_embs.append(embs)
            all_names.extend(names)

        if all_embs:
            combined_embs = np.concatenate(all_embs, axis=0)
            np.save(save_path, {"embeddings": combined_embs, "filenames": all_names})
            print(f"Saved {len(all_names)} features to {save_path}")


if __name__ == "__main__":
    run_processing(parse_args())

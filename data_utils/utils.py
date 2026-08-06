import ast
import hashlib
import json
import os

import pandas as pd
from PIL import Image


def load_image(image_file):
    return Image.open(image_file).convert("RGB")


def maybe_resize_for_perceptor(image, size: int = 1024):
    size = int(size)
    if size <= 0:
        return image
    if getattr(image, "size", None) == (size, size):
        return image
    if hasattr(image, "resize"):
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = 1
        return image.resize((size, size), resample=resample)
    return image


def make_unique_id(long_id: str, question_text: str) -> str:
    q_hash = hashlib.md5(question_text.encode("utf-8")).hexdigest()[:8]
    return f"{long_id}_{q_hash}"


def load_all_vqa_pairs(vqa_file, dataset_name='wsi_vqa', image_dir=None):
    assert os.path.exists(vqa_file), f"File not found: {vqa_file}"
    vqa_pairs = []

    if dataset_name.lower() == 'wsi_vqa':
        with open(vqa_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        case2qas = {}
        for item in data:
            sid = item["Id"]
            case2qas.setdefault(sid, []).append(item)

        if image_dir is not None:
            dirs = [d for d in os.listdir(image_dir) if "DX1" in d]
        else:
            dirs = case2qas.keys()

        for d in dirs:
            long_id = d if image_dir else None
            short_id = d[:12] if image_dir else d

            if short_id not in case2qas:
                continue

            for qa in case2qas[short_id]:
                vqa_pairs.append({
                    "short_id": short_id,
                    "long_id": long_id,
                    "question": qa["Question"],
                    "choices": qa.get("Choice"),
                    "answer": qa["Answer"]
                })

    elif dataset_name.lower() == 'slidebench_vqa':
        df = pd.read_csv(vqa_file)

        if image_dir is not None:
            valid_slides = set([d.split(".")[0] for d in os.listdir(image_dir) if "DX1" in d])
            df = df[df["Slide"].isin(valid_slides)]

        for _, row in df.iterrows():
            choices = [row["A"], row["B"], row["C"], row["D"]]
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            ans_letter = str(row["Answer"]).strip().upper()
            ans_text = choices[answer_map[ans_letter]] if ans_letter in answer_map else row["Answer"]

            vqa_pairs.append({
                "long_id": row["Slide"],
                "question": row["Question"],
                "choices": choices,
                "answer": ans_text
            })

    elif 'multipathqa' in dataset_name.lower():
        df = pd.read_csv(vqa_file)


        target_benchmark = dataset_name.split("_", 1)[1] if "_" in dataset_name else None
        if target_benchmark is not None:
            df = df[df["benchmark_name"] == target_benchmark]
            print(f"Filtered to {len(df)} entries in MultiPathQA for benchmark: {target_benchmark}")
            if df.empty:
                raise ValueError(f"No entries found for benchmark: {target_benchmark}")

        for _, row in df.iterrows():
            raw_options = row["options"]
            if pd.isna(raw_options):
                choices = None
            else:
                try:
                    choices = ast.literal_eval(raw_options)
                except Exception:
                    choices = raw_options

            # MultiPathQA answers are 1-indexed into `choices`; PANDA's answer
            # column already holds the target text, so it skips this mapping.
            ans_val = str(row["answer"])
            ans_text = ans_val
            if target_benchmark != 'panda' and ans_val.isdigit():
                idx = int(ans_val)
                if 1 <= idx <= len(choices):
                    ans_text = choices[idx - 1]

            slide_id = str(row["image_path"]).split('.')[0]

            vqa_pairs.append({
                "long_id": slide_id,
                "question": row["prompt"],
                "choices": choices,
                "answer": ans_text
            })

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    print(f"Loaded {len(vqa_pairs)} VQA pairs from {dataset_name}")
    return vqa_pairs


def extract_coords_from_name(patch_name: str):
    base = os.path.basename(patch_name)
    name_no_ext = os.path.splitext(base)[0]
    try:
        x_str, y_str = name_no_ext.split("_")
        return int(x_str), int(y_str)
    except Exception:
        return None, None


def build_descriptions_with_meta(items, mag_level=None, include_header=True, include_coords=True) -> str:
    header = ""
    if include_header and mag_level is not None:
        header = f"[Current Magnification: {mag_level}x]\n\n"

    parts = []
    for name, desc in items:
        x, y = extract_coords_from_name(name)
        coord_str = f"({x},{y})" if (x is not None and y is not None) else "(unknown)"
        if include_coords:
            parts.append(f"[{name} | Coord={coord_str}] {desc}")
        else:
            parts.append(f"[{name}] {desc}")

    body = "\n\n".join(parts)
    return header + body


def get_patch_fullpath(patch_root, target_long_id, patch_name) -> str:
    base = os.path.join(patch_root, target_long_id, patch_name)
    if os.path.exists(base):
        return base
    for ext in [".jpg", ".jpeg", ".png"]:
        p = base + ext
        if os.path.exists(p):
            return p
    return base


def split_patch_for_zoom(patch_path, zoom_level: int):
    if isinstance(patch_path, (str, os.PathLike)):
        img = Image.open(patch_path).convert("RGB")
        base_name = os.path.basename(patch_path)
        name_no_ext = os.path.splitext(base_name)[0]

    elif isinstance(patch_path, Image.Image):
        img = patch_path
        name_no_ext = "0_0"
    else:
        raise TypeError(f"patch_path must be a file path or PIL.Image.Image object, received {type(patch_path)}")

    width, height = img.size

    if zoom_level % 5 != 0 or zoom_level < 5:
        raise ValueError(f"zoom_level must be >=5 and a multiple of 5, received {zoom_level}")

    try:
        base_x, base_y = map(int, name_no_ext.split("_"))
    except ValueError:
        base_x, base_y = 0, 0

    factor = zoom_level // 5
    if factor == 1:
        return [(img, (base_x, base_y))]

    sub_w = width // factor
    sub_h = height // factor

    patches = []
    for i in range(factor):
        for j in range(factor):
            left = j * sub_w
            upper = i * sub_h
            right = (j + 1) * sub_w if j < factor - 1 else width
            lower = (i + 1) * sub_h if i < factor - 1 else height
            patch = img.crop((left, upper, right, lower))

            global_x = base_x + left
            global_y = base_y + upper

            patches.append((patch, (global_x, global_y)))

    return patches


def get_specific_case_descriptions(file_path, long_id):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        descriptions_dict = data.get(long_id)
        if descriptions_dict and isinstance(descriptions_dict, dict):
            return descriptions_dict
        else:
            print(f"Error: ID '{long_id}' not found in file or format is not a dictionary.")
            return None
    except Exception as e:
        print(f"Error: Error reading description file - {e}")
        return None

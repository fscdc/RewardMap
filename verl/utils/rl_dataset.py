import os
import io
import re
import base64
import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import datasets as hfds
from datasets import load_from_disk
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index

# add this line to support super big image
Image.MAX_IMAGE_PIXELS = None

# add padding token
def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    pad_token_id = int(features[0].get("pad_token_id", 0))

    input_ids_list = [f["input_ids"] for f in features if "input_ids" in f]
    attn_mask_list = [f["attention_mask"] for f in features if "attention_mask" in f]
    pos_ids_list = [f["position_ids"] for f in features if "position_ids" in f]

    def _seq_len(x: torch.Tensor) -> int:
        return x.shape[-1]

    max_len = 0
    if input_ids_list:
        max_len = max(max_len, max(_seq_len(x) for x in input_ids_list))
    if attn_mask_list:
        max_len = max(max_len, max(_seq_len(x) for x in attn_mask_list))
    if pos_ids_list:
        max_len = max(max_len, max(_seq_len(x) for x in pos_ids_list))

    def pad_1d(x: torch.Tensor, L: int, pad_val: int) -> torch.Tensor:
        if x.ndim != 1:
            raise ValueError(f"pad_1d expects 1D tensor, got {x.shape}")
        if x.shape[0] == L:
            return x
        out = x.new_full((L,), pad_val)
        out[: x.shape[0]] = x
        return out

    def pad_pos(x: torch.Tensor, L: int) -> torch.Tensor:
        if x.ndim == 1:
            if x.shape[0] == L:
                return x
            out = x.new_zeros((L,))
            out[: x.shape[0]] = x
            return out
        elif x.ndim == 2:
            C = x.shape[0]
            if x.shape[1] == L:
                return x
            out = x.new_zeros((C, L))
            out[:, : x.shape[1]] = x
            return out
        else:
            raise ValueError(f"Unexpected position_ids ndim={x.ndim}, shape={x.shape}")

    batch = []
    for f in features:
        g = dict(f)
        if "input_ids" in g:
            g["input_ids"] = pad_1d(g["input_ids"], max_len, pad_token_id)
        if "attention_mask" in g:
            g["attention_mask"] = pad_1d(g["attention_mask"], max_len, 0)
        if "position_ids" in g:
            g["position_ids"] = pad_pos(g["position_ids"], max_len)
        batch.append(g)

    tensors: Dict[str, List[torch.Tensor]] = defaultdict(list)
    non_tensors: Dict[str, List[Any]] = defaultdict(list)

    for f in batch:
        for k, v in f.items():
            if isinstance(v, torch.Tensor):
                tensors[k].append(v)
            else:
                non_tensors[k].append(v)

    stacked: Dict[str, torch.Tensor] = {}
    for k, vs in tensors.items():
        if k in ["pixel_values", "image_grid_thw"]:
            continue
        stacked[k] = torch.stack(vs, dim=0)

    out: Dict[str, Any] = {}
    out.update(stacked)
    for k, vs in non_tensors.items():
        out[k] = vs
    if "pixel_values" in tensors:
        out["pixel_values"] = tensors["pixel_values"]  # list[Tensor]
    if "image_grid_thw" in tensors:
        out["image_grid_thw"] = tensors["image_grid_thw"]  # list[Tensor]

    return out


def process_image(
    image: ImageObject, max_pixels: Optional[int], min_pixels: Optional[int]
) -> ImageObject:
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize(
            (int(image.width * resize_factor), int(image.height * resize_factor)),
            resample=Image.Resampling.NEAREST,
        )
    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize(
            (int(image.width * resize_factor), int(image.height * resize_factor)),
            resample=Image.Resampling.NEAREST,
        )
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\n\r]+$")


def _figure_to_pil(
    figure_value: Any, dataset_root_for_files: Optional[str]
) -> ImageObject:
    s = figure_value.strip()
    if s.startswith("data:image"):
        s = s.split(",", 1)[-1]
        
    candidate_paths = [s[len("file://"):] if s.startswith("file://") else s]

    roots: List[str] = []
    if dataset_root_for_files is not None:
        if isinstance(dataset_root_for_files, (list, tuple)):
            roots.extend([os.path.abspath(r) for r in dataset_root_for_files])
        else:
            r = os.path.abspath(dataset_root_for_files)
            roots.append(r)
            
    for p in list(candidate_paths):
        if os.path.isabs(p) and os.path.exists(p):
            return Image.open(p).convert("RGB")

    for root in roots:
        for p in candidate_paths:
            # print(f"candidate_paths: {candidate_paths}")
            abs_p = os.path.abspath(os.path.join(root, p))
            # print(f"abs_p: {abs_p}")
            if os.path.exists(abs_p):
                # print(f"abs_p: {abs_p}")
                return Image.open(abs_p).convert("RGB")
            
    try:
        raw = base64.b64decode(s, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Base64 解码失败：{e}")


class RLHFDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "question",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        add_meta_context: bool = True,
        figure_column: str = "figure",
        prompt_key_candidates: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.add_meta_context = add_meta_context
        self.figure_column = figure_column

        ds = load_from_disk(data_path)
        self.dataset = ds["train"]
        
        # @sicheng, kaiwen: difficulty-aware fine-grained curriculum order 
        types = self.dataset["type"]
        torf1_idx, torf2_idx, counting1_idx, counting2_idx, counting3_idx, planning_idx, other_idx = [], [], [], [], [], [], []
        for i, t in enumerate(types):
            s = (str(t) if t is not None else "").lower()
            if s == "torf1":                          
                torf1_idx.append(i)
            elif s == "torf2":
                torf2_idx.append(i)
            elif s == "counting1":
                counting1_idx.append(i)
            elif s == "counting2":
                counting2_idx.append(i)
            elif s == "counting3":
                counting3_idx.append(i)
            elif s == "planning":
                planning_idx.append(i)
            else:
                other_idx.append(i)

        assert len(other_idx) == 0, f"other_idx should be 0, but got {len(other_idx)}"

        # TODO@sicheng, kaiwen
        torf_idx = torf1_idx + torf2_idx
        random.shuffle(torf_idx)
        count23_idx = counting2_idx + counting3_idx
        random.shuffle(count23_idx)
        random.shuffle(planning_idx)
        random.shuffle(counting1_idx)
        new_order = torf_idx + count23_idx + counting1_idx + planning_idx

        # # @sicheng: coarse 
        # plus_idx = counting1_idx + counting2_idx + counting3_idx + torf1_idx + torf2_idx
        # random.shuffle(plus_idx)
        # new_order = plus_idx + planning_idx

        # # @sicheng: fine-grained
        # random.shuffle(torf1_idx)
        # random.shuffle(torf2_idx)
        # random.shuffle(counting1_idx)
        # random.shuffle(counting2_idx)
        # random.shuffle(counting3_idx)
        # new_order = torf1_idx + torf2_idx + counting3_idx + counting2_idx + counting1_idx + planning_idx

        self.dataset = self.dataset.select(new_order)
        
        print(
            f"[RLHFDataset] Curriculum order applied: "
            f"torf1: {len(torf1_idx)}, torf2: {len(torf2_idx)}, "
            f"counting1: {len(counting1_idx)}, counting2: {len(counting2_idx)}, counting3: {len(counting3_idx)}, "
            f"planning: {len(planning_idx)}"
        )
        
        # @Func: filter questions by type for training
        # self.dataset = self.dataset.filter(
        #     lambda ex: str(ex.get("type", "")).lower() == "planning",
        # )

        self._dataset_train_dir = os.path.join(os.path.abspath(data_path), "train")
        self._dataset_root_dir = os.path.abspath(data_path)
        self._project_root_dir  = os.path.abspath(os.path.join(data_path, "..", ".."))
        feat = self.dataset.features.get(self.figure_column, None)
        self._figure_is_hf_image = isinstance(feat, hfds.Image)

        all_cols = set(self.dataset.column_names)
        if self.prompt_key not in all_cols:
            fallbacks = (prompt_key_candidates or []) + [
                "question",
                "problem",
                "prompt",
            ]
            found = next((k for k in fallbacks if k in all_cols), None)
            if found is None:
                raise ValueError(
                    f"数据集缺少字段：{self.prompt_key}；且未找到回退列。可用列：{sorted(list(all_cols))}"
                )
            print(
                f"[RLHFDataset] 未找到列 '{self.prompt_key}'，已回退使用列 '{found}'."
            )
            self.prompt_key = found

        self.user_prompt = (
            "<image>\n"
            "{Question}"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        row["pad_token_id"] = self.tokenizer.pad_token_id

        q = str(row[self.prompt_key]).strip()

        if self.add_meta_context:
            meta_parts = []
            for k in [
                "country",
                "city",
                "type",
                "difficulty_city",
                "city_line_count",
                "city_transfer_count",
                "station_1",
                "station_2",
            ]:
                if k in row and row[k] is not None:
                    meta_parts.append(f"{k}={row[k]}")
            meta_str = (
                ("Context: " + ", ".join(map(str, meta_parts)) + "\n")
                if meta_parts
                else ""
            )
        else:
            meta_str = ""

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": self.user_prompt.format(
                    meta=meta_str,
                    Question=q,
                    Answer="2",
                ),
            }
        )
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        fig_value = row[self.figure_column]
        pil_img = _figure_to_pil(
            fig_value, dataset_root_for_files=[self._dataset_train_dir, self._dataset_root_dir, self._project_root_dir]
        )
        pil_img = process_image(pil_img, self.max_pixels, self.min_pixels)
        images = [pil_img]

        raw_prompt = prompt.replace(
            "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
        )

        image_inputs = self.processor.image_processor(images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]

        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            occ, p = 0, prompt
            while "<image>" in p:
                num_placeholders = int(
                    (image_grid_thw[occ].prod() // merge_length).item()
                )
                p = p.replace(
                    "<image>",
                    "<|vision_start|>"
                    + "<|placeholder|>" * num_placeholders
                    + "<|vision_end|>",
                    1,
                )
                occ += 1
            prompt = p.replace("<|placeholder|>", self.processor.image_token)

        full_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_len = len(full_ids)
        effective_max_len = max(self.max_prompt_length, full_len)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=effective_max_len,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="error",
        )
        print(f"prompt.shape: {len(prompt)}")

        position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )  # (3, L)

        out: Dict[str, Any] = dict(row)
        out["images"] = images
        out.update(image_inputs)
        out["input_ids"] = input_ids
        out["attention_mask"] = attention_mask
        out["position_ids"] = position_ids
        out["raw_prompt_ids"] = self.tokenizer.encode(
            raw_prompt, add_special_tokens=False
        )
        return out

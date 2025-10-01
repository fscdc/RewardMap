#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge FSDP-sharded DTensor checkpoints into a single HuggingFace model (LOCAL ONLY).

- Keeps original logic and assumptions:
  * Expect files like: model_world_size_{N}_rank_{r}.pt
  * Check and read sharding info from DTensor. Supports FSDP-only (no TP) like the original.
  * Loads config from a HF-style folder (defaults to {local_dir}/huggingface).
  * Instantiates an empty model on 'meta', then saves merged state_dict to out_dir.
- Removes any uploading logic.

Usage:
  python merge_fsdp_checkpoints.py \
      --local_dir /path/to/global_step_43/actor \
      --out_dir  /path/to/output/merged_model \
      --config_dir /path/to/global_step_43/actor/huggingface \
      --dtype bfloat16
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
)

# ----------------------------- utils ----------------------------- #
def merge_by_placement(tensors: List[torch.Tensor], placement: Placement) -> torch.Tensor:
    if placement.is_replicate():
        # replicated: all shards identical, take first
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet.")
    elif placement.is_shard():
        # concat along shard dim
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")

def infer_world_size(local_dir: str) -> int:
    for filename in os.listdir(local_dir):
        m = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if m:
            return int(m.group(1))
    raise FileNotFoundError("No model file like 'model_world_size_{N}_rank_0.pt' found.")

def pick_automodel_from_config(cfg: AutoConfig):
    arch = cfg.architectures[0] if getattr(cfg, "architectures", None) else ""
    if "ForTokenClassification" in arch:
        return AutoModelForTokenClassification
    if "ForCausalLM" in arch:
        return AutoModelForCausalLM
    if "ForConditionalGeneration" in arch:
        # for many VLMs (e.g., vision-encoder-decoder / vision2seq)
        return AutoModelForVision2Seq
    raise NotImplementedError(f"Unknown architecture in config: {cfg.architectures}")

def str_to_dtype(s: str):
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {s}")

# ----------------------------- main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str,
                        help="Directory containing sharded files model_world_size_{N}_rank_{r}.pt")
    parser.add_argument("--out_dir", required=True, type=str,
                        help="Output directory to save the merged HF model.")
    parser.add_argument("--config_dir", type=str, default=None,
                        help="HF config dir (defaults to {local_dir}/huggingface).")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Save dtype: bfloat16 | fp16 | fp32 (default: bfloat16).")
    args = parser.parse_args()

    local_dir = os.path.abspath(args.local_dir)
    out_dir = os.path.abspath(args.out_dir)
    config_dir = os.path.abspath(args.config_dir) if args.config_dir else os.path.join(local_dir, "huggingface")
    dtype = str_to_dtype(args.dtype)

    if not os.path.isdir(local_dir):
        raise NotADirectoryError(f"local_dir not found: {local_dir}")
    if not os.path.isdir(config_dir):
        raise NotADirectoryError(
            f"config_dir not found: {config_dir}. It must contain a HuggingFace config.json."
        )
    os.makedirs(out_dir, exist_ok=True)

    # Locate world_size and load rank0
    world_size = infer_world_size(local_dir)
    rank0_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_0.pt")
    print(f"[INFO] world_size={world_size}, loading rank0: {rank0_path}")

    rank0_state: Dict[str, torch.Tensor] = torch.load(rank0_path, map_location="cpu", weights_only=False)
    if not rank0_state:
        raise RuntimeError("Empty state_dict from rank 0 file.")
    pivot_key = sorted(rank0_state.keys())[0]
    first_weight = rank0_state[pivot_key]
    if not isinstance(first_weight, DTensor):
        raise TypeError("Expected DTensor in sharded weights (FSDP); got non-DTensor tensor.")

    device_mesh = first_weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names
    print(f"[INFO] device mesh: {mesh}, mesh_dim_names: {mesh_dim_names}")

    # Keep parity with the original: support only (('fsdp',),)
    assert mesh_dim_names in (("fsdp",),), f"Unsupported mesh_dim_names {mesh_dim_names} (only FSDP is supported)."

    total_shards = mesh.shape[-1]
    mesh_shape = (mesh.shape[-1],)
    print(f"[INFO] Detected FSDP-only, total_shards={total_shards}, mesh_shape={mesh_shape}")

    # Prepare containers and async loads for all ranks
    model_state_dict_lst: List[Dict[str, torch.Tensor]] = [None] * total_shards  # type: ignore
    model_state_dict_lst[0] = rank0_state

    def load_rank(r: int):
        p = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{r}.pt")
        return r, torch.load(p, map_location="cpu", weights_only=False)

    futures = []
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 32)) as ex:
        for r in range(1, total_shards):
            futures.append(ex.submit(load_rank, r))
        for fut in as_completed(futures):
            r, st = fut.result()
            model_state_dict_lst[r] = st
            print(f"[LOAD] rank {r} loaded ({len(st)} tensors).")

    # Merge by param key
    merged_state: Dict[str, torch.Tensor] = {}
    param_placements: Dict[str, Tuple[Placement, ...]] = {}

    all_keys = set(model_state_dict_lst[0].keys())
    # sanity: ensure all ranks contain same keys
    for r in range(1, total_shards):
        all_keys &= set(model_state_dict_lst[r].keys())
    missing = []
    for r in range(total_shards):
        if set(model_state_dict_lst[r].keys()) != all_keys:
            missing.append(r)
    if missing:
        print(f"[WARN] Some ranks have extra/missing keys. Proceeding with intersection of keys: {len(all_keys)}")

    # Collect shards for each key
    shards_bucket: Dict[str, List[torch.Tensor]] = {k: [] for k in all_keys}

    for key in sorted(all_keys):
        for r in range(total_shards):
            t = model_state_dict_lst[r].pop(key)
            if isinstance(t, DTensor):
                # Convert local shard to desired dtype immediately
                shards_bucket[key].append(t._local_tensor.to(dtype))
                placements = tuple(t.placements)
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements, \
                        f"Placement mismatch for key {key}"
            else:
                # Non-DTensor params (e.g., buffers) are fully replicated
                merged_state[key] = t.to(dtype)

    # Free per-rank dicts
    del model_state_dict_lst

    # Merge DTensor shards
    for key, shards in shards_bucket.items():
        if not shards:
            # already handled as replicated above
            continue
        placements: Tuple[Shard, ...] = param_placements[key]  # type: ignore
        # FSDP-only => expect single Shard placement
        assert len(placements) == 1, f"Expect 1 placement for FSDP-only, got {len(placements)} for {key}"
        merged_state[key] = merge_by_placement(shards, placements[0])

    # Instantiate HF model on meta and save
    print(f"[INFO] Loading HF config from: {config_dir}")
    config = AutoConfig.from_pretrained(config_dir)
    AutoModel = pick_automodel_from_config(config)

    print(f"[INFO] Building empty model on 'meta' with dtype={dtype} ...")
    with torch.device("meta"):
        model = AutoModel.from_config(config, torch_dtype=dtype)
    model.to_empty(device="cpu")

    print(f"[INFO] Saving merged model to: {out_dir}")
    model.save_pretrained(out_dir, state_dict=merged_state)
    # If tokenizer files exist in config_dir, you may want to copy them manually (optional)
    print("[DONE] Merge finished. You can load it via `from_pretrained(out_dir)`.")

if __name__ == "__main__":
    main()

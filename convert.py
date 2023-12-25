import argparse
import copy
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from model import ModelArgs, GPTNeoForCausalLM
from transformers import AutoModelForCausalLM


def replace_key(key: str) -> str:
    if key.startswith("transformer."):
        # remove transformer prefix
        key = key.replace("transformer.", "")

    return key


def convert(args):
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16
    )
    state_dict = model.state_dict()
    weights = {replace_key(k): v.numpy() for k, v in state_dict.items()}
    config = model.config.to_dict()


    np.savez(str(mlx_path / "weights.npz"), **weights)

    # write config
    with open(mlx_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen model to npz")

    parser.add_argument(
        "--model",
        help="The huggingface model to be converted",
        default="roneneldan/TinyStories-1M",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="The path to save the MLX model.",
    )
    args = parser.parse_args()
    convert(args)
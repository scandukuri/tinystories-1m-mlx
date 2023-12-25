# (Pdb) model
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer


@dataclass
class ModelArgs:
    max_sequence_length: int = 2048
    vocab_size: int = 50257
    attention_dims: int = 64
    mlp_dims: Tuple[int] = (64, 256, 64)
    embed_dim: int = 64
    num_blocks: int = 8


class GPTNeoBlock(nn.Module):
    def __init__(self, attention_dims, mlp_dims):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dims=attention_dims, eps=1e-05, affine=True)
        self.attn = GPTNeoAttention(attention_dims)
        self.ln_2 = nn.LayerNorm(dims=attention_dims, eps=1e-05, affine=True)
        self.mlp = GPTNeoMLP(mlp_dims[0], mlp_dims[1], mlp_dims[2])
                             
    
    def __call__(self):
        # TODO
        pass
    
    
class GPTNeoMLP(nn.Module):
    def __init__(self, dims_fc, dims_proj, output_dim):
        super().__init__()
        self.c_fc = nn.Linear(input_dims=dims_fc, output_dims=dims_proj)
        self.c_proj = nn.Linear(input_dims=dims_proj, output_dims=output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.0)
        
    
    def __call__(self):
        # TODO
        pass


class GPTNeoAttention(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.attention = GPTNeoSelfAttention(dims)
    
    
    def __call__(self, x, mask, cache):
        # TODO
        pass
    
    
class GPTNeoSelfAttention(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.attn_dropout = nn.Dropout(p=0.0)
        self.resid_dropout = nn.Dropout(p=0.0)
        self.k_proj = nn.Linear(input_dims=dims, output_dims=dims, bias=False)
        self.v_proj = nn.Linear(input_dims=dims, output_dims=dims, bias=False)
        self.q_proj = nn.Linear(input_dims=dims, output_dims=dims, bias=False)
        self.out_proj = nn.Linear(input_dims=dims, output_dims=dims, bias=True)
    
    
    def __call__(self, x, mask=None, cache=None):
        # TODO
        pass



class GPTNeoModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(num_embeddings=config.vocab_size, dims=config.embed_dim)
        self.wpe = nn.Embedding(num_embeddings=2048, dims=config.embed_dim)
        self.drop = nn.Dropout(p=0.0)
        # self.h should be 8 times the block below (or config.num_blocks times)
        self.h = [GPTNeoBlock(attention_dims=config.attention_dims, mlp_dims=config.mlp_dims)]
        self.single_block = GPTNeoBlock(attention_dims=config.attention_dims, mlp_dims=config.mlp_dims)
        self.ln_f = nn.LayerNorm(dims=(64,), eps=1e-05, affine=True)


    def __call__(self, x):
        pass
    
    
class GPTNeoForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPTNeoModel(config)
        self.lm_head = nn.Linear(input_dims=64, output_dims=50257, bias=False)
        
        
    def __call__(self, x):
        # TODO
        pass

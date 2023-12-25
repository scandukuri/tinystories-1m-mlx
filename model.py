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


class GPTNeoForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPTNeoModel(config)
        self.lm_head = nn.Linear(input_dims=config.embed_dim, output_dims=config.vocab_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        # Pass input through transformer model
        transformer_output, new_cache = self.transformer(x, mask=mask, cache=cache)

        # Pass transformer output through language modeling head
        lm_logits = self.lm_head(transformer_output)

        # Return the logits and the updated cache if it was used
        if cache is not None:
            return lm_logits, new_cache
        else:
            return lm_logits
    

class GPTNeoModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(num_embeddings=config.vocab_size, dims=config.embed_dim)
        self.wpe = nn.Embedding(num_embeddings=2048, dims=config.embed_dim)
        self.drop = nn.Dropout(p=0.0)
        # self.h should be 8 times the block below (or config.num_blocks times)
        self.h = [GPTNeoBlock(attention_dims=config.attention_dims, mlp_dims=config.mlp_dims) for _ in config.num_blocks]
        self.single_block = GPTNeoBlock(attention_dims=config.attention_dims, mlp_dims=config.mlp_dims)
        self.ln_f = nn.LayerNorm(dims=(64,), eps=1e-05, affine=True)


    def __call__(self, x, mask=None, cache=None):
        # Token Embeddings
        input_embeddings = self.wte(x)
        
        # Positional Embeddings
        position_ids = mx.arange(0, x.shape[1]).expand_dims(0).repeat(x.shape[0], axis=0)
        position_embeddings = self.wpe(position_ids)

        # Combine Token and Positional Embeddings
        hidden_states = input_embeddings + position_embeddings
        hidden_states = self.drop(hidden_states)

        # Initialize cache for transformer blocks if needed
        new_caches = []

        for i, block in enumerate(self.h):
            if cache is not None:
                hidden_states, new_cache = block(hidden_states, mask=mask, cache=cache[i])
                new_caches.append(new_cache)
            else:
                hidden_states, new_cache = block(hidden_states, mask=mask)
                new_caches.append(new_cache)

        # Apply final layer normalization
        hidden_states = self.ln_f(hidden_states)

        # Return the final hidden states and the updated caches
        if cache is not None:
            return hidden_states, new_caches
        else:
            return hidden_states


class GPTNeoBlock(nn.Module):
    def __init__(self, attention_dims, mlp_dims):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dims=attention_dims, eps=1e-05, affine=True)
        self.attn = GPTNeoAttention(attention_dims)
        self.ln_2 = nn.LayerNorm(dims=attention_dims, eps=1e-05, affine=True)
        self.mlp = GPTNeoMLP(mlp_dims[0], mlp_dims[1], mlp_dims[2])
                             
    
    def __call__(self, x, mask=None, cache=None):
        residual = x
        x = self.ln_1(x)
        x, cache = self.attn(x, mask=mask, cache=cache)
        residual = x + residual
        x = self.ln_2(residual)
        x = self.mlp(x)
        x = x + residual

        return x, cache
        

class GPTNeoAttention(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.attention = GPTNeoSelfAttention(dims)
    
    
    def __call__(self, x, mask, cache):
        return self.attention(x, mask=mask, cache=cache)
    
    
class GPTNeoSelfAttention(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.attn_dropout = nn.Dropout(p=0.0)
        self.resid_dropout = nn.Dropout(p=0.0)
        self.k_proj = nn.Linear(input_dims=dims, output_dims=dims, bias=False)
        self.v_proj = nn.Linear(input_dims=dims, output_dims=dims, bias=False)
        self.q_proj = nn.Linear(input_dims=dims, output_dims=dims, bias=False)
        self.out_proj = nn.Linear(input_dims=dims, output_dims=dims, bias=True)
        self.scale = 1 / math.sqrt(dims)
        
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        
        B, L, D = x.shape
        seq_length, dim = x.size()
        
        
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)


        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)


        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)


        if mask is not None:
            scores = scores + mask


        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        v_hat = (scores @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.c_proj(v_hat), (k, v)


class GPTNeoMLP(nn.Module):
    def __init__(self, dims_fc, dims_proj, output_dim):
        super().__init__()
        self.c_fc = nn.Linear(input_dims=dims_fc, output_dims=dims_proj)
        self.c_proj = nn.Linear(input_dims=dims_proj, output_dims=output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.0)
        
    
    def __call__(self,
                 x: mx.array
                ):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

"""
Offline SVD Calibration and Closed-Form Ridge Regression for Zero Cold-Start.

Collects activation covariance on WMT22-24 calibration data, computes closed-form
optimal router keys and 16D additive stream weights, and sets parafoveal base blocks.
"""

from __future__ import annotations

import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from foveal_indexer.foveal_llm_layer import FovealSparseSwiGLU, FovealSparseAttentionLayer


@torch.no_grad()
def calibrate_model_svd(
    model: nn.Module,
    calib_tokens: List[torch.Tensor],
    device: torch.device,
    block_size_mlp: int = 32,
    active_blocks_mlp: int = 48, # 48 blocks = 25.0% uniform active
    base_blocks_mlp: int = 0,    # 0 = pure uniform Top-K
    chunk_size_attn: int = 8,
    active_blocks_attn: int = 4, # 4 blocks of 8 = 32 dims = 25.0% uniform active
    layers_to_sparsify: Optional[List[int]] = None,
) -> nn.Module:
    """
    Replaces dense Attention and MLP layers with calibrated Foveal Sparse layers.
    """
    total_layers = len(model.model.layers)
    if layers_to_sparsify is None:
        # Uniform: all 32 layers
        layers_to_sparsify = list(range(total_layers))

    mlp_pct = (active_blocks_mlp / (model.config.intermediate_size // block_size_mlp)) * 100
    attn_pct = (active_blocks_attn / 16) * 100
    print(f"\n--- Initializing Uniform Foveal Sparsification on ALL {len(layers_to_sparsify)} Layers ---")
    print(f"MLP (Chunk 32): {active_blocks_mlp}/192 active blocks ({mlp_pct:.1f}% compute)")
    print(f"Attn (Chunk 8): {active_blocks_attn}/16 active blocks per head ({attn_pct:.1f}% compute)")
    print(f"LM Head: 100% Dense and Untouched")

    # 1. Collect true dense input activations from PURE DENSE MODEL first!
    print("\n--- Running Offline SVD & Ridge Regression on Dense Teacher Activations ---")
    activations_by_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in layers_to_sparsify}
    hooks = []
    for l_idx in layers_to_sparsify:
        def make_hook(idx):
            def hook_fn(m, inp, out):
                if len(activations_by_layer[idx]) < 16:
                    activations_by_layer[idx].append(inp[0].detach().cpu())
            return hook_fn
        h = model.model.layers[l_idx].mlp.register_forward_hook(make_hook(l_idx))
        hooks.append(h)

    model.eval()
    print(f"Passing {min(len(calib_tokens), 16)} calibration batches through pure dense model...")
    for batch in tqdm(calib_tokens[:16]):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn_mask)

    for h in hooks:
        h.remove()

    # 2. Now replace dense layers with properly calibrated Foveal Sparse layers
    print("Replacing layers and solving closed-form Ridge Regression for routers...")
    for l_idx in tqdm(layers_to_sparsify):
        layer = model.model.layers[l_idx]
        
        # Replace and calibrate SwiGLU MLP
        dense_mlp = layer.mlp
        sparse_mlp = FovealSparseSwiGLU.from_dense_mlp(
            dense_mlp,
            block_size=block_size_mlp,
            active_blocks=active_blocks_mlp,
            base_blocks=base_blocks_mlp,
            index_dim=16,
        )
        if len(activations_by_layer[l_idx]) > 0:
            acts = torch.cat(activations_by_layer[l_idx], dim=1).reshape(-1, sparse_mlp.hidden_dim)
            sparse_mlp.calibrate_svd_and_base(acts[:2048].to(device))

        layer.mlp = sparse_mlp.to(device=device, dtype=layer.input_layernorm.weight.dtype)

        # Replace Self Attention
        dense_attn = layer.self_attn
        sparse_attn = FovealSparseAttentionLayer.from_dense_attention(
            dense_attn,
            chunk_size=chunk_size_attn,
            active_blocks_per_head=active_blocks_attn,
        ).to(device=device, dtype=layer.input_layernorm.weight.dtype)
        layer.self_attn = sparse_attn

    print("Zero cold-start calibration complete!")
    return model

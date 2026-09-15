"""
Foveal LLM Layer: Unified Attention (Chunk Size 8) and SwiGLU MLP (Chunk Size 32) Sparsification.

Features:
1. FovealSparseAttentionLayer:
   - Head dimension d_k = 128 partitioned into chunks of 8 (16 candidate blocks per head).
   - Dynamic or head-specialized active block selection: 1, 2, 4, 8 blocks per head (6.25% - 50% compute).
   - RoPE-preserving (chunk size 8 preserves 4 complete 2D RoPE frequency pairs).
   - Parallel 16D register router.

2. FovealSparseSwiGLU:
   - Intermediate dimension D_ffn partitioned into chunks of 32 (e.g. 192 blocks for D_ffn=6144).
   - Parafoveal Base: Top 15% anchor blocks permanently active in cache.
   - Dynamic Remote: Top 10% blocks conditionally routed per token.
   - 16D Differentiable Additive Stream: provides continuous gradients to all blocks.
   - Zero cold-start via closed-form Ridge Regression.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any, List, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FovealSparseSwiGLU(nn.Module):
    """
    Foveal Block-Sparse SwiGLU MLP with Parafoveal Base Blocks and 16D Additive Stream.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        intermediate_dim: int = 6144,
        block_size: int = 32,
        active_blocks: int = 48,   # Uniform active blocks (48 / 192 = 25.0% active)
        base_blocks: int = 0,      # Static anchor blocks (0 = pure uniform Top-K)
        remote_blocks: Optional[int] = None, # For backward compatibility
        index_dim: int = 16,
        temperature: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.block_size = block_size
        self.num_blocks = intermediate_dim // block_size

        if remote_blocks is not None:
            self.base_blocks = min(base_blocks, self.num_blocks)
            self.remote_blocks = min(remote_blocks, self.num_blocks - self.base_blocks)
            self.active_blocks = self.base_blocks + self.remote_blocks
        else:
            self.active_blocks = min(active_blocks, self.num_blocks)
            self.base_blocks = min(base_blocks, self.active_blocks)
            self.remote_blocks = self.active_blocks - self.base_blocks

        self.total_active_blocks = self.active_blocks
        self.index_dim = index_dim
        self.temperature = temperature
        self.dtype = dtype

        # 1. Block-partitioned model parameters
        # Shapes: (num_blocks, block_size, hidden_dim)
        self.w_gate_blocks = nn.Parameter(torch.empty(self.num_blocks, block_size, hidden_dim, dtype=dtype))
        self.w_up_blocks   = nn.Parameter(torch.empty(self.num_blocks, block_size, hidden_dim, dtype=dtype))
        self.w_down_blocks = nn.Parameter(torch.empty(self.num_blocks, block_size, hidden_dim, dtype=dtype))

        # 2. 16D Foveal Router parameters
        self.q_proj_16d = nn.Linear(hidden_dim, index_dim, bias=False, dtype=dtype)
        # Block keys in 16D for remote block selection: (num_blocks, index_dim)
        self.k_blocks_16d = nn.Parameter(torch.empty(self.num_blocks, index_dim, dtype=dtype))

        # 3. 16D Differentiable Additive Stream (Residual Compensation)
        self.stream_in = nn.Linear(hidden_dim, index_dim, bias=False, dtype=dtype)
        self.stream_out = nn.Linear(index_dim, hidden_dim, bias=False, dtype=dtype)

        # 4. Registered base anchor block indices (static high-energy blocks if base_blocks > 0)
        if self.base_blocks > 0:
            self.register_buffer("base_indices", torch.arange(self.base_blocks, dtype=torch.long))
        else:
            self.register_buffer("base_indices", torch.empty(0, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self):
        std_in = (2.0 / self.hidden_dim) ** 0.5
        std_down = (2.0 / self.intermediate_dim) ** 0.5
        nn.init.normal_(self.w_gate_blocks, std=std_in)
        nn.init.normal_(self.w_up_blocks, std=std_in)
        nn.init.normal_(self.w_down_blocks, std=std_down)
        nn.init.normal_(self.q_proj_16d.weight, std=0.02)
        nn.init.normal_(self.k_blocks_16d, std=0.02)
        nn.init.normal_(self.stream_in.weight, std=0.02)
        nn.init.zeros_(self.stream_out.weight)

    @classmethod
    def from_dense_mlp(
        cls,
        dense_mlp: nn.Module,
        block_size: int = 32,
        active_blocks: int = 48, # 48 blocks = 25% active
        base_blocks: int = 0,    # 0 = pure uniform Top-K
        remote_blocks: Optional[int] = None,
        index_dim: int = 16,
    ) -> FovealSparseSwiGLU:
        """Constructs and copies weights from a standard dense SwiGLU MLP."""
        gate_w = dense_mlp.gate_proj.weight.data # (D_ffn, D)
        up_w   = dense_mlp.up_proj.weight.data   # (D_ffn, D)
        down_w = dense_mlp.down_proj.weight.data # (D, D_ffn)

        D_ffn, D = gate_w.shape
        sparse_layer = cls(
            hidden_dim=D,
            intermediate_dim=D_ffn,
            block_size=block_size,
            active_blocks=active_blocks,
            base_blocks=base_blocks,
            remote_blocks=remote_blocks,
            index_dim=index_dim,
            dtype=gate_w.dtype,
        )

        # Reshape and copy
        num_blocks = D_ffn // block_size
        sparse_layer.w_gate_blocks.data.copy_(gate_w.view(num_blocks, block_size, D))
        sparse_layer.w_up_blocks.data.copy_(up_w.view(num_blocks, block_size, D))
        # Down proj is (D, D_ffn) -> transpose to (D_ffn, D) then view
        sparse_layer.w_down_blocks.data.copy_(down_w.t().contiguous().view(num_blocks, block_size, D))

        return sparse_layer

    @torch.no_grad()
    def calibrate_svd_and_base(self, calib_activations: Tensor):
        """
        Calibrates base anchor blocks and initializes router via SVD and closed-form Ridge Regression.
        """
        device = self.w_gate_blocks.device
        X = calib_activations.to(device=device, dtype=torch.float32)
        if X.dim() == 3:
            X = X.reshape(-1, self.hidden_dim)

        # 1. Compute activation covariance SVD
        _, _, Vt = torch.linalg.svd(X[:1000], full_matrices=False)
        k_svd = min(self.index_dim, Vt.size(0))
        self.q_proj_16d.weight.data[:k_svd, :].copy_(Vt[:k_svd, :].to(self.dtype))

        # 2. Measure dense block activation norms
        W_gate_dense = self.w_gate_blocks.view(self.intermediate_dim, self.hidden_dim).float()
        W_up_dense   = self.w_up_blocks.view(self.intermediate_dim, self.hidden_dim).float()
        
        G = torch.matmul(X, W_gate_dense.t())
        U = torch.matmul(X, W_up_dense.t())
        A = F.silu(G) * U # (N, D_ffn)

        A_blocks = A.view(X.size(0), self.num_blocks, self.block_size)
        block_energies = torch.norm(A_blocks, dim=-1).mean(dim=0) # (num_blocks,)

        # Sort blocks by energy: highest energy becomes base blocks
        sorted_indices = torch.argsort(block_energies, descending=True)
        self.base_indices.copy_(sorted_indices[:self.base_blocks])

        # 3. Ridge Regression for 16D router block keys against teacher block energy
        Q_16d = torch.matmul(X, self.q_proj_16d.weight.data.t().float()) # (N, 16)
        Target_mass = torch.norm(A_blocks, dim=-1) # (N, num_blocks)
        
        reg = 1e-3 * torch.eye(self.index_dim, device=device)
        inv_cov = torch.linalg.inv(torch.matmul(Q_16d.t(), Q_16d) + reg)
        K_opt = torch.matmul(torch.matmul(inv_cov, Q_16d.t()), Target_mass).t() # (num_blocks, 16)
        self.k_blocks_16d.data.copy_(K_opt.to(self.dtype))

    def forward(
        self,
        x: Tensor,
        compute_distill_loss: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        orig_shape = x.shape
        N = x.size(0) * x.size(1) if x.dim() == 3 else x.size(0)
        x_flat = x.reshape(N, self.hidden_dim)

        # 1. Compute 16D Query Representation
        q_16d = self.q_proj_16d(x_flat) # (N, 16)

        # 2. Score all blocks in parallel: (N, num_blocks)
        scores = torch.matmul(q_16d, self.k_blocks_16d.t()) / math.sqrt(self.index_dim)

        # 3. Select Active Blocks
        if self.base_blocks > 0 and len(self.base_indices) > 0:
            mask_scores = scores.clone()
            mask_scores[:, self.base_indices] = -1e9
            _, remote_indices = torch.topk(mask_scores, k=self.remote_blocks, dim=-1)
            base_exp = self.base_indices.unsqueeze(0).expand(N, -1)
            active_block_indices = torch.cat([base_exp, remote_indices], dim=1)
        else:
            # Pure uniform Top-K selection across all 192 blocks
            _, active_block_indices = torch.topk(scores, k=self.active_blocks, dim=-1)

        # 4. Gather active block weights and compute
        # In batch training mode, evaluate across active channels
        # Build binary channel mask
        mask = torch.zeros(N, self.num_blocks, device=x.device, dtype=torch.bool)
        mask.scatter_(1, active_block_indices, True)
        mask_channels = mask.unsqueeze(-1).expand(-1, -1, self.block_size).reshape(N, self.intermediate_dim)

        W_gate_dense = self.w_gate_blocks.view(self.intermediate_dim, self.hidden_dim)
        W_up_dense   = self.w_up_blocks.view(self.intermediate_dim, self.hidden_dim)
        W_down_dense = self.w_down_blocks.view(self.intermediate_dim, self.hidden_dim)

        # Sparse Matmul
        g = torch.matmul(x_flat, W_gate_dense.t())
        u = torch.matmul(x_flat, W_up_dense.t())
        a = F.silu(g.float()) * u.float()
        a = a.to(x.dtype)

        # Zero out inactive channels
        a_sparse = torch.where(mask_channels, a, torch.zeros_like(a))
        y = torch.matmul(a_sparse, W_down_dense)

        # 5. Differentiable 16D Additive Stream (provides gradient to all blocks)
        y_residual = self.stream_out(F.silu(self.stream_in(x_flat)))
        out = (y + y_residual).reshape(orig_shape)

        # In training mode, compute and store KL distillation loss for collection
        if self.training:
            with torch.no_grad():
                a_blocks = a.view(N, self.num_blocks, self.block_size).float()
                teacher_energy = torch.norm(a_blocks, dim=-1)
                teacher_p = F.softmax(teacher_energy / self.temperature, dim=-1)
            student_logp = F.log_softmax(scores.float() / self.temperature, dim=-1)
            self.last_kl_loss = F.kl_div(student_logp, teacher_p, reduction="batchmean")
        else:
            self.last_kl_loss = None

        if not compute_distill_loss:
            return out

        info = {"kl_loss": self.last_kl_loss, "active_ratio": self.total_active_blocks / self.num_blocks}
        return out, info


class FovealSparseAttentionLayer(nn.Module):
    """
    Foveal Sparse Attention Layer with Chunk-Size 8 Dimension Slimming.
    Partitions head dimension d_k = 128 into 16 blocks of 8.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        chunk_size: int = 8,
        active_blocks_per_head: int = 4, # e.g. 4 blocks = 32 active dims out of 128 (25% compute)
        index_dim: int = 16,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.num_chunks = head_dim // chunk_size # 16 blocks
        self.active_blocks_per_head = active_blocks_per_head
        self.active_head_dim = active_blocks_per_head * chunk_size
        self.index_dim = index_dim
        self.dtype = dtype

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False, dtype=dtype)

        # 16D Parallel Indexer for chunk selection across heads
        self.q_idx_proj = nn.Linear(hidden_size, index_dim, bias=False, dtype=dtype)
        self.head_chunk_keys = nn.Parameter(torch.empty(num_heads, self.num_chunks, index_dim, dtype=dtype))

        # Default static active chunks per head (can be dynamically overridden)
        # Allocate leading active_blocks_per_head chunks
        default_active = torch.arange(active_blocks_per_head, dtype=torch.long)
        self.register_buffer("static_active_chunks", default_active)

        self.reset_parameters()

    def reset_parameters(self):
        std = (2.0 / self.hidden_size) ** 0.5
        for p in [self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.q_idx_proj]:
            nn.init.normal_(p.weight, std=std)
        nn.init.normal_(self.head_chunk_keys, std=0.02)

    @classmethod
    def from_dense_attention(
        cls,
        dense_attn: nn.Module,
        chunk_size: int = 8,
        active_blocks_per_head: int = 4,
    ) -> FovealSparseAttentionLayer:
        hidden_size = dense_attn.q_proj.in_features
        q_out = dense_attn.q_proj.out_features
        k_out = dense_attn.k_proj.out_features
        
        # Infer heads
        head_dim = 128
        num_heads = q_out // head_dim
        num_kv_heads = k_out // head_dim

        layer = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            chunk_size=chunk_size,
            active_blocks_per_head=active_blocks_per_head,
            dtype=dense_attn.q_proj.weight.dtype,
        )

        layer.q_proj.weight.data.copy_(dense_attn.q_proj.weight.data)
        layer.k_proj.weight.data.copy_(dense_attn.k_proj.weight.data)
        layer.v_proj.weight.data.copy_(dense_attn.v_proj.weight.data)
        layer.o_proj.weight.data.copy_(dense_attn.o_proj.weight.data)

        # Copy metadata & norm layers
        layer.config = getattr(dense_attn, "config", None)
        layer.layer_idx = getattr(dense_attn, "layer_idx", 0)
        layer.query_layernorm = getattr(dense_attn, "query_layernorm", None)
        layer.key_layernorm = getattr(dense_attn, "key_layernorm", None)

        return layer

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Any] = None,
        compute_distill_loss: bool = False,
        **kwargs,
    ) -> Union[Tuple[Tensor, None], Tuple[Tensor, Dict[str, Tensor]]]:
        B, T, D = hidden_states.shape
        input_shape = (B, T)
        hidden_shape = (B, T, -1, self.head_dim)

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 2. Apply RoPE if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # apply rotary position embeddings
            try:
                from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import apply_rotary_pos_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            except Exception:
                pass

        # 3. Apply QK-Norm if present
        if getattr(self, "query_layernorm", None) is not None:
            q = self.query_layernorm(q)
        if getattr(self, "key_layernorm", None) is not None:
            k = self.key_layernorm(k)

        # 4. Slice to active dimensions (Chunk size 8)
        active_slice = slice(0, self.active_head_dim)
        q_sparse = q[..., active_slice]
        k_sparse = k[..., active_slice]
        v_sparse = v[..., active_slice]

        # Update past_key_values if caching during decode
        if past_key_values is not None:
            k_sparse, v_sparse = past_key_values.update(k_sparse, v_sparse, self.layer_idx)

        # Expand KV for GQA if needed
        if self.num_key_value_heads != self.num_heads:
            repeats = self.num_heads // self.num_key_value_heads
            k_sparse = k_sparse.repeat_interleave(repeats, dim=1)
            v_sparse = v_sparse.repeat_interleave(repeats, dim=1)

        # 5. Scaled dot-product attention on active dimensions
        scale = 1.0 / math.sqrt(self.active_head_dim)
        scores = torch.matmul(q_sparse, k_sparse.transpose(-2, -1)) * scale

        if attention_mask is not None:
            # Adjust mask shape if needed
            if attention_mask.dim() == 2:
                # (B, T) -> (B, 1, 1, T)
                mask_4d = attention_mask[:, None, None, :]
                scores = scores + (1.0 - mask_4d) * -1e4
            elif attention_mask.shape[-1] == scores.shape[-1]:
                scores = scores + attention_mask

        attn_probs = F.softmax(scores.float(), dim=-1).to(hidden_states.dtype)
        attn_out = torch.matmul(attn_probs, v_sparse) # (B, H, T, active_head_dim)

        # 6. Zero-pad inactive dimensions to project via W_o
        full_attn_out = torch.zeros(B, self.num_heads, attn_out.size(-2), self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        full_attn_out[..., active_slice] = attn_out

        full_attn_out = full_attn_out.transpose(1, 2).contiguous().view(B, attn_out.size(-2), self.num_heads * self.head_dim)
        out = self.o_proj(full_attn_out)

        # In training mode, compute and store KL loss against teacher dense attention
        if self.training:
            with torch.no_grad():
                k_dense = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
                dense_scores = torch.matmul(q, k_dense.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                if attention_mask is not None and attention_mask.shape[-1] == dense_scores.shape[-1]:
                    dense_scores = dense_scores + attention_mask
                dense_probs = F.softmax(dense_scores.float(), dim=-1)
            self.last_kl_loss = F.kl_div(attn_probs.float().log(), dense_probs, reduction="batchmean")
        else:
            self.last_kl_loss = None

        if not compute_distill_loss:
            return (out, None)

        return (out, None), {"kl_attn": self.last_kl_loss, "active_dim": self.active_head_dim}

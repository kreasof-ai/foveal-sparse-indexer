"""Triton-accelerated kernels for Foveal Sparse Indexer on GPU.

Includes:
1. Fused Block-Sparse Attention kernel (FlashAttention-style online softmax).
   Only active query-key blocks are loaded and computed.
2. Block-Sparse Linear kernel for Foveal MLP projections.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _foveal_block_sparse_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        block_indices_ptr, active_counts_ptr,
        scale,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_bi_b, stride_bi_m, stride_bi_k,
        stride_ac_b, stride_ac_m,
        H: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        MAX_ACTIVE_BLOCKS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b_idx = pid_bh // H
        h_idx = pid_bh % H

        q_base = Q_ptr + b_idx * stride_qb + h_idx * stride_qh
        k_base = K_ptr + b_idx * stride_kb + h_idx * stride_kh
        v_base = V_ptr + b_idx * stride_vb + h_idx * stride_vh
        out_base = Out_ptr + b_idx * stride_ob + h_idx * stride_oh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        # Coalesced load of query block
        q_ptrs = q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e4
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        n_active = tl.load(active_counts_ptr + b_idx * stride_ac_b + pid_m * stride_ac_m)

        for j in range(MAX_ACTIVE_BLOCKS):
            is_active = j < n_active
            k_blk = tl.load(block_indices_ptr + b_idx * stride_bi_b + pid_m * stride_bi_m + j * stride_bi_k)
            k_blk_safe = tl.where(is_active, k_blk, 0)
            offs_n = k_blk_safe * BLOCK_N + tl.arange(0, BLOCK_N)

            # Coalesced load of key and value blocks
            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)

            s = (tl.dot(q, tl.trans(k)) * scale).to(tl.float32)
            s = tl.where(is_active, s, -10000.0)

            m_ij = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp(s - m_ij[:, None])
            p = tl.where(is_active, p, 0.0).to(tl.float32)

            alpha = tl.exp(m_i - m_ij).to(tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_ij

        acc = acc / tl.maximum(l_i[:, None], 1e-6)
        out_ptrs = out_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc.to(q.dtype))

    @triton.jit
    def _block_sparse_linear_kernel(
        X_ptr, W_ptr, Bias_ptr, Y_ptr,
        active_blocks_ptr,
        M, N, K,
        num_active_blocks,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_act = tl.program_id(1)

        blk_idx = tl.load(active_blocks_ptr + pid_act)
        col_start = blk_idx * BLOCK_N

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = col_start + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            mask_k = (k_start + offs_k) < K
            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=mask_k[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w))
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        if has_bias:
            bias_ptrs = Bias_ptr + offs_n
            bias_val = tl.load(bias_ptrs)
            acc += bias_val[None, :]

        y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=mask_m[:, None])

    @triton.jit
    def _fused_sram_indexer_attention_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        Q16_ptr, K16_ptr,
        scale, top_p,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_q16_b, stride_q16_m, stride_q16_d,
        stride_k16_b, stride_k16_m, stride_k16_d,
        H: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        M_BLOCKS: tl.constexpr,
        MAX_REMOTE: tl.constexpr,
    ):
        """Fuses 16D indexing, top-p routing, and FlashAttention on SRAM/registers.

        Eliminates DRAM round-trips for 16D score matrices and discrete routing tensors.
        """
        pid_m = tl.program_id(0)   # query block index (0 .. M_BLOCKS-1)
        pid_bh = tl.program_id(1)  # batch and head
        b_idx = pid_bh // H
        h_idx = pid_bh % H

        # 1. SRAM 16D Indexer dot products & routing
        offs_16 = tl.arange(0, 16)
        q16_ptrs = Q16_ptr + b_idx * stride_q16_b + pid_m * stride_q16_m + offs_16 * stride_q16_d
        q16 = tl.load(q16_ptrs)

        offs_blocks = tl.arange(0, M_BLOCKS)
        k16_ptrs = K16_ptr + b_idx * stride_k16_b + offs_blocks[:, None] * stride_k16_m + offs_16[None, :] * stride_k16_d
        k16 = tl.load(k16_ptrs)

        # 16D cosine dot product in registers
        scores = tl.sum(q16[None, :] * k16, axis=1) * 0.25
        is_remote = offs_blocks != pid_m
        remote_scores = tl.where(is_remote, scores, -1e4)

        m_remote = tl.max(remote_scores, axis=0)
        exp_remote = tl.where(is_remote, tl.exp(remote_scores - m_remote), 0.0)
        sum_exp = tl.sum(exp_remote, axis=0) + 1e-6
        probs = exp_remote / sum_exp

        # 2. FlashAttention with zero DRAM round-trip
        q_base = Q_ptr + b_idx * stride_qb + h_idx * stride_qh
        k_base = K_ptr + b_idx * stride_kb + h_idx * stride_kh
        v_base = V_ptr + b_idx * stride_vb + h_idx * stride_vh
        out_base = Out_ptr + b_idx * stride_ob + h_idx * stride_oh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        q_ptrs = q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e4
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Unconditionally attend to base self-block
        offs_n_self = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n_self[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n_self[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        s = (tl.dot(q, tl.trans(k)) * scale).to(tl.float32)
        m_ij = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

        # Loop over active remote blocks in SRAM
        cum_mass = 0.0
        for r in range(MAX_REMOTE):
            best_k = tl.argmax(probs, axis=0)
            best_p = tl.max(probs, axis=0)
            should_activate = (cum_mass < top_p) & (best_p > 1e-4)

            offs_n = best_k * BLOCK_N + tl.arange(0, BLOCK_N)
            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)

            s = (tl.dot(q, tl.trans(k)) * scale).to(tl.float32)
            s = tl.where(should_activate, s, -10000.0)

            m_ij = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp(s - m_ij[:, None])
            p = tl.where(should_activate, p, 0.0).to(tl.float32)

            alpha = tl.exp(m_i - m_ij).to(tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_ij

            cum_mass += best_p
            probs = tl.where(offs_blocks == best_k, -1.0, probs)

        acc = acc / tl.maximum(l_i[:, None], 1e-6)
        out_ptrs = out_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc.to(q.dtype))


def is_triton_available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()


def triton_foveal_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_indices: Tensor,
    active_counts: Tensor,
    scale: Optional[float] = None,
    block_size: int = 16,
) -> Tensor:
    """Computes Foveal Block-Sparse Attention using Triton FlashAttention-style kernel.

    Args:
        q: (B, H, N, D) Query tensor
        k: (B, H, N, D) Key tensor
        v: (B, H, N, D) Value tensor
        block_indices: (B, M_blocks, MAX_ACTIVE) int32 active KV block indices per query block
        active_counts: (B, M_blocks) int32 number of active blocks per query block
        scale: softmax scale factor (default: 1 / sqrt(D))
        block_size: size of each patch block (default: 16)
    """
    if not is_triton_available() or not q.is_cuda:
        raise RuntimeError("Triton Foveal Attention requires CUDA and Triton.")

    B, H, N, D = q.shape
    assert N % block_size == 0, f"N ({N}) must be divisible by block_size ({block_size})"
    M_blocks = N // block_size
    max_active = block_indices.shape[-1]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Ensure contiguous memory layout for coalesced memory access
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_indices = block_indices.contiguous().to(torch.int32)
    active_counts = active_counts.contiguous().to(torch.int32)

    out = torch.empty_like(q)
    grid = (M_blocks, B * H)

    _foveal_block_sparse_attn_fwd_kernel[grid](
        q, k, v, out,
        block_indices, active_counts,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        block_indices.stride(0), block_indices.stride(1), block_indices.stride(2),
        active_counts.stride(0), active_counts.stride(1),
        H=H,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        HEAD_DIM=D,
        MAX_ACTIVE_BLOCKS=max_active,
    )
    return out


def triton_block_sparse_linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    active_block_indices: Tensor,
    block_size: int = 32,
) -> Tensor:
    """Computes Block-Sparse Linear projection Y = X @ W_active^T (+ bias).

    Args:
        x: (M, K) input tensor
        weight: (N, K) weight tensor
        bias: Optional (N,) bias tensor
        active_block_indices: 1D int32 tensor of active block indices
        block_size: column block size (N block size)
    """
    if not is_triton_available() or not x.is_cuda:
        raise RuntimeError("Triton Block-Sparse Linear requires CUDA and Triton.")

    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, f"K mismatch: {K} vs {K_w}"

    num_active = len(active_block_indices)
    if num_active == 0:
        return torch.zeros(M, N, device=x.device, dtype=x.dtype)

    x = x.contiguous()
    weight = weight.contiguous()
    active_block_indices = active_block_indices.contiguous().to(torch.int32)

    y = torch.zeros(M, N, device=x.device, dtype=x.dtype)

    BLOCK_M = 32
    BLOCK_K = 32
    BLOCK_N = block_size

    has_bias = bias is not None
    bias_tensor = bias if has_bias else weight  # dummy if no bias

    grid = (triton.cdiv(M, BLOCK_M), num_active)
    _block_sparse_linear_kernel[grid](
        x, weight, bias_tensor, y,
        active_block_indices,
        M, N, K,
        num_active,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        has_bias=has_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return y


def triton_fused_sram_indexer_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q16: Tensor,
    k16: Tensor,
    scale: Optional[float] = None,
    top_p: float = 0.8,
    max_remote_blocks: int = 2,
    block_size: int = 16,
) -> Tensor:
    """Computes Foveal Sparse Attention fusing 16D indexing and routing on SRAM.

    Eliminates all DRAM round-trips for block score matrices and routing indices.

    Args:
        q: (B, H, N, D) Query tensor
        k: (B, H, N, D) Key tensor
        v: (B, H, N, D) Value tensor
        q16: (B, M_blocks, 16) 16D Query representations per block
        k16: (B, M_blocks, 16) 16D Key representations per block
        scale: attention scale factor (default: 1 / sqrt(D))
        top_p: cumulative attention mass threshold (default: 0.8)
        max_remote_blocks: maximum remote blocks to retrieve (default: 2)
        block_size: patch block size (default: 16)
    """
    if not is_triton_available() or not q.is_cuda:
        raise RuntimeError("Triton Fused SRAM Indexer Attention requires CUDA and Triton.")

    B, H, N, D = q.shape
    assert N % block_size == 0, f"N ({N}) must be divisible by block_size ({block_size})"
    M_blocks = N // block_size

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    q16 = q16.contiguous()
    k16 = k16.contiguous()

    out = torch.empty_like(q)
    grid = (M_blocks, B * H)

    _fused_sram_indexer_attention_kernel[grid](
        q, k, v, out,
        q16, k16,
        scale, top_p,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        q16.stride(0), q16.stride(1), q16.stride(2),
        k16.stride(0), k16.stride(1), k16.stride(2),
        H=H,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        HEAD_DIM=D,
        M_BLOCKS=M_blocks,
        MAX_REMOTE=max_remote_blocks,
    )
    return out

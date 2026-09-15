"""
Triton Fused Pattention Kernel with Arbitrary Activation Functions.

Computes Token-Parameter Attention with fused non-softmax activations:
  - SwiGLU: (SiLU(X @ K_gate^T) * (X @ K_up^T)) @ V
  - GeLU:   GeLU(X @ K^T) @ V
  - SiLU:   SiLU(X @ K^T) @ V
  - ReLU / ReLU^2: max(0, X @ K^T)^2 @ V

Executes entirely in GPU SRAM registers without writing intermediate activation
matrices to DRAM, supporting arbitrary parameter block sparsity.
"""

from typing import Optional, Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(
        G_ptr, U_ptr, Out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        g = tl.load(G_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(U_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        sig = tl.sigmoid(g)
        silu = g * sig
        act = silu * u

        tl.store(Out_ptr + offsets, act.to(Out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        dOut_ptr, G_ptr, U_ptr, dG_ptr, dU_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        dout = tl.load(dOut_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(G_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(U_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        sig = tl.sigmoid(g)
        silu = g * sig

        du = dout * silu
        dsilu = sig * (1.0 + g * (1.0 - sig))
        dg = dout * u * dsilu

        tl.store(dG_ptr + offsets, dg.to(dG_ptr.dtype.element_ty), mask=mask)
        tl.store(dU_ptr + offsets, du.to(dU_ptr.dtype.element_ty), mask=mask)

    class TritonFusedSwiGLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, g: Tensor, u: Tensor) -> Tensor:
            g_c = g.contiguous()
            u_c = u.contiguous()
            ctx.save_for_backward(g_c, u_c)
            out = torch.empty_like(g_c)
            n_elements = g_c.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            _swiglu_fwd_kernel[grid](g_c, u_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
            return out

        @staticmethod
        def backward(ctx, dout: Tensor):
            g, u = ctx.saved_tensors
            dout_c = dout.contiguous()
            dg = torch.empty_like(g)
            du = torch.empty_like(u)
            n_elements = g.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            _swiglu_bwd_kernel[grid](dout_c, g, u, dg, du, n_elements, BLOCK_SIZE=BLOCK_SIZE)
            return dg, du

    def triton_fused_swiglu(g: Tensor, u: Tensor) -> Tensor:
        """Fused SwiGLU: SiLU(g) * u evaluated in SRAM with exact backward gradients."""
        return TritonFusedSwiGLUFunction.apply(g, u)

else:
    def triton_fused_swiglu(g: Tensor, u: Tensor) -> Tensor:
        return F.silu(g) * u


if HAS_TRITON:
    @triton.jit
    def _pattention_fused_kernel(
        X_ptr, K_gate_ptr, K_up_ptr, V_ptr, Out_ptr,
        active_blocks_ptr,
        M, N_p, K, D_out,
        num_active_blocks,
        stride_xm, stride_xk,
        stride_kgn, stride_kgk,
        stride_kun, stride_kuk,
        stride_vn, stride_vd,
        stride_om, stride_od,
        HAS_K_UP: tl.constexpr,
        IS_SPARSE: tl.constexpr,
        ACTIVATION_CODE: tl.constexpr,  # 0: SwiGLU, 1: SiLU, 2: GeLU, 3: ReLU, 4: ReLU2
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_d = offs_d < D_out

        # Output accumulator in SRAM registers
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        # Number of iteration tiles
        num_tiles = num_active_blocks if IS_SPARSE else tl.cdiv(N_p, BLOCK_N)

        for tile_idx in range(0, num_tiles):
            if IS_SPARSE:
                blk_idx = tl.load(active_blocks_ptr + tile_idx)
                n_start = blk_idx * BLOCK_N
            else:
                n_start = tile_idx * BLOCK_N

            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_p

            # 1. Compute scores_gate = X @ K_gate^T
            scores_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            scores_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_start in range(0, K, BLOCK_K):
                mask_k = (k_start + offs_k) < K
                x_ptrs = X_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :] * stride_xk
                kg_ptrs = K_gate_ptr + offs_n[None, :] * stride_kgn + (k_start + offs_k)[:, None] * stride_kgk

                x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                kg_tile = tl.load(kg_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                scores_gate += tl.dot(x_tile, kg_tile)

                if HAS_K_UP:
                    ku_ptrs = K_up_ptr + offs_n[None, :] * stride_kun + (k_start + offs_k)[:, None] * stride_kuk
                    ku_tile = tl.load(ku_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                    scores_up += tl.dot(x_tile, ku_tile)

            # 2. Evaluate Non-Softmax Activation Function in SRAM registers
            act = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if ACTIVATION_CODE == 0:
                # SwiGLU: SiLU(scores_gate) * scores_up
                sig = tl.sigmoid(scores_gate)
                act = (scores_gate * sig) * scores_up
            elif ACTIVATION_CODE == 1:
                # SiLU: s * sigmoid(s)
                sig = tl.sigmoid(scores_gate)
                act = scores_gate * sig
            elif ACTIVATION_CODE == 2:
                # GeLU approximation: 0.5 * s * (1.0 + tanh(sqrt(2/pi) * (s + 0.044715 * s^3)))
                # or erf approximation
                act = scores_gate * 0.5 * (1.0 + tl.sin(scores_gate * 0.7978845608))
            elif ACTIVATION_CODE == 3:
                # ReLU: max(0, s)
                act = tl.maximum(0.0, scores_gate)
            elif ACTIVATION_CODE == 4:
                # Squared ReLU: max(0, s)^2
                pos = tl.maximum(0.0, scores_gate)
                act = pos * pos

            # 3. Multiply by V and accumulate into acc (all in registers)
            v_ptrs = V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

            acc += tl.dot(act.to(v_tile.dtype), v_tile)

        # 4. Store accumulated output to DRAM
        out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


def triton_pattention(
    x: Tensor,
    k_gate: Tensor,
    v: Tensor,
    k_up: Optional[Tensor] = None,
    active_blocks: Optional[Tensor] = None,
    activation: Literal["swiglu", "silu", "gelu", "relu", "relu2"] = "swiglu",
    block_n: int = 64,
) -> Tensor:
    """
    Executes Fused Non-Softmax Pattention on GPU via Triton.

    Args:
        x: (M, K) Input query tensor
        k_gate: (N_p, K) Parameter gate keys
        v: (N_p, D_out) Parameter values
        k_up: Optional (N_p, K) Parameter up keys for SwiGLU
        active_blocks: Optional (num_active,) int32 tensor of active block indices
        activation: Activation function string ('swiglu', 'silu', 'gelu', 'relu', 'relu2')
        block_n: Block tile size along parameter dimension N_p
    """
    if not HAS_TRITON or not x.is_cuda:
        # Fallback to PyTorch
        return _pytorch_pattention_fallback(x, k_gate, v, k_up, active_blocks, activation, block_n)

    orig_shape = x.shape
    M = x.reshape(-1, x.shape[-1]).shape[0]
    K = x.shape[-1]
    N_p, _ = k_gate.shape
    _, D_out = v.shape

    x_2d = x.reshape(M, K).contiguous()
    k_gate = k_gate.contiguous()
    v = v.contiguous()
    has_k_up = k_up is not None
    k_up_tensor = k_up.contiguous() if has_k_up else k_gate

    is_sparse = active_blocks is not None
    num_active = len(active_blocks) if is_sparse else 0
    active_blocks_tensor = active_blocks.contiguous().to(torch.int32) if is_sparse else torch.empty(0, dtype=torch.int32, device=x.device)

    act_map = {"swiglu": 0, "silu": 1, "gelu": 2, "relu": 3, "relu2": 4}
    act_code = act_map.get(activation.lower(), 0)

    out = torch.empty((M, D_out), device=x.device, dtype=x.dtype)

    BLOCK_M = 32
    BLOCK_N = block_n
    BLOCK_K = 32
    BLOCK_D = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D_out, BLOCK_D))
    _pattention_fused_kernel[grid](
        x_2d, k_gate, k_up_tensor, v, out,
        active_blocks_tensor,
        M, N_p, K, D_out,
        num_active,
        x_2d.stride(0), x_2d.stride(1),
        k_gate.stride(0), k_gate.stride(1),
        k_up_tensor.stride(0), k_up_tensor.stride(1),
        v.stride(0), v.stride(1),
        out.stride(0), out.stride(1),
        HAS_K_UP=has_k_up,
        IS_SPARSE=is_sparse,
        ACTIVATION_CODE=act_code,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
    )

    if len(orig_shape) > 2:
        return out.reshape(*orig_shape[:-1], D_out)
    return out


def _pytorch_pattention_fallback(x, k_gate, v, k_up, active_blocks, activation, block_n):
    """PyTorch reference implementation."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    if active_blocks is not None:
        offsets = torch.arange(block_n, device=x.device)
        indices = (active_blocks.view(-1, 1) * block_n + offsets).flatten()
        k_gate = k_gate[indices]
        v = v[indices]
        if k_up is not None:
            k_up = k_up[indices]

    scores_g = torch.matmul(x_2d, k_gate.t())
    if activation == "swiglu":
        assert k_up is not None, "k_up required for swiglu"
        scores_u = torch.matmul(x_2d, k_up.t())
        act = F.silu(scores_g) * scores_u
    elif activation == "silu":
        act = F.silu(scores_g)
    elif activation == "gelu":
        act = F.gelu(scores_g)
    elif activation == "relu":
        act = F.relu(scores_g)
    elif activation == "relu2":
        act = F.relu(scores_g) ** 2
    else:
        act = scores_g

    out = torch.matmul(act, v)
    if len(orig_shape) > 2:
        return out.reshape(*orig_shape[:-1], v.shape[-1])
    return out


class TritonSwiGLUPattentionMLP(nn.Module):
    """
    Drop-in SwiGLU MLP replacement using Triton Fused Pattention.

    Reparameterizes SwiGLU as non-softmax parameter attention executed entirely
    in SRAM registers without intermediate DRAM writes.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_param_tokens: int = 6144,
        block_size: int = 64,
        activation: str = "swiglu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_param_tokens = num_param_tokens
        self.block_size = block_size
        self.activation = activation
        self.num_blocks = num_param_tokens // block_size

        self.k_gate = nn.Parameter(torch.empty(num_param_tokens, hidden_dim, dtype=dtype))
        self.k_up = nn.Parameter(torch.empty(num_param_tokens, hidden_dim, dtype=dtype))
        self.v = nn.Parameter(torch.empty(num_param_tokens, hidden_dim, dtype=dtype))

        nn.init.normal_(self.k_gate, std=hidden_dim ** -0.5)
        nn.init.normal_(self.k_up, std=hidden_dim ** -0.5)
        nn.init.normal_(self.v, std=num_param_tokens ** -0.5)

    @classmethod
    def from_dense_mlp(cls, dense_mlp: nn.Module, block_size: int = 64) -> "TritonSwiGLUPattentionMLP":
        """Initializes directly from a pretrained SwiGLU MLP."""
        hidden_dim = dense_mlp.gate_proj.in_features
        intermediate_dim = dense_mlp.gate_proj.out_features
        dtype = dense_mlp.gate_proj.weight.dtype
        device = dense_mlp.gate_proj.weight.device

        mlp = cls(
            hidden_dim=hidden_dim,
            num_param_tokens=intermediate_dim,
            block_size=block_size,
            activation="swiglu",
            dtype=dtype,
        ).to(device)

        mlp.k_gate.data.copy_(dense_mlp.gate_proj.weight.data)
        mlp.k_up.data.copy_(dense_mlp.up_proj.weight.data)
        mlp.v.data.copy_(dense_mlp.down_proj.weight.data.t())
        return mlp

    def forward(self, x: Tensor, active_blocks: Optional[Tensor] = None) -> Tensor:
        return triton_pattention(
            x,
            self.k_gate,
            self.v,
            k_up=self.k_up,
            active_blocks=active_blocks,
            activation=self.activation,
            block_n=self.block_size,
        )


def convert_llm_mlp_to_pattention(
    model: nn.Module,
    layers: Optional[list] = None,
    block_size: int = 64,
    activation: str = "swiglu",
) -> nn.Module:
    """Converts dense MLPs across model layers into TritonSwiGLUPattentionMLP."""
    model_layers = getattr(model, "model", model).layers
    target_layers = layers if layers is not None else list(range(len(model_layers)))

    for idx in target_layers:
        if hasattr(model_layers[idx], "mlp") and not isinstance(model_layers[idx].mlp, TritonSwiGLUPattentionMLP):
            model_layers[idx].mlp = TritonSwiGLUPattentionMLP.from_dense_mlp(
                model_layers[idx].mlp,
                block_size=block_size,
            )

    return model

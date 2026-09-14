"""
Foveal LLM Sparsification Module.

Provides algorithms to profile layer-level and neuron-level redundancy,
construct low-rank SVD residual bridges, prune redundant transformer layers,
and perform fast knowledge distillation adaptation for LLMs (e.g., HunYuanDenseV1 / Hy-MT2).
"""

from typing import List, Tuple, Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDBridge(nn.Module):
    """
    Low-Rank SVD Residual Bridge.
    
    Replaces a sequence of dropped transformer layers with a low-rank residual projection:
        y = x + (x @ V) @ U_sigma
    where V is [hidden_dim, rank] and U_sigma is [rank, hidden_dim].
    This preserves stage-boundary representations with <0.02% parameter overhead.
    """

    def __init__(self, hidden_dim: int, rank: int = 64, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = min(rank, hidden_dim)
        
        # Factorized low-rank weights
        self.down_proj = nn.Parameter(torch.zeros(hidden_dim, self.rank, dtype=dtype))
        self.up_proj = nn.Parameter(torch.zeros(self.rank, hidden_dim, dtype=dtype))
        
        # Initialize to near-zero residual identity
        nn.init.normal_(self.down_proj, std=0.01)
        nn.init.zeros_(self.up_proj)

    @classmethod
    def from_weight(cls, W: torch.Tensor, rank: int = 64, dtype: Optional[torch.dtype] = None) -> "SVDBridge":
        """
        Factorize a full [hidden_dim, hidden_dim] bridge matrix into rank-r SVD components.
        W represents Delta = X @ W^T, so (Delta) = (X @ V) @ (U * S)^T.
        """
        hidden_dim = W.shape[0]
        out_dtype = dtype or W.dtype
        bridge = cls(hidden_dim, rank=rank, dtype=out_dtype)
        
        # SVD on float32 for numerical stability
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        r = min(rank, len(S))
        
        # W = U[:, :r] * S[:r] @ Vh[:r, :]
        # Output delta = X @ W^T = X @ Vh[:r, :].T @ (U[:, :r] * S[:r]).T
        down = Vh[:r, :].T.to(dtype=out_dtype)            # [hidden_dim, r]
        up = (U[:, :r] * S[:r]).T.to(dtype=out_dtype)     # [r, hidden_dim]
        
        bridge.down_proj.data.copy_(down)
        bridge.up_proj.data.copy_(up)
        return bridge

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass compatible with Transformer DecoderLayer signatures.
        """
        # Ensure precision compatibility
        down = self.down_proj.to(hidden_states.dtype)
        up = self.up_proj.to(hidden_states.dtype)
        low_rank = torch.matmul(hidden_states, down)
        delta = torch.matmul(low_rank, up)
        return hidden_states + delta


class FovealBlockSparseMLP(nn.Module):
    """
    Foveal Block-Sparse SwiGLU MLP with 16D Routing & Additive Context Stream.
    
    Partitions the intermediate dimension (e.g. 6,144) into M blocks (e.g. 96 blocks of size 64).
    The 16D Foveal Indexer scores the blocks and selects only the top active blocks per token batch
    (e.g., 2 base blocks + 6 remote blocks = 8 blocks = 512 channels out of 6,144 = 91.7% sparsity).
    
    A differentiable 16D additive context stream provides soft gradient flow to all parameter
    blocks during distillation fine-tuning.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        intermediate_dim: int = 6144,
        block_size: int = 64,
        base_blocks: int = 2,
        max_remote_blocks: int = 6,
        index_dim: int = 16,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.block_size = block_size
        self.num_blocks = intermediate_dim // block_size
        self.base_blocks = base_blocks
        self.max_remote_blocks = max_remote_blocks
        self.index_dim = index_dim

        # Core SwiGLU projections
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=dtype)

        # 16D Foveal Router
        self.q_proj_16d = nn.Linear(hidden_dim, index_dim, bias=False, dtype=dtype)
        nn.init.normal_(self.q_proj_16d.weight, std=0.02)

        # 16D Block Key Representations (one per parameter block)
        self.block_keys_16d = nn.Parameter(torch.empty(self.num_blocks, index_dim, dtype=dtype))
        nn.init.normal_(self.block_keys_16d, std=index_dim ** -0.5)

        # 16D Additive Stream Output Projection
        self.out_proj_16d = nn.Linear(index_dim, hidden_dim, bias=False, dtype=dtype)
        nn.init.normal_(self.out_proj_16d.weight, std=1e-3)

        self.register_buffer("offsets", torch.arange(block_size))

    @classmethod
    def from_dense_mlp(
        cls,
        dense_mlp: nn.Module,
        block_size: int = 64,
        base_blocks: int = 2,
        max_remote_blocks: int = 6,
        index_dim: int = 16,
    ) -> "FovealBlockSparseMLP":
        """Initializes a FovealBlockSparseMLP from an existing dense SwiGLU MLP."""
        hidden_dim = dense_mlp.gate_proj.in_features
        intermediate_dim = dense_mlp.gate_proj.out_features
        dtype = dense_mlp.gate_proj.weight.dtype
        device = dense_mlp.gate_proj.weight.device

        foveal_mlp = cls(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            block_size=block_size,
            base_blocks=base_blocks,
            max_remote_blocks=max_remote_blocks,
            index_dim=index_dim,
            dtype=dtype,
        ).to(device)

        # Copy dense weights
        foveal_mlp.gate_proj.weight.data.copy_(dense_mlp.gate_proj.weight.data)
        foveal_mlp.up_proj.weight.data.copy_(dense_mlp.up_proj.weight.data)
        foveal_mlp.down_proj.weight.data.copy_(dense_mlp.down_proj.weight.data)

        # Initialize 16D block keys from dominant principal components of each block
        gw = dense_mlp.gate_proj.weight.data.float()
        num_blocks = intermediate_dim // block_size
        block_pcas = []
        for b in range(num_blocks):
            blk_w = gw[b * block_size : (b + 1) * block_size]  # [block_size, hidden_dim]
            _, _, v = torch.pca_lowrank(blk_w, q=1)
            block_pcas.append(v.squeeze(1))  # [hidden_dim]
        block_pcas_mat = torch.stack(block_pcas, dim=0).to(dtype=dtype, device=device)

        with torch.no_grad():
            foveal_mlp.block_keys_16d.copy_(foveal_mlp.q_proj_16d(block_pcas_mat))

        return foveal_mlp

    def forward(self, x: torch.Tensor, return_sparsity: bool = False):
        orig_shape = x.shape
        B, L, D = orig_shape[0], orig_shape[1], orig_shape[2]
        N_tokens = B * L
        x_flat = x.reshape(N_tokens, D)

        # 1. 16D cosine routing
        q_16 = F.normalize(self.q_proj_16d(x_flat), dim=-1)
        k_16 = F.normalize(self.block_keys_16d, dim=-1)
        scores = torch.matmul(q_16, k_16.t())  # [N_tokens, num_blocks]

        # 2. Select active blocks (base blocks + top remote blocks)
        mean_scores = scores.mean(dim=0)
        _, top_remote = torch.topk(mean_scores[self.base_blocks:], k=self.max_remote_blocks)
        active_blocks = torch.cat([
            torch.arange(self.base_blocks, device=x.device),
            top_remote + self.base_blocks,
        ])

        # Active channel indices
        active_channels = (active_blocks.view(-1, 1) * self.block_size + self.offsets).flatten()
        num_active_channels = len(active_channels)

        # 3. Strictly compute SwiGLU over active channels
        w_g = self.gate_proj.weight[active_channels]
        w_u = self.up_proj.weight[active_channels]
        w_d = self.down_proj.weight[:, active_channels]

        g = F.silu(F.linear(x, w_g))
        u = F.linear(x, w_u)
        out_active = F.linear(g * u, w_d)

        # 4. 16D Additive Context Stream for gradient flow and background semantic mass
        soft = F.softmax(scores, dim=-1)
        pooled_16d = torch.matmul(soft, self.block_keys_16d)
        add = self.out_proj_16d(pooled_16d).reshape(B, L, D)

        y = out_active + add
        if return_sparsity:
            sparsity = (1.0 - num_active_channels / self.intermediate_dim) * 100.0
            return y, sparsity
        return y


def sparsify_llm_mlps(
    model: nn.Module,
    block_size: int = 64,
    base_blocks: int = 2,
    max_remote_blocks: int = 6,
    index_dim: int = 16,
    layers_to_sparsify: Optional[List[int]] = None,
) -> nn.Module:
    """Replaces dense MLPs across model layers with FovealBlockSparseMLP."""
    layers = getattr(model, "model", model).layers
    total_layers = len(layers)
    target_indices = layers_to_sparsify if layers_to_sparsify is not None else list(range(total_layers))

    for idx in target_indices:
        if hasattr(layers[idx], "mlp"):
            dense_mlp = layers[idx].mlp
            layers[idx].mlp = FovealBlockSparseMLP.from_dense_mlp(
                dense_mlp,
                block_size=block_size,
                base_blocks=base_blocks,
                max_remote_blocks=max_remote_blocks,
                index_dim=index_dim,
            )

    return model


class FovealDynamicSkipMLP(nn.Module):
    """
    Dynamic Foveal SwiGLU MLP with 16D Skip Routing & Residual Compensation Bridge.
    
    Controls execution of the 6,144-dimensional SwiGLU MLP via a 16D Foveal Router:
    - If gate probability >= threshold: executes the full SwiGLU MLP.
    - If gate probability < threshold: BYPASSES the 6,144-dim SwiGLU completely
      (0 FLOPs, 0 weight reads) and executes only a lightweight 16D compensation bridge
      (2048 -> 16 -> 2048).
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        hidden_dim: int = 2048,
        index_dim: int = 16,
        threshold: float = 0.5,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.mlp = base_mlp
        self.hidden_dim = hidden_dim
        self.index_dim = index_dim
        self.threshold = threshold

        # 16D Foveal Router
        self.q_proj_16d = nn.Linear(hidden_dim, index_dim, bias=False, dtype=dtype)
        self.gate = nn.Linear(index_dim, 1, bias=True, dtype=dtype)

        # 16D Residual Compensation Bridge (2048 -> 16 -> 2048)
        self.bridge_in = nn.Linear(hidden_dim, index_dim, bias=False, dtype=dtype)
        self.bridge_out = nn.Linear(index_dim, hidden_dim, bias=False, dtype=dtype)

        # Initialization: start with moderate execution, near-zero bridge output
        nn.init.normal_(self.q_proj_16d.weight, std=0.02)
        nn.init.constant_(self.gate.bias, 0.0)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.bridge_in.weight, std=0.02)
        nn.init.zeros_(self.bridge_out.weight)

        self.skip_calls = 0
        self.exec_calls = 0

    def forward(self, hidden_states: torch.Tensor, return_gate: bool = False):
        q_16 = self.q_proj_16d(hidden_states)
        gate_prob = torch.sigmoid(self.gate(q_16))  # [B, L, 1]
        self.last_gate_prob = gate_prob

        if self.training:
            # Differentiable soft gating during training
            mlp_out = self.mlp(hidden_states)
            bridge_out = self.bridge_out(self.bridge_in(hidden_states))
            y = gate_prob * mlp_out + (1.0 - gate_prob) * bridge_out
            if return_gate:
                return y, gate_prob
            return y

        # Fast hard gating during inference
        mean_gate = gate_prob.mean().item()
        if mean_gate < self.threshold:
            self.skip_calls += 1
            # 6,144-dim SwiGLU is completely bypassed!
            y = self.bridge_out(self.bridge_in(hidden_states))
        else:
            self.exec_calls += 1
            y = self.mlp(hidden_states)

        if return_gate:
            return y, gate_prob
        return y


def sparsify_llm_dynamic_skips(
    model: nn.Module,
    layers_to_skip: Optional[List[int]] = None,
    threshold: float = 0.5,
    index_dim: int = 16,
) -> nn.Module:
    """Wraps candidate layers' MLPs with FovealDynamicSkipMLP."""
    layers = getattr(model, "model", model).layers
    total_layers = len(layers)
    targets = layers_to_skip if layers_to_skip is not None else list(range(20, min(28, total_layers)))

    for idx in targets:
        if hasattr(layers[idx], "mlp") and not isinstance(layers[idx].mlp, FovealDynamicSkipMLP):
            dev = next(layers[idx].mlp.parameters()).device
            dt = next(layers[idx].mlp.parameters()).dtype
            layers[idx].mlp = FovealDynamicSkipMLP(
                layers[idx].mlp,
                hidden_dim=model.config.hidden_size,
                index_dim=index_dim,
                threshold=threshold,
                dtype=dt,
            ).to(dev)

    return model


class DynamicSparsityDistillationLoss(nn.Module):
    """
    Distillation loss with an L1 sparsity penalty on dynamic gating probabilities.
    Encourages the 16D router to route to the 0-FLOP skip bridge for redundant tokens.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.5, lambda_sparse: float = 0.1):
        super().__init__()
        self.base_loss = LLMDistillationLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.lambda_sparse = lambda_sparse

    def forward(
        self,
        student_logits: Optional[torch.Tensor],
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        gate_probs: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_dict = self.base_loss(student_logits, student_hidden, teacher_hidden, labels)
        if gate_probs:
            mean_gates = torch.stack([g.mean() for g in gate_probs]).mean()
            sparse_loss = self.lambda_sparse * mean_gates
            loss_dict["sparse_reg"] = sparse_loss
            loss_dict["loss"] = loss_dict["loss"] + sparse_loss
            loss_dict["mean_gate"] = mean_gates
        return loss_dict


def compute_layer_redundancy(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> List[Tuple[int, float]]:
    """
    Measures the angular distance (1 - cosine_similarity) between consecutive layer representations.
    Lower angular distance indicates higher layer redundancy.
    """
    model.eval()
    layers = getattr(model, "model", model).layers
    hidden_states = []
    
    def get_hook():
        def hook(m, inp, out):
            # Handle both tensor outputs and tuple outputs
            t = out[0] if isinstance(out, (tuple, list)) else out
            hidden_states.append(t.detach().float())
        return hook

    hooks = [layer.register_forward_hook(get_hook()) for layer in layers]
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()
        
    redundancies = []
    for i in range(len(hidden_states) - 1):
        cos_sim = F.cosine_similarity(hidden_states[i], hidden_states[i+1], dim=-1).mean().item()
        angular_dist = 1.0 - cos_sim
        redundancies.append((i, angular_dist))
        
    return redundancies


def calibrate_svd_bridge(
    model: nn.Module,
    calibration_batches: List[torch.Tensor],
    start_layer: int,
    end_layer: int,
    rank: int = 128,
    ridge_reg: float = 1.0,
) -> SVDBridge:
    """
    Fits an SVD residual bridge that predicts the transformation Delta = Y - X
    across the block of dropped layers from start_layer to end_layer.
    """
    model.eval()
    layers = getattr(model, "model", model).layers
    device = next(model.parameters()).device
    
    captured = {}
    def hook_in(m, inp, out):
        captured["in"] = inp[0].detach()
    def hook_out(m, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        captured["out"] = t.detach()

    X_list, Y_list = [], []
    for batch in calibration_batches:
        batch = batch.to(device)
        h1 = layers[start_layer].register_forward_hook(hook_in)
        h2 = layers[end_layer].register_forward_hook(hook_out)
        with torch.no_grad():
            model(batch)
        h1.remove()
        h2.remove()
        
        # Flatten batch and sequence dimensions
        X_list.append(captured["in"].reshape(-1, captured["in"].shape[-1]).float())
        Y_list.append(captured["out"].reshape(-1, captured["out"].shape[-1]).float())

    X_all = torch.cat(X_list, dim=0) # [N_tokens, hidden_dim]
    Y_all = torch.cat(Y_list, dim=0)
    Delta_all = Y_all - X_all

    hidden_dim = X_all.shape[-1]
    XtX = X_all.T @ X_all + ridge_reg * torch.eye(hidden_dim, device=X_all.device)
    W_bridge = torch.linalg.solve(XtX, X_all.T @ Delta_all).T # [hidden_dim, hidden_dim]

    model_dtype = next(model.parameters()).dtype
    bridge = SVDBridge.from_weight(W_bridge, rank=rank, dtype=model_dtype)
    return bridge.to(device)


class LLMDistillationLoss(nn.Module):
    """
    Compound Distillation Loss for adapting pruned LLMs:
      L = alpha * L_CE + beta * (1 - cosine(h_s, h_t)) + gamma * MSE(h_s, h_t)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        student_logits: Optional[torch.Tensor],
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        
        # Representation Cosine Loss
        cos_sim = F.cosine_similarity(student_hidden.float(), teacher_hidden.float(), dim=-1).mean()
        rep_cos_loss = 1.0 - cos_sim
        loss_dict["rep_cos"] = rep_cos_loss
        
        # Representation MSE Loss
        mse_loss = F.mse_loss(student_hidden.float(), teacher_hidden.float())
        loss_dict["rep_mse"] = mse_loss
        
        total_loss = self.beta * rep_cos_loss + self.gamma * mse_loss
        
        # Optional Cross-Entropy loss on target tokens
        if student_logits is not None and labels is not None:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = self.ce_loss(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss_dict["ce"] = ce_loss
            total_loss = total_loss + self.alpha * ce_loss
            
        loss_dict["loss"] = total_loss
        return loss_dict

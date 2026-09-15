"""
Hybrid Optimizer: Muon for 2D Square Weight Matrices + AdamW for 16D Routers & 1D Parameters.

Implements Newton-Schulz orthogonalization (Muon) for standard 2D projection weights,
paired with AdamW for low-rank router projections, 16D additive streams, and 1D parameters.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any, Union, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer, AdamW


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    r"""
    Newton-Schulz iteration to compute the approximate polar decomposition / zeroth power
    of momentum matrix G: O = G (G^T G)^{-1/2}.
    
    Coefficients from Keller Jordan's Muon optimizer:
    p(x) = 3.4445 * x - 4.7750 * x^3 + 2.0315 * x^5
    """
    assert len(G.shape) == 2, f"Newton-Schulz requires 2D tensor, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype != torch.bfloat16 and G.dtype != torch.float16 else G
    
    # Ensure spectral norm is roughly <= 1.0
    norm = X.norm() + eps
    X = X / norm
    
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if transposed:
        X = X.T
        
    return X.to(dtype=G.dtype)


class Muon(Optimizer):
    r"""
    Muon - Momentum Orthogonalized by Newton-Schulz.
    
    Applies polar decomposition to momentum updates for 2D parameters, ensuring
    all singular values of the parameter update are normalized to 1.0.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g_update = g.add(buf, alpha=momentum)
                else:
                    g_update = buf

                # Apply weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                if len(p.shape) == 2:
                    u = zeropower_via_newtonschulz5(g_update, steps=ns_steps)
                    # Aspect ratio scaling heuristic from Keller Jordan
                    scale = max(1.0, float(p.size(0)) / float(p.size(1))) ** 0.5
                    p.add_(u, alpha=-lr * scale)
                else:
                    p.add_(g_update, alpha=-lr)

        return loss


class HybridMuonAdamW:
    """
    Unified Hybrid Optimizer:
    - 2D internal weight matrices (square / broad projections) -> Muon
    - 16D routers, additive stream, 1D norms, biases, and embeddings -> AdamW
    """

    def __init__(
        self,
        named_params: Iterable[Tuple[str, Tensor]],
        lr_muon: float = 0.02,
        lr_adamw: float = 2e-4,
        momentum: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay_muon: float = 0.0,
        weight_decay_adamw: float = 0.01,
        min_dim_for_muon: int = 32,
        max_aspect_ratio_for_muon: float = 8.0,
    ):
        muon_params: List[Tensor] = []
        adamw_params: List[Tensor] = []
        self.param_categorization: Dict[str, str] = {}

        for name, p in named_params:
            if not p.requires_grad:
                continue

            # Check if parameter qualifies for Muon:
            # 1. Must be strictly 2D
            # 2. Both dimensions must be >= min_dim_for_muon (e.g. >= 32, excludes 16D routers)
            # 3. Aspect ratio must not be extreme (< max_aspect_ratio_for_muon)
            # 4. Must not be embedding or norm
            is_2d = len(p.shape) == 2
            not_small = is_2d and min(p.shape) >= min_dim_for_muon
            aspect_ratio = (max(p.shape) / min(p.shape)) if is_2d else 999.0
            not_extreme = aspect_ratio <= max_aspect_ratio_for_muon
            not_special = not any(k in name.lower() for k in ["embed", "lm_head", "norm", "bias", "indexer", "svd", "16d"])

            if is_2d and not_small and not_extreme and not_special:
                muon_params.append(p)
                self.param_categorization[name] = "Muon"
            else:
                adamw_params.append(p)
                self.param_categorization[name] = "AdamW"

        self.muon_opt = (
            Muon(muon_params, lr=lr_muon, momentum=momentum, weight_decay=weight_decay_muon)
            if muon_params
            else None
        )
        self.adamw_opt = (
            AdamW(adamw_params, lr=lr_adamw, betas=betas, eps=eps, weight_decay=weight_decay_adamw)
            if adamw_params
            else None
        )

        self.param_groups: List[Dict[str, Any]] = []
        if self.muon_opt:
            self.param_groups.extend(self.muon_opt.param_groups)
        if self.adamw_opt:
            self.param_groups.extend(self.adamw_opt.param_groups)

    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.muon_opt:
            self.muon_opt.step()
        if self.adamw_opt:
            self.adamw_opt.step()

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon_opt:
            self.muon_opt.zero_grad(set_to_none=set_to_none)
        if self.adamw_opt:
            self.adamw_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "muon": self.muon_opt.state_dict() if self.muon_opt else None,
            "adamw": self.adamw_opt.state_dict() if self.adamw_opt else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.muon_opt and state_dict.get("muon"):
            self.muon_opt.load_state_dict(state_dict["muon"])
        if self.adamw_opt and state_dict.get("adamw"):
            self.adamw_opt.load_state_dict(state_dict["adamw"])

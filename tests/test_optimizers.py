import pytest
import torch
import torch.nn as nn
from foveal_indexer.optimizers import Muon, HybridMuonAdamW, zeropower_via_newtonschulz5


def test_zeropower_newtonschulz():
    # Test on a random 2D matrix
    G = torch.randn(64, 64)
    O = zeropower_via_newtonschulz5(G, steps=5)
    assert O.shape == G.shape
    # Check that O O^T has spectral norm near 1
    s = torch.linalg.svdvals(O.float())
    assert (s.max() - 1.0).abs() < 0.25


def test_muon_optimizer_step():
    model = nn.Sequential(
        nn.Linear(64, 64, bias=False),
        nn.Linear(64, 32, bias=False)
    )
    optimizer = Muon(model.parameters(), lr=0.01)
    
    x = torch.randn(4, 64)
    loss = model(x).sum()
    loss.backward()
    
    # Capture weights before step
    w_before = model[0].weight.clone()
    optimizer.step()
    w_after = model[0].weight.clone()
    
    assert not torch.allclose(w_before, w_after)


def test_hybrid_muon_adamw():
    # Create model with both broad 2D linear weights and 16D skinny router
    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_2d = nn.Linear(128, 128, bias=False)
            self.indexer_proj = nn.Linear(128, 16, bias=False)
            self.norm = nn.LayerNorm(128)

        def forward(self, x):
            return self.norm(self.linear_2d(x)) + self.indexer_proj(x).sum(dim=-1, keepdim=True)

    module = TestModule()
    hybrid = HybridMuonAdamW(module.named_parameters(), lr_muon=0.01, lr_adamw=1e-3)
    
    # Check categorization
    assert hybrid.param_categorization["linear_2d.weight"] == "Muon"
    assert hybrid.param_categorization["indexer_proj.weight"] == "AdamW"
    assert hybrid.param_categorization["norm.weight"] == "AdamW"
    assert hybrid.param_categorization["norm.bias"] == "AdamW"
    
    # Step verification
    x = torch.randn(4, 128)
    loss = module(x).sum()
    loss.backward()
    
    hybrid.step()
    hybrid.zero_grad()
    
    # State dict verification
    sd = hybrid.state_dict()
    assert "muon" in sd and "adamw" in sd
    hybrid.load_state_dict(sd)

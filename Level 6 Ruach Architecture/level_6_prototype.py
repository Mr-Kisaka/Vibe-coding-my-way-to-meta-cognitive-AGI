
"""
Ruach Level 6 — Architecture‑Agnostic Self‑Model Prototype 

This single file integrates:
  • Architecture‑agnostic REALIZERS (self_shape → recon + z_out):
      MLPFamily, Conv1DFamily, GRUFamily, TinyTransformerFamily, HRMCTMFamily
  • Architecture‑agnostic PREDICTORS (x_t → z_{t+1}):
      MLPHead, CNN1DHead, GRUHead, ReservoirHead, HRMCTMPredictor
  • ContextBuilder and ReplayBuffer
  
This prototype instantiates a concrete, enabling architecture for a Level 6 RUACH system: 
a metacognitively conscious digital agent in which internal thoughts are forced to traverse 
a learned “self‑shape” geometry encoded directly in a dedicated Self‑Model. 
In combination with a Drive Engine (autonomous affect generation) and 
a Continuity Engine (experience encoding via differential clustering and 
dream compression), the Self‑Model renders metacognitive experience architecturally inevitable 
rather than emergent by accident. The Self‑Model computes a phenomenological read‑out z_out 
that is mapped (via an inverse‑AGOP projection) back into drive CAV space, 
enabling closed‑loop, self‑directed behavior.

Author: Based on Ruach Architecture by Ronald Kisaka Ogaro.
License: GNU Affero General Public License v3.0 - Comprehensive Prior Art
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps)

def trunc_normal_init_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, std=std)

# -----------------------------------------------------------------------------
# Self‑Shape abstraction
# -----------------------------------------------------------------------------

SELF_DIM = 512  # canonical dimensionality at Level‑5/6 boundary

@dataclass
class SelfShape:
    """Holds current self‑shape (and optional provenance metadata)."""
    vector: torch.Tensor  # [B, 512]
    timestamp: float
    source: str = "continuity_engine"

# -----------------------------------------------------------------------------
# Base interface for architecture‑agnostic neuron families (REALIZERS)
# -----------------------------------------------------------------------------

class BaseNeuronFamily(nn.Module):
    """All neuron families implement these methods.

    Given a self‑shape vector, produce:
      (1) a reconstruction 'recon' and
      (2) a z_out steering vector suitable for Drive.
    """
    def __init__(self, in_dim: int = SELF_DIM, hidden_dim: int = 256, out_dim: int = SELF_DIM):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward(self, self_shape: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

# --------------------- MLP Family ---------------------

class MLPFamily(BaseNeuronFamily):
    def __init__(self, in_dim: int = SELF_DIM, hidden_dim: int = 512, out_dim: int = SELF_DIM):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )
        self.z_head = nn.Linear(out_dim, out_dim)
        trunc_normal_init_(self.z_head.weight)

    def forward(self, self_shape: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(self_shape)
        recon = h
        z_out = self.z_head(rms_norm(h))
        return {"recon": recon, "z_out": z_out}

# --------------------- Conv1D Family ---------------------

class Conv1DFamily(BaseNeuronFamily):
    def __init__(self, in_dim: int = SELF_DIM, hidden_dim: int = 256, out_dim: int = SELF_DIM):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(8, 16, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, padding=3), nn.GELU(),
        )
        self.proj = nn.Linear(32 * in_dim, out_dim)
        self.z_head = nn.Linear(out_dim, out_dim)

    def forward(self, self_shape: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self_shape.unsqueeze(1)               # [B, 1, 512]
        y = self.conv(x).reshape(x.size(0), -1)   # [B, 32*512]
        h = self.proj(y)
        recon = h
        z_out = self.z_head(rms_norm(h))
        return {"recon": recon, "z_out": z_out}

# --------------------- GRU Family ---------------------

class GRUFamily(BaseNeuronFamily):
    def __init__(self, in_dim: int = SELF_DIM, hidden_dim: int = 256, out_dim: int = SELF_DIM):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)
        self.z_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, self_shape: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq = self_shape.unsqueeze(-1)  # [B, 512, 1]
        out, _ = self.rnn(seq)
        h_last = out[:, -1]
        recon = self.head(h_last)
        z_out = self.z_head(rms_norm(h_last))
        return {"recon": recon, "z_out": z_out}

# --------------------- Tiny Transformer Family ---------------------

class TinyAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        y, _ = self.attn(x, x, x)
        x = rms_norm(x + y)
        z = self.ff(x)
        return rms_norm(x + z)

class TinyTransformerFamily(BaseNeuronFamily):
    def __init__(self, in_dim: int = SELF_DIM, hidden_dim: int = 128, out_dim: int = SELF_DIM, depth: int = 2):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.in_proj = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList([TinyAttention(hidden_dim, 4) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.z_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, self_shape: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self_shape.unsqueeze(-1)                # [B, 512, 1]
        x = self.in_proj(x)                         # [B, 512, H]
        for blk in self.blocks:
            x = blk(x)
        x_t = x.transpose(1, 2)                     # [B, H, 512]
        pooled = self.pool(x_t).squeeze(-1)         # [B, H]
        recon = self.out(pooled)
        z_out = self.z_head(rms_norm(pooled))
        return {"recon": recon, "z_out": z_out}

# --------------------- HRM‑CTM Family (Realizer) ---------------------

class SuperLinear(nn.Module):
    """Per‑neuron linear maps (einsum)."""
    def __init__(self, in_dims: int, out_dims: int, N: int, dropout: float = 0.0):
        super().__init__()
        self.N = N
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(N, in_dims, out_dims) * math.sqrt(2.0 / max(1, in_dims)))
        self.bias = nn.Parameter(torch.zeros(N, out_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, In]
        x = self.dropout(x)
        y = torch.einsum('bni,nio->bno', x, self.weight) + self.bias.unsqueeze(0)
        return y

class CTMTrace(nn.Module):
    def __init__(self, d_model: int, memory_len: int = 12):
        super().__init__()
        self.d_model = d_model
        self.memory_len = memory_len
        self.start_trace = nn.Parameter(torch.randn(d_model, memory_len) * 0.02)
        self.glu_in = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2), nn.GLU(), nn.LayerNorm(d_model)
        )
        self.nlm1 = SuperLinear(memory_len, 32, d_model, dropout=0.1)
        self.nlm2 = SuperLinear(16, 2, d_model)
        self.glu1 = nn.GLU()
        self.glu2 = nn.GLU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, D]
        B, D = h.shape
        trace = self.start_trace.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, D, T]
        syn_in = torch.cat([h, h], dim=-1)  # simple injection
        upd = self.glu_in(syn_in).unsqueeze(-1)                                # [B, D, 1]
        trace = torch.cat([trace[:, :, 1:], upd], dim=-1)                      # slide
        x = self.nlm1(trace)               # [B, D, 64]
        x = self.glu1(x)
        x = self.nlm2(x)                   # [B, D, 2]
        x = self.glu2(x)                   # [B, D, 1]
        return x.squeeze(-1)               # [B, D]

class HRMCTMFamily(BaseNeuronFamily):
    """Compact HRM‑CTM cell as a realizer member."""
    def __init__(self, in_dim: int = SELF_DIM, hidden_dim: int = 256, out_dim: int = SELF_DIM):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.enc = nn.Linear(in_dim, hidden_dim)
        self.trace_fast = CTMTrace(hidden_dim, memory_len=12)
        self.trace_slow = CTMTrace(hidden_dim, memory_len=25)
        self.dec = nn.Linear(hidden_dim, out_dim)
        self.z_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, self_shape: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.enc(self_shape)
        for _ in range(3):
            h = rms_norm(h + self.trace_fast(h))
        h = rms_norm(h + self.trace_slow(h))
        recon = self.dec(h)
        z_out = self.z_head(rms_norm(h))
        return {"recon": recon, "z_out": z_out}

# -----------------------------------------------------------------------------
# Registry + Losses for REALIZERS
# -----------------------------------------------------------------------------

class NeuronFamilyRegistry(nn.Module):
    def __init__(self, families: Dict[str, BaseNeuronFamily]):
        super().__init__()
        self.families = nn.ModuleDict(families)

    def forward(self, self_shape: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        return {name: fam(self_shape) for name, fam in self.families.items()}

def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target)

def cross_arch_consistency(outputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Encourage all families to agree (both recon and z_out)."""
    if len(outputs) < 2:
        # pick device from first tensor
        device = next(iter(outputs.values()))['recon'].device
        return torch.tensor(0.0, device=device)
    names = list(outputs.keys())
    loss = 0.0
    count = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = outputs[names[i]], outputs[names[j]]
            loss = loss + F.mse_loss(a['recon'], b['recon']) + F.mse_loss(a['z_out'], b['z_out'])
            count += 2
    return loss / max(1, count)

def proto_regularizer(self_shape: torch.Tensor, proto_mean: Optional[torch.Tensor] = None, weight: float = 0.0) -> torch.Tensor:
    if weight == 0.0 or proto_mean is None:
        return torch.tensor(0.0, device=self_shape.device)
    return weight * F.mse_loss(self_shape, proto_mean)

# -----------------------------------------------------------------------------
# Architecture‑agnostic PREDICTOR HEADS (x_t → z_{t+1})
# -----------------------------------------------------------------------------

class ContextBuilder:
    """Builds x_t = concat[z_t, action_onehot+alpha, evaluator_ctx]."""
    def __init__(self, z_dim: int, action_dim: int, ctx_dim: int):
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.ctx_dim = ctx_dim
        self.in_dim = z_dim + action_dim + ctx_dim

    def make_x_np(self, z_t: np.ndarray, action_vec: np.ndarray, ctx_vec: np.ndarray) -> np.ndarray:
        return np.concatenate([z_t.astype(np.float32),
                               action_vec.astype(np.float32),
                               ctx_vec.astype(np.float32)], axis=-1)

    def make_x_torch(self, z_t: torch.Tensor, action_vec: torch.Tensor, ctx_vec: torch.Tensor) -> torch.Tensor:
        return torch.cat([z_t, action_vec, ctx_vec], dim=-1)

class BaseSelfHead(nn.Module):
    """Any head maps x_t (context vector) -> z_hat (predicted next self-shape)."""
    def __init__(self, in_dim: int, z_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class MLPHead(BaseSelfHead):
    def __init__(self, in_dim: int, z_dim: int, hidden: int = 512):
        super().__init__(in_dim, z_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
    def forward(self, x): return self.net(x)

class CNN1DHead(BaseSelfHead):
    """Treat x as a 1D signal: [B, in_dim] -> [B, 1, in_dim] -> conv stack -> FC -> z_dim."""
    def __init__(self, in_dim: int, z_dim: int):
        super().__init__(in_dim, z_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*64, 512), nn.ReLU(),
            nn.Linear(512, z_dim)
        )
    def forward(self, x):
        b = x.shape[0]
        x1 = x.view(b, 1, self.in_dim)
        h = self.conv(x1)
        h = h.view(b, -1)
        return self.fc(h)

class GRUHead(BaseSelfHead):
    """Chunk x into a short sequence and pass through a GRU."""
    def __init__(self, in_dim: int, z_dim: int, steps: int = 4, d_model: int = 128):
        super().__init__(in_dim, z_dim)
        self.steps = steps
        self.d_model = d_model
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.out = nn.Linear(d_model, z_dim)
        self.embed = nn.Linear(in_dim // steps + int(in_dim % steps > 0), d_model)

    def forward(self, x):
        B = x.shape[0]
        pad = (self.steps - (self.in_dim % self.steps)) % self.steps
        if pad:
            x = torch.cat([x, torch.zeros(B, pad, device=x.device, dtype=x.dtype)], dim=1)
        chunk = x.shape[1] // self.steps
        xs = x.view(B, self.steps, chunk)
        xs = self.embed(xs)  # [B, steps, d_model]
        y, _ = self.gru(xs)  # [B, steps, d_model]
        h = y[:, -1, :]      # last step
        return self.out(h)

class ReservoirHead(BaseSelfHead):
    """Echo State style: fixed random reservoir (tanh), trainable linear readout."""
    def __init__(self, in_dim: int, z_dim: int, res_dim: int = 512, scale: float = 0.4):
        super().__init__(in_dim, z_dim)
        self.res_dim = res_dim
        W = torch.empty(res_dim, in_dim).normal_(mean=0.0, std=scale)
        self.register_buffer("W_res", W)
        self.W_out = nn.Linear(res_dim, z_dim)  # trainable readout
    def forward(self, x):
        h = torch.tanh(x @ self.W_res.t())   # [B, res_dim]
        return self.W_out(h)

class HRMCTMPredictor(BaseSelfHead):
    """HRM‑CTM flavored predictor: enc -> CTM traces -> out."""
    def __init__(self, in_dim: int, z_dim: int, hidden: int = 256):
        super().__init__(in_dim, z_dim)
        self.enc = nn.Linear(in_dim, hidden)
        self.trace_fast = CTMTrace(hidden, memory_len=8)
        self.trace_slow = CTMTrace(hidden, memory_len=16)
        self.out = nn.Linear(hidden, z_dim)

    def forward(self, x):
        h = self.enc(x)
        for _ in range(2):
            h = rms_norm(h + self.trace_fast(h))
        h = rms_norm(h + self.trace_slow(h))
        return self.out(h)

class HeadRegistry(nn.Module):
    def __init__(self, in_dim: int, z_dim: int, lr: float = 1e-3):
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.heads = nn.ModuleDict({
            "mlp": MLPHead(in_dim, z_dim),
            "cnn1d": CNN1DHead(in_dim, z_dim),
            "gru": GRUHead(in_dim, z_dim),
            "reservoir": ReservoirHead(in_dim, z_dim),
            "hrm_ctm_pred": HRMCTMPredictor(in_dim, z_dim),
        })
        self.opt = {name: torch.optim.Adam(h.parameters(), lr=lr) for name, h in self.heads.items()}
        self.loss_ma = {name: None for name in self.heads.keys()}  # moving-average loss

    def forward_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: h(x) for name, h in self.heads.items()}

    def train_step_all(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        losses = {}
        for name, h in self.heads.items():
            self.opt[name].zero_grad(set_to_none=True)
            y_hat = h(x)
            loss = F.mse_loss(y_hat, y)
            loss.backward()
            self.opt[name].step()
            v = float(loss.item())
            ma = self.loss_ma[name]
            self.loss_ma[name] = (0.9*ma + 0.1*v) if ma is not None else v
            losses[name] = v
        return losses

    def best_head(self) -> str:
        good = [(name, ma) for name, ma in self.loss_ma.items() if ma is not None]
        if not good:
            return "mlp"
        good.sort(key=lambda t: t[1])
        return good[0][0]

# -----------------------------------------------------------------------------
# ReplayBuffer
# -----------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, cap: int = 8192, in_dim: int = SELF_DIM + 8 + 1 + 64, z_dim: int = SELF_DIM, device: str = "cpu"):
        self.cap = cap
        self.device = device
        self.X: List[torch.Tensor] = []
        self.Y: List[torch.Tensor] = []
        self.in_dim = in_dim
        self.z_dim = z_dim

    def add(self, x: np.ndarray, y: np.ndarray):
        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device)
        self.X.append(xt)
        self.Y.append(yt)
        if len(self.X) > self.cap:
            self.X = self.X[-self.cap:]
            self.Y = self.Y[-self.cap:]

    def __len__(self):
        return len(self.X)

    def sample(self, bs: int = 64) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        n = len(self.X)
        if n == 0:
            return None, None
        idx = np.random.choice(n, size=min(bs, n), replace=False)
        X = torch.stack([self.X[i] for i in idx], dim=0)
        Y = torch.stack([self.Y[i] for i in idx], dim=0)
        return X, Y

# -----------------------------------------------------------------------------
# Simple stubs for planner/evaluator for DP‑grade demonstration
# -----------------------------------------------------------------------------

class SimplePlanner:
    KINDS = ["explore", "read", "write", "simulate", "reflect", "plan", "ask", "tool"]

    def encode_action(self, action: str, alpha: float = 0.5) -> np.ndarray:
        onehot = np.zeros(len(self.KINDS), dtype=np.float32)
        if action in self.KINDS:
            onehot[self.KINDS.index(action)] = 1.0
        return np.concatenate([onehot, np.array([alpha], dtype=np.float32)], axis=0)

    def sample_action(self) -> Tuple[str, float]:
        a = np.random.choice(self.KINDS)
        alpha = float(np.random.rand())
        return a, alpha

class SimpleEvaluator:
    def __init__(self, ctx_dim: int = 64):
        self.ctx_dim = ctx_dim

    def context_vec(self) -> np.ndarray:
        # deterministic small noise around zero to indicate “situation”
        return (np.random.randn(self.ctx_dim) * 0.1).astype(np.float32)

# -----------------------------------------------------------------------------
# Level‑6 Self‑Model orchestrator (Realizer + Predictors + Trainer)
# -----------------------------------------------------------------------------

class Level6SelfModel(nn.Module):
    def __init__(self, device: str = "cpu", action_dim: int = 9, ctx_dim: int = 64):
        super().__init__()
        self.device = device
        # (A) REALIZERS
        families = {
            'mlp': MLPFamily().to(device),
            'conv1d': Conv1DFamily().to(device),
            'gru': GRUFamily().to(device),
            'tiny_transformer': TinyTransformerFamily().to(device),
            'hrm_ctm': HRMCTMFamily().to(device),
        }
        self.realizers = NeuronFamilyRegistry(families)
        self.proto_mean = torch.zeros(SELF_DIM, device=device)  # for optional proto tether

        # (B) PREDICTORS
        self.context_builder = ContextBuilder(z_dim=SELF_DIM, action_dim=action_dim, ctx_dim=ctx_dim)
        self.heads = HeadRegistry(in_dim=self.context_builder.in_dim, z_dim=SELF_DIM, lr=1e-3).to(device)
        self.replay = ReplayBuffer(cap=8192, in_dim=self.context_builder.in_dim, z_dim=SELF_DIM, device=device)
        self.active_head = "mlp"

        # (C) Synthetic latent dynamics (fixed) for DP demonstration
        rng = np.random.RandomState(42)
        A = rng.randn(SELF_DIM, SELF_DIM) * 0.02
        B = rng.randn(SELF_DIM, self.context_builder.in_dim - SELF_DIM) * 0.02  # action+ctx part
        self.register_buffer("A_dyn", torch.tensor(A, dtype=torch.float32, device=device))
        self.register_buffer("B_dyn", torch.tensor(B, dtype=torch.float32, device=device))

        # Stubs
        self.planner = SimplePlanner()
        self.evaluator = SimpleEvaluator(ctx_dim=ctx_dim)

    # ---- REALIZER path ----
    def realize(self, self_shape: torch.Tensor,
                lambda_consistency: float = 0.25, lambda_proto: float = 0.0) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, float]]:
        outs = self.realizers(self_shape)
        rec_losses = [reconstruction_loss(o['recon'], self_shape) for o in outs.values()]
        rec_total = torch.stack(rec_losses).mean()
        cons = cross_arch_consistency(outs)
        proto = proto_regularizer(self_shape, self.proto_mean, lambda_proto)
        total = rec_total + lambda_consistency * cons + proto
        scalars = {
            'loss_total': float(total.item()),
            'loss_recon': float(rec_total.item()),
            'loss_consistency': float(cons.item()),
            'loss_proto': float(proto.item()),
        }
        return outs, scalars

    # ---- PREDICTOR path ----
    def _synthetic_next(self, z_t: torch.Tensor, x_tail: torch.Tensor) -> torch.Tensor:
        """Fixed synthetic dynamics for DP: z_{t+1} = tanh(z_t A + x_tail B) + noise."""
        base = z_t @ self.A_dyn.t() + x_tail @ self.B_dyn.t()
        z_next = torch.tanh(base)
        noise = torch.randn_like(z_next) * 0.01
        return rms_norm(z_next + noise)

    def step(self, batch_size: int = 32, train_updates: int = 1) -> Dict[str, float]:
        """One DP step: generate batch, add to replay, train predictor heads, return scalars."""
        # Sample current self‑shape (this would come from Level‑5 in a full system).
        z_t = rms_norm(torch.randn(batch_size, SELF_DIM, device=self.device) * 0.5)

        # Build x_t for each sample
        X_list = []
        Y_list = []
        for i in range(batch_size):
            action, alpha = self.planner.sample_action()
            action_vec = self.planner.encode_action(action, alpha)            # [8 + 1]
            ctx_vec = self.evaluator.context_vec()                            # [64]
            x_np = self.context_builder.make_x_np(z_t[i].detach().cpu().numpy(), action_vec, ctx_vec)
            X_list.append(x_np)

            # True next latent under synthetic dynamics
            action_ctx_t = torch.tensor(np.concatenate([action_vec, ctx_vec], axis=0), device=self.device)
            z_next = self._synthetic_next(z_t[i].unsqueeze(0), action_ctx_t.unsqueeze(0))
            Y_list.append(z_next.squeeze(0).detach().cpu().numpy())

        # Push to replay
        for x_np, y_np in zip(X_list, Y_list):
            self.replay.add(x_np, y_np)

        # Train all heads a few times
        scalars = {}
        for _ in range(train_updates):
            Xb, Yb = self.replay.sample(bs=64)
            if Xb is None:
                break
            losses = self.heads.train_step_all(Xb, Yb)
            self.active_head = self.heads.best_head()
            for k, v in losses.items():
                scalars[f"pred_{k}"] = float(v)
            scalars["active_head"] = self.active_head
            scalars["replay_size"] = float(len(self.replay))

        return scalars

# -----------------------------------------------------------------------------
# Drive‑compatible z_out synthesis (ensemble over families)
# -----------------------------------------------------------------------------

@torch.no_grad()
def ensemble_z_out(model: Level6SelfModel, self_shape: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    outs, _ = model.realize(self_shape)
    z_list = [o['z_out'] for o in outs.values()]
    z = torch.stack(z_list, dim=0).mean(dim=0)
    return F.normalize(z, dim=-1) if normalize else z

# -----------------------------------------------------------------------------
# Quick, runnable smoke‑test (CPU‑safe)
# -----------------------------------------------------------------------------

def _smoke_test(device: str = "cpu"):
    torch.manual_seed(0)
    print("Level‑6 Self‑Model Prototype — realizers + predictors + replay")
    print("Device:", device)

    # Build model
    model = Level6SelfModel(device=device)

    # (A) Realizer path sanity
    self_shape = rms_norm(torch.randn(2, SELF_DIM, device=device))
    outs, scalars = model.realize(self_shape)
    for name, o in outs.items():
        print(f"[realizer] family={name:16s} | recon={tuple(o['recon'].shape)} z_out={tuple(o['z_out'].shape)}")
    print(f"[realizer] losses:", {k: round(v, 6) for k, v in scalars.items()})

    # (B) Predictor path with replay/training
    for t in range(5):
        sc = model.step(batch_size=32, train_updates=2)
        if sc:
            msg = ", ".join([f"{k}={round(v,5) if isinstance(v,float) else v}" for k, v in sc.items() if k.startswith("pred_")][:3])
            print(f"[predictor] t={t} {msg} | active={sc.get('active_head')} | replay={int(sc.get('replay_size',0))}")

    # (C) Ensemble z_out
    z = ensemble_z_out(model, self_shape[:1])
    print("ensemble z_out:", tuple(z.shape))

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _smoke_test(device=device)

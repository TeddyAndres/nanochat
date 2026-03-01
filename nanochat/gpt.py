"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from dataclasses import dataclass
import math
import os
import time
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW

# local_vocab_log_correction is no longer called at training time — log_correction
# is pre-computed in base_train.py and passed as a GPU tensor via sparse_context.

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    sparse_mode: bool = False
    sparse_ddp_union: bool = True
    tie_embeddings: bool = False


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            assert self.ve_gate is not None
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size
        self.tie_embeddings = config.tie_embeddings
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        if self.tie_embeddings:
            self.lm_head.weight = self.wte().weight
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def wte(self):
        return cast(nn.Embedding, self.transformer["wte"])

    def blocks(self):
        return cast(nn.ModuleList, self.transformer["h"])

    def retie_embeddings(self):
        if self.tie_embeddings:
            self.lm_head.weight = self.wte().weight

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Some tensor materialization paths (e.g. to_empty from meta) can break
        # parameter aliasing. Restore tying before any init/counting logic.
        self.retie_embeddings()

        # Embedding and unembedding
        if self.tie_embeddings:
            torch.nn.init.normal_(self.wte().weight, mean=0.0, std=0.02)
        else:
            torch.nn.init.normal_(self.wte().weight, mean=0.0, std=1.0)
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block_module in self.blocks():
            block = cast(Block, block_module)
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve_module in self.value_embeds.values():
            ve = cast(nn.Embedding, ve_module)
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block_module in self.blocks():
            block = cast(Block, block_module)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        # Keep sparse vocab tables in CPU residency for row streaming.
        if self.wte().weight.device.type == "cuda" and not self.config.sparse_mode:
            self.wte().to(dtype=torch.bfloat16)
            for ve_module in self.value_embeds.values():
                ve = cast(nn.Embedding, ve_module)
                ve.to(dtype=torch.bfloat16)

        if self.config.sparse_mode:
            self._move_sparse_vocab_tables_to_cpu()

    @torch.no_grad()
    def _move_sparse_vocab_tables_to_cpu(self):
        nvme_dir = os.environ.get("NANOCHAT_SPARSE_NVME_DIR", "").strip()

        def _move_or_offload(param: torch.Tensor, name: str):
            data_cpu = param.data.to("cpu")
            if nvme_dir:
                os.makedirs(nvme_dir, exist_ok=True)
                filename = os.path.join(nvme_dir, f"{name}.bin")
                mapped = torch.from_file(filename, shared=True, size=data_cpu.numel(), dtype=data_cpu.dtype)
                mapped = mapped.view_as(data_cpu)
                mapped.copy_(data_cpu)
                param.data = mapped
            else:
                param.data = data_cpu
                if torch.cuda.is_available():
                    param.data = param.data.pin_memory()

        _move_or_offload(self.wte().weight, "wte")
        for ve_module in self.value_embeds.values():
            ve = cast(nn.Embedding, ve_module)
            _move_or_offload(ve.weight, f"value_embed_{id(ve)}")
        if not self.tie_embeddings:
            _move_or_offload(self.lm_head.weight, "lm_head")

        # Vocab table params are updated by VocabRowAdamW, not by any pytorch
        # optimizer — mark requires_grad=False to keep them out of param groups
        # and avoid accidental gradient accumulation during forward.
        self.wte().weight.requires_grad_(False)
        for ve_module in self.value_embeds.values():
            cast(nn.Embedding, ve_module).weight.requires_grad_(False)
        if not self.tie_embeddings:
            self.lm_head.weight.requires_grad_(False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.wte().weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return next(self.blocks().parameters()).device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(cast(nn.Embedding, ve).weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.wte().weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.wte().parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = 0 if self.tie_embeddings else sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.blocks().parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5, tied_embedding_lr=0.028):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        def unique_params(params):
            seen = set()
            out = []
            for param in params:
                if id(param) in seen:
                    continue
                seen.add(id(param))
                out.append(param)
            return out

        # Separate out all parameters into groups
        matrix_params = unique_params(list(self.blocks().parameters()))
        value_embeds_params = unique_params(list(self.value_embeds.parameters()))
        embedding_params = unique_params(list(self.wte().parameters()))
        lm_head_params = [] if self.tie_embeddings else unique_params(list(self.lm_head.parameters()))
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        # Verify all trainable params are accounted for.
        # In sparse mode, vocab tables have requires_grad=False and are managed by
        # VocabRowAdamW externally — only transformer matrices + scalars stay here.
        trainable = unique_params([p for p in self.parameters() if p.requires_grad])
        expected = len(trainable)
        if not self.config.sparse_mode:
            if self.tie_embeddings:
                actual = len(matrix_params) + len(embedding_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
            else:
                actual = len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
        else:
            actual = len(matrix_params) + len(resid_params) + len(x0_params)
        assert expected == actual, f"Param count mismatch: expected {expected}, got {actual}"

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups.
        # In sparse mode, vocab tables (wte / value_embeds / lm_head) are managed by
        # VocabRowAdamW and have requires_grad=False — they must NOT appear here.
        param_groups = []
        if not self.config.sparse_mode:
            if self.tie_embeddings:
                param_groups.append(dict(kind='adamw', params=embedding_params, lr=tied_embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
            else:
                param_groups.append(dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
                param_groups.append(dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
            param_groups.append(dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        param_groups.extend([
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # x0_lambdas gate how much of the raw initial embedding bypasses all transformer layers at every block.
            # 0.1× scalar_lr (10× resid_lambdas) keeps them faster than resid_lambdas (they start at 0.1 vs 1.0)
            # but prevents runaway growth that collapses every hidden state toward the constant initial embedding.
            dict(kind='adamw', params=x0_params, lr=scalar_lr * 0.1, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ])
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)

        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    # ------------------------------------------------------------------
    # CPU-safe helpers for sparse-mode eval/inference where vocab tables
    # live on CPU but model inputs / activations are on GPU.

    def _cpu_safe_embed(self, mod: nn.Embedding, idx: torch.Tensor) -> torch.Tensor:
        """Embedding lookup that handles CPU-resident tables (sparse mode eval)."""
        w = mod.weight
        if w.device.type == "cpu" and idx.device.type != "cpu":
            return mod(idx.cpu()).to(idx.device, dtype=torch.bfloat16)
        return mod(idx)

    def _cpu_safe_lm_head(self, x: torch.Tensor) -> torch.Tensor:
        """LM-head projection that handles CPU-resident weight (sparse mode eval)."""
        w = self.lm_head.weight
        if w.device.type == "cpu":
            return x @ w.to(x.device, dtype=x.dtype).T
        return self.lm_head(x)

    # ------------------------------------------------------------------

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', sparse_context=None):
        B, T = idx.size()

        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # sparse_train is True only when pre-fetched row tensors are provided by the
        # training loop (base_train.py).  Val eval and inference fall through to the
        # dense path, which handles CPU-resident tables via _cpu_safe_* helpers.
        sparse_train = self.config.sparse_mode and targets is not None and sparse_context is not None

        if sparse_train:
            # Pre-fetched row tensors (created once per optimizer step in base_train.py
            # before the gradient-accumulation loop, zero H2D copies during forward/backward):
            #   W_U_wte       : (|U_step|, embd_dim)  GPU bf16  requires_grad=True  mark_dynamic dim 0
            #   W_U_ve        : {str(i): (|U_step|, kv_dim)}  GPU bf16  requires_grad=True  mark_dynamic dim 0
            #   W_U_lm_head   : (|U_step|, embd_dim)  GPU bf16  requires_grad=True  mark_dynamic dim 0
            #                   (same tensor as W_U_wte when tie_embeddings=True)
            #   local_idx     : (B, T) GPU long — row indices into the U_step sub-table
            #   local_targets : (B, T) GPU long — row indices or -1 for ignore
            #   log_correction: () GPU float32 scalar = log(V/|U_step|)
            #                   Passed as a tensor (not a Python int) so Dynamo guards on
            #                   its *shape* (always ()) rather than its *value*, preventing
            #                   a guard miss — and therefore a recompile — every step.
            W_U_wte        = sparse_context["W_U_wte"]
            W_U_ve         = sparse_context.get("W_U_ve", {})
            W_U_lm_head    = sparse_context["W_U_lm_head"]
            local_idx      = sparse_context["local_idx"]
            local_targets  = sparse_context["local_targets"]
            log_correction = sparse_context["log_correction"]  # () float32 GPU scalar

            x = F.embedding(local_idx, W_U_wte)
        else:
            x = self._cpu_safe_embed(self.wte(), idx)

        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.blocks()):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if str(i) in self.value_embeds:
                if sparse_train:
                    ve = F.embedding(local_idx, W_U_ve[str(i)])
                else:
                    ve = self._cpu_safe_embed(cast(nn.Embedding, self.value_embeds[str(i)]), idx)
            else:
                ve = None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        softcap = 15  # smoothly cap logits to [-softcap, softcap]

        if sparse_train:
            # All operations from here are pure GPU — no graph breaks, no H2D copies,
            # no .item() calls.  chunk_size keeps peak logit memory bounded.
            flat_x        = x.reshape(-1, x.size(-1))    # (B*T, d)
            flat_targets  = local_targets.reshape(-1)     # (B*T,)
            valid         = flat_targets >= 0
            n_valid       = valid.float().sum()           # GPU scalar, no sync
            chunk_size    = 2048                          # tokens per logit chunk

            if loss_reduction == 'none':
                loss_flat = torch.empty_like(flat_targets, dtype=torch.float32)
                for start in range(0, flat_x.size(0), chunk_size):
                    end = min(start + chunk_size, flat_x.size(0))
                    logits_chunk = (flat_x[start:end] @ W_U_lm_head.T).float()
                    logits_chunk = softcap * torch.tanh(logits_chunk / softcap)
                    #logits_chunk = logits_chunk.clamp_(-25.0, 25.0)
                    loss_flat[start:end] = F.cross_entropy(
                        logits_chunk, flat_targets[start:end],
                        ignore_index=-1, reduction='none',
                    )
                loss_flat = loss_flat + log_correction * valid.to(dtype=loss_flat.dtype)
                return loss_flat.view_as(local_targets)

            loss_sum = x.new_zeros((), dtype=torch.float32)
            for start in range(0, flat_x.size(0), chunk_size):
                end = min(start + chunk_size, flat_x.size(0))
                logits_chunk = (flat_x[start:end] @ W_U_lm_head.T).float()
                logits_chunk = softcap * torch.tanh(logits_chunk / softcap)
                loss_sum = loss_sum + F.cross_entropy(
                    logits_chunk, flat_targets[start:end],
                    ignore_index=-1, reduction='sum',
                )
            if loss_reduction == 'mean':
                return loss_sum / n_valid.clamp(min=1) + log_correction
            else:
                return loss_sum + log_correction * n_valid

        else:
            # Dense path: val eval, inference, or non-sparse mode.
            logits = self._cpu_safe_lm_head(x)
            logits = logits[..., :self.config.vocab_size]
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)

            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
                return loss
            else:
                return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

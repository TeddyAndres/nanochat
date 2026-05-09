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

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

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


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


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
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = min(32, self.n_embd)
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

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
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

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
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
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

    @torch.no_grad()
    def init_weights(self, lm_head_init_std=None, lm_head_init_dist="normal"):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal/uniform, std=0.001 by default
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        lm_head_std = 0.001 if lm_head_init_std is None else float(lm_head_init_std)
        lm_head_init_dist = str(lm_head_init_dist).lower()
        if lm_head_init_dist not in {"normal", "uniform"}:
            raise ValueError(f"Unsupported lm_head_init_dist '{lm_head_init_dist}', expected 'normal' or 'uniform'")
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        if lm_head_init_dist == "normal":
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=lm_head_std)
        else:
            # Match the requested standard deviation under a uniform distribution.
            bound = (3.0 ** 0.5) * lm_head_std
            torch.nn.init.uniform_(self.lm_head.weight, -bound, bound)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
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
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
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
        return self.cos.device

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
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
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
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
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

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, value_embedding_lr=None, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5, include_vocab_tables=True):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        value_embedding_lr = embedding_lr if value_embedding_lr is None else value_embedding_lr

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        managed_param_count = len(matrix_params) + len(resid_params) + len(x0_params)
        if include_vocab_tables:
            managed_param_count += len(embedding_params) + len(lm_head_params) + len(value_embeds_params)
        assert len(list(self.parameters())) >= managed_param_count

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
        ]
        if include_vocab_tables:
            param_groups = [
                dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=value_embeds_params, lr=value_embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            ] + param_groups
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

    def forward_features(self, idx, kv_cache=None, active_vocab=None):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        if active_vocab is None:
            x = self.transformer.wte(idx) # embed current token
        else:
            x = F.embedding(idx, active_vocab["wte"])
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if str(i) in self.value_embeds:
                if active_vocab is None:
                    ve = self.value_embeds[str(i)](idx).to(x.dtype)
                else:
                    ve = F.embedding(idx, active_vocab["value_embeds"][str(i)]).to(x.dtype)
            else:
                ve = None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)
        return x

    def compute_logits(self, x, active_vocab=None, logit_scale=1.0, logit_bias=None, force_float=True):
        # Forward the lm_head (compute logits)
        softcap = 20 # smoothly cap the logits to the range [-softcap, softcap]
        if active_vocab is None:
            logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
            logits = logits[..., :self.config.vocab_size] # slice to remove padding
        else:
            active_lm_head = active_vocab["lm_head"]
            if active_lm_head.dtype != x.dtype:
                active_lm_head = active_lm_head.to(dtype=x.dtype)
            logits = F.linear(x, active_lm_head)
        if force_float:
            logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap) # squash the logits
        if active_vocab is not None and "cold_logit_bias" in active_vocab:
            sparse_logit_bias = active_vocab["cold_logit_bias"]
            if sparse_logit_bias.device != logits.device or sparse_logit_bias.dtype != logits.dtype:
                sparse_logit_bias = sparse_logit_bias.to(device=logits.device, dtype=logits.dtype)
            logits = logits + sparse_logit_bias.view(1, 1, -1)
        if logit_bias is not None:
            logit_bias = logit_bias.to(device=logits.device, dtype=logits.dtype)
            logits = logits + logit_bias.view(1, 1, -1)
        if active_vocab is not None and "logit_mask" in active_vocab:
            logit_mask = active_vocab["logit_mask"]
            if logit_mask.device != logits.device or logit_mask.dtype != torch.bool:
                logit_mask = logit_mask.to(device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(~logit_mask.view(1, 1, -1), -1e9)
        if logit_scale != 1.0:
            logits = logits * logit_scale
        return logits

    @torch.compiler.disable()  # generator function (yield) + data-dependent chunking loop cannot be traced
    def iter_logits(self, x, active_vocab=None, logit_scale=1.0, logit_bias=None, force_float=True, vocab_chunk_size=None):
        if active_vocab is None:
            total_vocab = self.config.vocab_size
            if vocab_chunk_size is None or vocab_chunk_size <= 0 or vocab_chunk_size >= total_vocab:
                yield 0, total_vocab, self.compute_logits(
                    x,
                    active_vocab=active_vocab,
                    logit_scale=logit_scale,
                    logit_bias=logit_bias,
                    force_float=force_float,
                )
                return
            lm_head_weight = self.lm_head.weight
        else:
            lm_head_weight = active_vocab["lm_head"]
            total_vocab = lm_head_weight.size(0)
            if vocab_chunk_size is None or vocab_chunk_size <= 0 or vocab_chunk_size >= total_vocab:
                yield 0, total_vocab, self.compute_logits(
                    x,
                    active_vocab=active_vocab,
                    logit_scale=logit_scale,
                    logit_bias=logit_bias,
                    force_float=force_float,
                )
                return

        softcap = 20
        cold_logit_bias = None
        logit_mask = None
        if active_vocab is not None and "cold_logit_bias" in active_vocab:
            cold_logit_bias = active_vocab["cold_logit_bias"]
        if active_vocab is not None and "logit_mask" in active_vocab:
            logit_mask = active_vocab["logit_mask"]
        if logit_bias is not None:
            logit_bias = logit_bias.to(device=x.device)

        expand_shape = [1] * (x.ndim - 1)
        for start in range(0, total_vocab, vocab_chunk_size):
            end = min(start + vocab_chunk_size, total_vocab)
            weight_chunk = lm_head_weight[start:end]
            if weight_chunk.dtype != x.dtype:
                weight_chunk = weight_chunk.to(dtype=x.dtype)
            logits = F.linear(x, weight_chunk)
            if force_float:
                logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)
            if cold_logit_bias is not None:
                bias_chunk = cold_logit_bias[start:end].to(device=logits.device, dtype=logits.dtype)
                logits = logits + bias_chunk.view(*expand_shape, -1)
            if logit_bias is not None:
                bias_chunk = logit_bias[start:end].to(dtype=logits.dtype)
                logits = logits + bias_chunk.view(*expand_shape, -1)
            if logit_mask is not None:
                mask_chunk = logit_mask[start:end].to(device=logits.device, dtype=torch.bool)
                logits = logits.masked_fill(~mask_chunk.view(*expand_shape, -1), -1e9)
            if logit_scale != 1.0:
                logits = logits * logit_scale
            yield start, end, logits

    def forward(
        self,
        idx,
        targets=None,
        kv_cache=None,
        loss_reduction='mean',
        active_vocab=None,
        logit_scale=1.0,
        logit_bias=None,
        return_logits=False,
        return_token_losses=False,
        return_sparse_analysis=False,
    ):
        x = self.forward_features(idx, kv_cache=kv_cache, active_vocab=active_vocab)
        logits = self.compute_logits(
            x,
            active_vocab=active_vocab,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            force_float=(targets is None),
        )

        if targets is not None:
            # training: given the targets, compute and return the loss
            token_losses = None
            if return_token_losses:
                token_losses = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                    reduction='none',
                ).view_as(targets)
                if loss_reduction == 'none':
                    loss = token_losses
                elif loss_reduction == 'sum':
                    loss = token_losses.sum()
                else:
                    valid_mask = targets != -1
                    loss = token_losses[valid_mask].mean() if valid_mask.any() else token_losses.sum()
            else:
                # TODO experiment with chunked cross-entropy?
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            if return_sparse_analysis:
                analysis_logits = logits.detach()
                safe_targets = targets.clamp_min(0)
                target_logits = analysis_logits.gather(2, safe_targets.unsqueeze(-1)).squeeze(-1)
                # Per-token CE losses reuse the same fused kernel path as the main loss above —
                # no extra logsumexp pass over B×T×V is needed.
                analysis_token_losses = F.cross_entropy(
                    analysis_logits.to(dtype=torch.float32).view(-1, analysis_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                    reduction="none",
                ).view_as(targets)
                analysis_k = min(2, analysis_logits.size(-1))
                topk_logits, topk_local = torch.topk(analysis_logits, k=analysis_k, dim=-1)
                if analysis_k < 2:
                    pad_shape = (*topk_logits.shape[:2], 2 - analysis_k)
                    topk_logits = torch.cat(
                        (
                            topk_logits,
                            torch.full(pad_shape, -float("inf"), dtype=topk_logits.dtype, device=topk_logits.device),
                        ),
                        dim=-1,
                    )
                    topk_local = torch.cat(
                        (
                            topk_local,
                            torch.full(pad_shape, -1, dtype=topk_local.dtype, device=topk_local.device),
                        ),
                        dim=-1,
                    )
                return loss.float(), analysis_token_losses, topk_logits, topk_local, target_logits
            if return_logits and return_token_losses:
                return loss.float(), logits, token_losses
            if return_logits:
                return loss.float(), logits
            if return_token_losses:
                return loss.float(), token_losses
            return loss.float()
        else:
            # inference: just return the logits directly
            return logits

    @torch.compiler.disable()  # generator function (yield) + dynamic seq growth cannot be traced
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

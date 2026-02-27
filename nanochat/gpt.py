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
from nanochat.sparse_optim import SparseHybridOptimizer
from nanochat.sparse_vocab import compute_batch_token_set, sparse_embedding, sparse_logits, sparse_embedding_cached, sparse_logits_cached, local_vocab_log_correction, SparseVocabPool

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
    # Capacity of each per-table SparseVocabPool in number of *rows* (not bytes).
    # -1 means auto: set to vocab_size (correct but generous; tune down for huge vocabs).
    # For very large vocabs set this to ~2.5 × expected |U_batch| to bound VRAM.
    sparse_pool_capacity: int = -1


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
        # SparseVocabPool instances are created in init_weights() when we have a real device.
        # Dict maps table-key → SparseVocabPool.
        self._sparse_pools: dict = {}

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
            self._init_sparse_pools()

    @torch.no_grad()
    def _init_sparse_pools(self):
        """Allocate one SparseVocabPool per vocab table (wte, ve_*, lm_head).

        Called once from init_weights() after tables have been moved to CPU.
        Pools are sized to hold up to `sparse_pool_capacity` rows.  For the
        2-step lookahead to yield 100% hit rate the capacity must be ≥
        |U_t ∪ U_{t+1}|.  The default (-1 → vocab_size) is always sufficient;
        for huge vocabs set GPTConfig.sparse_pool_capacity explicitly.
        """
        device = self.get_device()
        if device.type != "cuda":
            return  # pools are a GPU optimisation; skip on CPU-only runs
        cache_dtype = torch.bfloat16

        capacity = (
            self.config.vocab_size
            if self.config.sparse_pool_capacity <= 0
            else min(self.config.sparse_pool_capacity, self.config.vocab_size)
        )

        # wte / lm_head (shared when tie_embeddings=True)
        self._sparse_pools["wte"] = SparseVocabPool(
            self.config.vocab_size,
            self.wte().weight.size(1),
            capacity,
            cache_dtype,
            device,
        )
        if not self.tie_embeddings:
            self._sparse_pools["lm_head"] = SparseVocabPool(
                self.config.vocab_size,
                self.lm_head.weight.size(1),
                capacity,
                cache_dtype,
                device,
            )
        # value embed tables
        for i, ve_module in self.value_embeds.items():
            ve = cast(nn.Embedding, ve_module)
            self._sparse_pools[f"ve_{i}"] = SparseVocabPool(
                self.config.vocab_size,
                ve.weight.size(1),
                capacity,
                cache_dtype,
                device,
            )

        total_vram = sum(p.vram_bytes() for p in self._sparse_pools.values())
        print0(
            f"SparseVocabPool: {len(self._sparse_pools)} tables, "
            f"capacity={capacity:,} rows each, "
            f"total VRAM index+rows: {total_vram / 1024**2:.1f} MiB"
        )

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

    def _get_pool(self, key: str, master_weight: torch.Tensor) -> "SparseVocabPool | None":
        """Return the pool for *key*, lazily creating it if pools not yet inited."""
        if key in self._sparse_pools:
            return self._sparse_pools[key]
        # CPU-only path (no pools): fall back to direct index_select
        return None

    def _pool_get_rows(self, key: str, master_weight: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Retrieve rows from pool (GPU) or directly from master_weight (CPU fallback)."""
        pool = self._get_pool(key, master_weight)
        if pool is not None:
            return pool.get_rows(tokens, master_weight)
        # CPU-only fallback (no pool)
        return master_weight.index_select(0, tokens.to(master_weight.device)).to(
            tokens.device, dtype=torch.bfloat16 if tokens.device.type == "cuda" else master_weight.dtype
        )

    def pop_sparse_cache_stats(self) -> dict:
        """Aggregate and reset stats from all pools."""
        hits = misses = bytes_loaded = 0
        load_ms = 0.0
        for pool in self._sparse_pools.values():
            s = pool.pop_stats()
            hits         += s["hits"]
            misses       += s["misses"]
            bytes_loaded += s["bytes_loaded"]
            load_ms      += s["load_ms"]
        return {"hits": hits, "misses": misses, "bytes_loaded": bytes_loaded, "load_ms": load_ms}

    def clear_sparse_cache(self):
        """Fully invalidate all pools (e.g. after loading a checkpoint)."""
        for pool in self._sparse_pools.values():
            pool.global_to_slot.fill_(-1)
            pool.slot_to_global.fill_(-1)
            pool._evict_ptr = 0

    def invalidate_sparse_cache_tokens(self, tokens: torch.Tensor):
        """Evict *tokens* from every pool after the optimizer updates them.

        O(|tokens|) per table — no GPU sync, no pool scan.
        tokens should be a CPU tensor (from the dataloader's sparse_context).
        """
        if tokens.numel() == 0:
            return
        # Keep tokens on CPU; pool.invalidate handles CPU tensors directly.
        if tokens.device.type != "cpu":
            tokens = tokens.cpu()
        for pool in self._sparse_pools.values():
            pool.invalidate(tokens)

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
        if self.tie_embeddings:
            shared_vocab_params = embedding_params
            expected = len(unique_params(list(self.parameters())))
            actual = len(matrix_params) + len(shared_vocab_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
            assert expected == actual
        else:
            expected = len(unique_params(list(self.parameters())))
            actual = len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
            assert expected == actual

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = []
        sparse_param_groups = []
        if self.config.sparse_mode:
            if self.tie_embeddings:
                sparse_param_groups.append(dict(params=embedding_params, lr=tied_embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10))
            else:
                sparse_param_groups.append(dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10))
                sparse_param_groups.append(dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10))
            sparse_param_groups.append(dict(params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10))
        else:
            if self.tie_embeddings:
                param_groups.append(dict(kind='adamw', params=embedding_params, lr=tied_embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
            else:
                param_groups.append(dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
                param_groups.append(dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
            param_groups.append(dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        param_groups.extend([
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
        ])
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        dense_optimizer = Factory(param_groups)

        if self.config.sparse_mode:
            sparse_params = []
            for group in sparse_param_groups:
                sparse_params.extend(group['params'])
            sparse_optimizer = torch.optim.SparseAdam(sparse_param_groups)
            optimizer = SparseHybridOptimizer(dense_optimizer, sparse_optimizer, sparse_params)
        else:
            optimizer = dense_optimizer

        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', sparse_context=None):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        sparse_train = self.config.sparse_mode and targets is not None
        U = None
        local_idx = None
        local_targets = None
        desired_U_cpu = None   # CPU tensor for pool lookups (no GPU sync needed)
        U_pos = None
        wte_rows_u = None
        if sparse_train:
            if sparse_context is not None:
                # U, local_idx, local_targets, next_U all originate from the CPU dataloader.
                # Compute desired_U (union of current + next batch) and U_pos on CPU
                # *before* moving anything to GPU.  That way get_rows() receives a
                # CPU tensor and never needs to synchronise the GPU stream.
                U_cpu          = sparse_context["U"]                  # CPU tensor
                local_idx_cpu  = sparse_context["local_idx"]          # CPU tensor
                local_targets_cpu = sparse_context["local_targets"]   # CPU tensor
                next_U_cpu     = sparse_context.get("next_U")         # CPU or None

                desired_U_cpu = (
                    U_cpu if next_U_cpu is None
                    else torch.unique(torch.cat([U_cpu, next_U_cpu]), sorted=True)
                )
                # U ⊆ desired_U, both sorted → searchsorted gives U's positions in desired_U.
                U_pos_cpu = torch.searchsorted(desired_U_cpu, U_cpu)  # CPU op

                # Move model-computation tensors to GPU (all non-blocking)
                U             = U_cpu.to(idx.device, non_blocking=True)
                local_idx     = local_idx_cpu.to(idx.device, non_blocking=True)
                local_targets = local_targets_cpu.to(idx.device, non_blocking=True)
                U_pos         = U_pos_cpu.to(idx.device, non_blocking=True)
            else:
                # Fallback: no sparse_context (rare), compute on GPU then bring index to CPU.
                assert targets is not None
                U, _, local_idx, local_targets = compute_batch_token_set(
                    idx,
                    targets,
                    self.config.vocab_size,
                    use_ddp_union=self.config.sparse_ddp_union,
                )
                desired_U_cpu = U.cpu()   # D2H sync accepted only in this fallback path
                U_pos = torch.arange(U.numel(), device=idx.device, dtype=torch.long)

            # Pool lookups use desired_U_cpu (CPU tensor) → zero GPU synchronization
            desired_rows_wte = self._pool_get_rows("wte", self.wte().weight, desired_U_cpu)
            wte_rows_u = desired_rows_wte.index_select(0, U_pos)
            x = sparse_embedding_cached(self.wte().weight, local_idx, U, wte_rows_u)
        else:
            x = self.wte()(idx) # embed current token
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.blocks()):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if str(i) in self.value_embeds:
                ve_weight = cast(nn.Embedding, self.value_embeds[str(i)]).weight
                if sparse_train:
                    assert local_idx is not None and U is not None and desired_U_cpu is not None and U_pos is not None
                    desired_rows_ve = self._pool_get_rows(f"ve_{i}", ve_weight, desired_U_cpu)
                    ve_rows_u = desired_rows_ve.index_select(0, U_pos)
                    ve = sparse_embedding_cached(ve_weight, local_idx, U, ve_rows_u)
                else:
                    ve = cast(nn.Embedding, self.value_embeds[str(i)])(idx)
            else:
                ve = None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        if sparse_train:
            assert U is not None and desired_U_cpu is not None and U_pos is not None
            lm_weight = self.wte().weight if self.tie_embeddings else self.lm_head.weight
            lm_rows_u = wte_rows_u if self.tie_embeddings else self._pool_get_rows("lm_head", lm_weight, desired_U_cpu).index_select(0, U_pos)
            assert lm_rows_u is not None
            # Sparse mode can create a very large (B*T*|U|) tensor.
            # Compute cross-entropy in chunks to keep peak activation memory bounded
            # for both training and evaluation (including reduction='none').
            if targets is not None:
                assert local_targets is not None
                flat_x = x.reshape(-1, x.size(-1))
                flat_targets = local_targets.reshape(-1)
                valid = flat_targets >= 0
                valid_count = valid.sum().item()
                # Dynamically size chunks from currently available free VRAM.
                # This avoids hard fixed caps while still preventing large spikes.
                if flat_x.device.type == "cuda":
                    free_mem, total_mem = torch.cuda.mem_get_info(flat_x.device)
                    # Dynamic safe envelope: tie budget to both current free memory
                    # and total VRAM so chunk size adapts but cannot balloon.
                    logits_budget_bytes = int(min(free_mem * 0.03, total_mem * 0.04))
                    logits_budget_bytes = max(logits_budget_bytes, 64 * 1024 * 1024)
                    # logits are cast to fp32 for CE => 4 bytes per element
                    budget_elems = max(1, logits_budget_bytes // 4)
                else:
                    budget_elems = flat_x.size(0) * max(1, int(U.numel()))
                chunk_tokens = max(128, min(flat_x.size(0), budget_elems // max(1, int(U.numel()))))

                if loss_reduction == 'none':
                    loss_flat = torch.empty_like(flat_targets, dtype=torch.float32)
                    for start in range(0, flat_x.size(0), chunk_tokens):
                        end = min(start + chunk_tokens, flat_x.size(0))
                        logits_chunk = sparse_logits_cached(flat_x[start:end], lm_weight, U, lm_rows_u)
                        logits_chunk = softcap * torch.tanh(logits_chunk / softcap)
                        loss_flat[start:end] = F.cross_entropy(
                            logits_chunk.float(),
                            flat_targets[start:end],
                            ignore_index=-1,
                            reduction='none',
                        )
                    correction = local_vocab_log_correction(self.config.vocab_size, int(U.numel()))
                    loss_flat = loss_flat + correction * valid.to(dtype=loss_flat.dtype)
                    return loss_flat.view_as(local_targets)

                loss_sum = x.new_zeros((), dtype=torch.float32)
                for start in range(0, flat_x.size(0), chunk_tokens):
                    end = min(start + chunk_tokens, flat_x.size(0))
                    logits_chunk = sparse_logits_cached(flat_x[start:end], lm_weight, U, lm_rows_u)
                    logits_chunk = softcap * torch.tanh(logits_chunk / softcap)
                    loss_sum = loss_sum + F.cross_entropy(
                        logits_chunk.float(),
                        flat_targets[start:end],
                        ignore_index=-1,
                        reduction='sum',
                    )
                if valid_count == 0:
                    loss = loss_sum
                elif loss_reduction == 'mean':
                    loss = loss_sum / valid_count
                else:
                    loss = loss_sum
                correction = local_vocab_log_correction(self.config.vocab_size, int(U.numel()))
                if loss_reduction == 'mean':
                    loss = loss + correction
                else:
                    loss = loss + correction * valid_count
                return loss

            logits = sparse_logits_cached(x, lm_weight, U, lm_rows_u)
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)
        else:
            logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
            logits = logits[..., :self.config.vocab_size] # slice to remove padding
            logits = logits.float() # switch to fp32 for logit softcap and loss computation
            logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            target_tensor = local_targets if sparse_train else targets
            assert target_tensor is not None
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tensor.view(-1), ignore_index=-1, reduction=loss_reduction)
            if sparse_train:
                assert U is not None
                correction = local_vocab_log_correction(self.config.vocab_size, int(U.numel()))
                if loss_reduction == 'none':
                    valid = target_tensor.view(-1) >= 0
                    loss = loss + correction * valid.to(dtype=loss.dtype)
                else:
                    loss = loss + correction
            return loss
        else:
            # inference: just return the logits directly
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

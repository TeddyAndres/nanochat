# Dynamic Vocabulary Sparsity — Design Document

**Project**: Nanochat Dynamic Vocabulary Sparsity  
**Date**: February 25, 2026  
**Version**: 2.0  
**Status**: Design Document

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Vocab-Sized Components](#2-vocab-sized-components)
3. [Component Treatment](#3-component-treatment)
   - 3.1 [wte — Input Embedding (Exact)](#31-wte--input-embedding-exact)
   - 3.2 [value_embeds — Value Embeddings (Exact)](#32-value_embeds--value-embeddings-exact)
   - 3.3 [lm_head — Output Projection (Design Decision)](#33-lm_head--output-projection-design-decision)
4. [Token Set Definition](#4-token-set-definition)
5. [Local Index Remapping](#5-local-index-remapping)
6. [Optimizer Treatment](#6-optimizer-treatment)
7. [DDP and Gradient Synchronization](#7-ddp-and-gradient-synchronization)
8. [Softcap Compatibility](#8-softcap-compatibility)
9. [Inference Mode](#9-inference-mode)
10. [Memory Analysis](#10-memory-analysis)
11. [Implementation Plan](#11-implementation-plan)
12. [Testing Plan](#12-testing-plan)

---

## 1. Problem Statement

Every vocab-sized component in the model allocates memory proportional to `V x d`, where `V` is the vocabulary size and `d` is the embedding dimension. At 50k tokens this is manageable. The cost becomes a hard wall at large vocabularies:

| Vocab Size | wte | value_embeds (x6 layers) | lm_head | Optimizer states (x2) | Total |
|---|---|---|---|---|---|
| 50k (current) | 77MB | ~100MB | 77MB | ~508MB | ~762MB |
| 500k | 770MB | ~1GB | 770MB | ~5GB | ~7.5GB |
| 1M | 1.5GB | ~2GB | 1.5GB | ~10GB | ~15GB |

At 1M vocab the embedding system alone consumes more VRAM than the entire rest of the model (attention, MLP layers, activations). This is the problem being solved. The design applies dynamic sparsity to all three vocab-sized components, loading only the rows corresponding to unique tokens present in the current batch.

---

## 2. Vocab-Sized Components

There are exactly three components in this model that scale with vocabulary size `V`:

| Component | Location in Code | Shape | What It Does |
|---|---|---|---|
| `wte` | `self.transformer.wte` | `(V, n_embd)` | Maps input token IDs to embedding vectors |
| `value_embeds[i]` | `self.value_embeds` (alternating layers) | `(V, kv_dim)` per active layer | Per-position value residual (ResFormer-style) |
| `lm_head` | `self.lm_head` | `(V, n_embd)` | Projects residual stream to logits over vocabulary |

The design adopts **weight tying**: `lm_head.weight` is set equal to `wte.weight`, sharing a single `(V, n_embd)` tensor. This reduces the number of distinct vocab-sized weight tensors from three to two (`wte/lm_head` shared + `value_embeds`). At 1M vocab this halves the dominant weight memory cost. See Section 3.3 for the full treatment.

Each component receives a different treatment because each participates differently in the computation graph.

---

## 3. Component Treatment

### 3.1 `wte` — Input Embedding (Exact)

**Operation**: The forward pass is `x = wte(idx)`, which is a gather: for each token ID in `idx`, retrieve the corresponding row of the weight matrix. Only rows whose token IDs appear in `idx` are ever touched.

**Sparse treatment**: Load only the rows for `U = unique_tokens(batch)` to VRAM. All other rows stay in CPU memory (or NVMe for very large vocabs).

**Mathematical status**: **Exact.** The gather operation is identical whether the full matrix is in VRAM or only the required rows. No approximation is introduced.

**Gradient**: The gradient `dL/d(wte[i])` is non-zero only for token IDs `i` that appear in the batch. This is a naturally sparse gradient of shape `(|U|, n_embd)`. The full weight matrix `wte.weight` never needs to live in VRAM; only the active rows do.

**Implementation**:
1. Store the full weight matrix in CPU pinned memory
2. On each forward pass, gather only `U` rows to VRAM into a small dense tensor `W_U` of shape `(|U|, n_embd)`
3. Remap token IDs from global (`0...V`) to local (`0...|U|-1`) indices (see Section 5)
4. Apply `F.embedding(local_idx, W_U)` — standard dense gather on the small matrix
5. On backward, receive sparse gradient `dL/dW_U` of shape `(|U|, n_embd)`, scatter back to the appropriate CPU rows

### 3.2 `value_embeds` — Value Embeddings (Exact)

Value embeddings are used identically to `wte`: `ve = value_embeds[i](idx)` is also a gather. The same analysis applies without modification.

**Mathematical status**: **Exact.** Same gather semantics as `wte`.

**Implementation**: Apply the same sparse wrapper as `wte` independently to each active `value_embeds[i]`. Because the token set `U` is the same for both `wte` and `value_embeds` within a batch (same `idx`), the index remapping computed once for `wte` can be reused for all value embedding layers.

**Shape note**: `value_embeds[i]` has shape `(V, kv_dim)` where `kv_dim = n_kv_head x head_dim`, which may differ from `n_embd`. The same approach applies directly.

### 3.3 `lm_head` — Output Projection (Design Decision)

This component requires two deliberate design decisions: (1) whether to tie `lm_head.weight` to `wte.weight`, and (2) whether to compute the softmax over the full vocabulary or only over the batch-local token set `U`.

#### 3.3.0 Weight Tying

**Decision**: `lm_head.weight = wte.weight` — they share the same `(V, n_embd)` tensor.

**Rationale**: At large vocabulary sizes the embedding matrix and output projection matrix are each `O(V * d)` and are the dominant memory cost. Tying them halves this cost with no increase in code complexity. Weight tying is standard practice in the field (original GPT-2, T5, many others) and typically has neutral-to-positive effect on model quality due to the regularization effect of constrained parameter space.

**Memory impact**:

| Vocab | Two separate | Tied | Saving |
|---|---|---|---|
| 50k | 154MB | 77MB | 77MB |
| 1M | 3.07GB | 1.54GB | 1.54GB |

**Gradient flow under tying**: During a training step, row `i` of the shared weight receives gradient contributions from two sources:

```
total_grad[i] = d(L)/d(wte[i])     # from embedding lookup
              + d(L)/d(lm_head[i])  # from output projection
```

Both contributions are non-zero only for `i in U` (the batch token set). The combined gradient is therefore still sparse with support on `U`, exactly as for the untied case. No change to the sparse gradient accumulation logic.

**Initialization conflict**: The current code initializes `wte` with `std=1.0` and `lm_head` with `std=0.001`. These cannot coexist on a shared tensor. With tying, a single initialization must be chosen. The standard approach (used in GPT-2 and the original Karpathy nanoGPT) is to initialize the tied weight with `std=0.02` (a middle-ground value that stabilizes both the embedding lookup scale and the output logit scale). Alternatively, `1/sqrt(n_embd)` (uniform) is used here for all other matrix parameters and provides a principled choice. **This initialization must be validated empirically** — the current separate initializations were tuned for untied parameters.

**Optimizer conflict**: The current `setup_optimizer` uses `embedding_lr=0.2` for `wte` and `unembedding_lr=0.004` for `lm_head`. Tied parameters share a single optimizer parameter group and therefore a single learning rate. The appropriate unified LR is a hyperparameter that must be tuned. A reasonable starting point is the geometric mean: `sqrt(0.2 * 0.004) ~= 0.028`, or the Muon-inspired `1/sqrt(d_model)` scaling. This is a key hyperparameter decision that may require ablation.

#### 3.3.1 Standard (Full) Forward Pass

The standard forward pass computes logits over the entire vocabulary:

```python
logits = x @ lm_head.weight.T    # shape: (B, T, V)
loss = cross_entropy(logits, targets)
```

The cross-entropy loss at position (b, t) with target y_{b,t}:

    L_full = -log( exp(z_{y}) / sum_{i=1}^{V} exp(z_i) )

where z_i = x_{b,t} . W_i^T is the logit for token i.

The gradient for each row W_i of `lm_head.weight`:

    dL/dW_i = ( softmax(z)_i - 1[i == y_{b,t}] ) * x_{b,t}

This gradient is **non-zero for every row i in [1, V]** at every position, because `softmax(z)_i > 0` for all i.

#### 3.3.2 Batch-Local Forward Pass (This Design)

**Design decision**: Restrict both the forward projection and the softmax denominator to only the unique tokens present in the batch, U.

The sparse forward pass computes logits only for tokens in U:

```python
W_U = lm_head.weight[U]           # shape: (|U|, n_embd) — only |U| rows loaded to VRAM
logits_U = x @ W_U.T              # shape: (B, T, |U|) — much smaller tensor
```

The batch-local loss at position (b, t) with target y_{b,t} (where y_{b,t} is in U always, since targets come from the same batch):

    L_local = -log( exp(z_{y}) / sum_{i in U} exp(z_i) )

**This is not numerically identical to the full softmax loss.** The denominator is smaller (fewer terms), so the raw loss value will be lower. This is a deliberate design choice: the model is trained to compete only among tokens that are contextually present.

#### 3.3.3 Importance-Weighted Correction

The batch-local loss is a biased estimator of the full loss. To obtain an **unbiased estimator** in expectation, apply an importance weight to the denominator.

The full normalizer is Z = sum_{i=1}^{V} exp(z_i). We approximate it using the sampled set U:

    Z_hat = (V / |U|) * sum_{i in U} exp(z_i)

This is the standard importance sampling estimator: each sample is weighted by the inverse of its sampling probability (|U| / V), which scales up the sampled sum to represent the full vocabulary.

The importance-weighted loss:

    L_corrected = -log( exp(z_{y}) / Z_hat )
                = -log( exp(z_{y}) / ( (V/|U|) * sum_{i in U} exp(z_i) ) )
                = -z_{y} + log(V/|U|) + logsumexp(z_U)

This simplifies to: add `log(V / |U|)` as a scalar correction to the standard cross-entropy computed over local logits:

```python
# Correct implementation:
logits_U = x @ W_U.T                                   # (B, T, |U|)
logits_U = logits_U.float()
logits_U = softcap * torch.tanh(logits_U / softcap)    # apply softcap element-wise
loss = F.cross_entropy(logits_U.view(-1, len(U)),       # standard CE over local vocab
                       local_targets.view(-1),
                       ignore_index=-1,
                       reduction='mean')
loss = loss + math.log(V / len(U))                     # importance weight correction
```

**Why the correction is additive on the loss scalar, not on the logits**:

CE computes: -(z_y) + logsumexp(z_U)

We want:    -(z_y) + log(V/|U|) + logsumexp(z_U)

These differ by the constant `log(V/|U|)` which depends only on batch statistics, not on the model
output. Adding it to the logits (i.e., `logits_U + log(V/|U|)`) would cancel in CE because CE
subtracts the same constant from both numerator and denominator. The correct place is to add it
to the loss scalar directly after `F.cross_entropy`.

**Mathematical status**: This is an **unbiased estimator** of the full cross-entropy gradient in expectation, under the assumption that tokens in U are sampled proportionally to their probability mass. In practice the sampling is not uniform (frequent tokens appear more), which introduces a mild bias in favor of high-frequency tokens — consistent with frequency-weighted training objectives throughout NLP, and arguably the correct inductive bias.

#### 3.3.4 Gradient Under Batch-Local Softmax

With the importance-weighted correction, the gradient for row W_i of `lm_head.weight` for i in U:

    dL_corrected/dW_i = ( softmax_U(z)_i - 1[i == y_{b,t}] ) * x_{b,t}

where `softmax_U(z)_i` is the softmax computed only over U (using the full local denominator, not importance-scaled).

For i not in U: **zero gradient**. Rows not present in the batch receive no update. Over a full training run with sufficient steps, all tokens will appear in batches and receive updates. Rare tokens receive fewer updates, which is consistent with their lower contribution to the loss.

#### 3.3.5 Logit Tensor Memory Savings

This is where the most immediate memory savings materialize, even at 50k vocab. At `B=4, T=2048` with a typical batch coverage of `|U| ~= 10,000` unique tokens:

| Tensor | Full softmax shape | Sparse shape | Memory (fp32) |
|---|---|---|---|
| `logits` | `(4, 2048, 50257)` | `(4, 2048, 10000)` | 1.6GB -> 320MB |
| `lm_head.weight` loaded | `(50257, 768)` | `(10000, 768)` | 77MB -> 15MB |

At 1M vocab:

| Tensor | Full softmax | Sparse |
|---|---|---|
| `logits` | `(4, 2048, 1M)` = 32GB | `(4, 2048, 10000)` = 320MB |
| `lm_head.weight` loaded | `(1M, 768)` = 3GB | `(10000, 768)` = 15MB |

**The logit tensor is the single largest allocation in the forward pass for large vocabularies. Sparse lm_head eliminates this bottleneck.**

---

## 4. Token Set Definition

The unique token set `U` for a batch is computed as:

```python
# idx shape: (B, T), targets shape: (B, T)
U = torch.unique(torch.cat([idx.flatten(), targets.flatten()]))
```

**Why include targets**: The target tokens must be included in `U` because:
1. Their rows in `lm_head.weight` must be loaded to compute the target logit `z_y` (the numerator of CE)
2. The target local index `local_targets` must map to a valid position in `[0, |U|)`

In practice `targets[b, t] = idx[b, t+1]` (next-token prediction), so including both `idx` and `targets` adds at most `B` tokens beyond `idx` (the last token of each sequence). The sets are nearly identical.

---

## 5. Local Index Remapping

All three sparse components operate on a local dense index space `[0, |U|)` rather than global indices `[0, V)`.

**Remapping procedure** (computed once per batch, reused across all components):

```python
# U is a sorted 1D tensor of unique global token IDs, shape (|U|,)
# Build global -> local mapping
global_to_local = torch.full((V,), -1, dtype=torch.long, device=device)
global_to_local[U] = torch.arange(len(U), device=device)

# Remap input tokens and targets
local_idx = global_to_local[idx]             # (B, T), values in [0, |U|)
local_targets = global_to_local[targets]     # (B, T), values in [0, |U|)
```

The `global_to_local` tensor occupies `V * 4` bytes (int32). At 1M vocab this is 4MB — acceptable to keep in VRAM throughout training. For extremely large vocabularies (100M+), it can be replaced with a sparse hash map.

**During backward**, local gradient indices are mapped back to global via `U[local_grad_idx]` before scattering into the CPU-resident full weight matrices.

**Verification**: `U[local_idx]` must equal `idx` exactly for all positions. This is an assertion to add in debug mode.

---

## 6. Optimizer Treatment

Standard AdamW maintains first and second moment vectors (`m`, `v`) of the same shape as each parameter. For vocab-sized parameters of shape `(V, d)`, this is `2 * V * d * 4` bytes of optimizer state per parameter — the dominant memory cost at large vocabulary.

The Adam update for a row W_i only requires m_i and v_i if W_i received a gradient in this step. For all vocab-sized components, gradients are sparse (only rows in `U` are non-zero).

**Correct approach**: Use `torch.optim.SparseAdam` for `wte` (which also covers `lm_head` since they are tied) and `value_embeds`. SparseAdam only updates the moment tensors for rows that received non-zero gradients:

```
m_i <- beta1 * m_i + (1-beta1) * g_i      # only for i in U
v_i <- beta2 * v_i + (1-beta2) * g_i^2    # only for i in U
W_i <- W_i - lr * m_i / (sqrt(v_i) + eps) # only for i in U
```

This requires gradients to be emitted as `torch.sparse_coo_tensor` or handled via a custom optimizer that accepts `(U, grad_U)` pairs directly.

**`setup_optimizer` must change in sparse mode**: The current code puts `lm_head_params`, `embedding_params`, and `value_embeds_params` all into `kind='adamw'` groups with separate learning rates. In sparse mode with weight tying:
- `lm_head` and `wte` share a single parameter tensor — they become **one** optimizer group with a unified learning rate (see the LR discussion in Section 3.3.0)
- `value_embeds` parameters become a separate sparse optimizer group
- The Muon groups for matrix parameters (`transformer.h`) are **unchanged** — those parameters are not vocab-sized

**Optimizer state memory comparison** at 1M vocab, `n_embd=768`:

| Group | Dense AdamW state (GPU) | Sparse effective per-step GPU usage |
|---|---|---|
| wte (tied, covers lm_head) | 3GB | ~15MB (for 10k rows) |
| value_embeds (x6) | 18GB | ~20MB |
| ~~lm_head (separate)~~ | ~~3GB~~ | eliminated by tying |
| **Total vocab-sized** | **21GB -> 21GB** (3GB eliminated by tying) | **~35MB** |

Weight tying saves one full `(V, d)` moment tensor pair from both VRAM working state and CPU RAM. At 1M vocab, 3GB of optimizer state disappears. Full moment tensors for the remaining groups still exist at size `(V, d)` in CPU RAM (needed for accumulation). Only the `U` rows are brought to GPU during the update step, then scattered back.

---

## 7. DDP and Gradient Synchronization

DDP's `all_reduce` aggregates gradients across ranks after each backward pass. It operates on dense fixed-shape tensors. Sparse gradients require a different approach.

**Problem**: Different ranks process different batches with different `U` sets. The sparse gradient for `wte` on rank 0 has non-zero rows at different positions than rank 1.

### Approach: U-Union Dense Reduction

1. Each rank computes its local `U_rank = unique_tokens(batch_rank)`
2. Broadcast all `U_rank` to all ranks; compute `U_union = union(U_0, ..., U_N)`
3. Each rank gathers `W_{U_union}` rows, runs forward/backward, computes gradients for rows in `U_union`
4. Perform standard dense `all_reduce` on gradient tensor of shape `(|U_union|, d)` — this is small relative to `(V, d)`
5. Apply optimizer update for rows in `U_union`

**Key constraint**: The forward pass for each rank still only uses its own `U_rank` for embedding lookups. But for gradient correctness under DDP averaging, each rank must hold gradient rows for all tokens in `U_union` (i.e., it must compute gradients for tokens it didn't see, with zero gradient for those). In practice, this means the `lm_head` gradient all_reduce only exchanges the non-zero row indices and values.

**At 50k vocab**: `|U_union|` across 8 GPUs with 10k unique tokens each is close to the full 50k. The reduction tensor is at most `(50k, 768) * 4 bytes = 147MB` — comparable to current dense all_reduce. No regression.

**At 1M vocab**: `|U_union|` across 8 GPUs is roughly `8 * 10k = 80k` (with overlap). The reduction tensor is `(80k, 768) * 4 bytes = 235MB` — still far better than `(1M, 768) = 3GB`.

---

## 8. Softcap Compatibility

The current model applies a logit softcap before loss computation:

```python
logits = softcap * torch.tanh(logits / softcap)  # softcap = 15
```

This operation is **fully compatible** with sparse logits. It applies element-wise and requires no knowledge of the full vocabulary. The softcap is applied to `logits_U` (shape `(B, T, |U|)`) before the cross-entropy, with no modification required.

**Correct ordering** (the softcap must come before the loss, but after the importance correction is separated out):

```python
logits_U = x @ W_U.T                                   # (B, T, |U|)
logits_U = logits_U.float()
logits_U = softcap * torch.tanh(logits_U / softcap)    # apply softcap to local logits
loss = F.cross_entropy(logits_U.view(-1, len(U)),
                       local_targets.view(-1),
                       ignore_index=-1,
                       reduction='mean')
loss = loss + math.log(V / len(U))                     # importance correction added to scalar
```

The importance correction term `log(V / |U|)` is added to the **loss scalar** after the softcap has been applied to the logits. The softcap does not interact with the correction term.

---

## 9. Inference Mode

During autoregressive generation, the model samples the next token from the full vocabulary. The output distribution must cover all `V` tokens — there is no batch to restrict to, and restricting the output space would make many tokens permanently unreachable.

**Inference uses full vocab unconditionally.** The `model.generate()` method always uses the dense path:
- `lm_head.weight` is loaded fully to VRAM
- `wte` and `value_embeds` use standard dense `nn.Embedding` lookup
- No importance correction is applied (standard softmax over full V)
- No local index remapping

**Checkpoint compatibility**: Sparse training and dense training produce identical weight tensor shapes. The only difference during training is gradient flow (sparse vs dense) and the training loss objective (importance-weighted vs full softmax). A checkpoint trained with sparse vocab loads into the standard dense inference path without modification.

---

## 10. Memory Analysis

### At 50k Vocabulary (Current Model)

Realistic batch: `B=4, T=2048`. Typical unique token count for BPE tokenization on natural language:

| Quantity | Value |
|---|---|
| Total positions B*T | 8,192 |
| Expected unique tokens |U| | ~8,000–12,000 |
| Fraction of vocabulary covered | ~16–24% |

VRAM for vocab-sized tensors (with weight tying):

| Component | Dense untied | Dense tied | Sparse tied (|U|=10k) | Savings vs dense untied |
|---|---|---|---|---|
| `wte` on VRAM | 77MB | 77MB | 15MB | 62MB |
| `lm_head` rows loaded | 77MB | 0MB (tied) | 0MB (tied) | 77MB |
| `value_embeds` (x6) | ~100MB | ~100MB | ~20MB | 80MB |
| **Logit tensor (fp32)** | **1.6GB** | **1.6GB** | **320MB** | **1.28GB** |
| **Total** | **~1.9GB** | **~1.8GB** | **~355MB** | **~1.55GB** |

The logit tensor dominates at 50k vocab. Tying alone saves 77MB; sparse lm_head saves a further 1.28GB by shrinking the logit tensor.

### At 1M Vocabulary

| Component | Dense untied | Dense tied | Sparse tied (|U|=10k) | Savings vs dense untied |
|---|---|---|---|---|
| `wte` on VRAM | 1.54GB | 1.54GB | 15MB | 1.53GB |
| `lm_head` rows loaded | 1.54GB | 0MB (tied) | 0MB (tied) | 1.54GB |
| `value_embeds` (x6) | ~9GB | ~9GB | ~20MB | ~9GB |
| **Logit tensor (fp32)** | **32GB** | **32GB** | **320MB** | **31.7GB** |
| **Total** | **~44GB** | **~42.5GB** | **~355MB** | **~43.6GB** |

Without sparse vocab, 1M vocab training is physically impossible on a single 80GB H100. With sparse vocab and tied weights it requires roughly the same VRAM as a 50k vocab run. Tying alone (without sparsity) saves 1.54GB at 1M vocab but does not make the problem tractable — you need both tying and sparse logit computation.

### Optimizer State in CPU RAM (at 1M vocab)

| Config | wte moments | lm_head moments | value_embeds moments | Total |
|---|---|---|---|---|
| Dense untied | 3GB | 3GB | 18GB | 24GB |
| Dense tied | 3GB | 0GB (tied) | 18GB | 21GB |
| Sparse tied | 3GB (CPU, sparse access) | 0GB (tied) | 18GB (CPU, sparse access) | 21GB CPU, ~50MB GPU per step |

### What This Does Not Change

- Transformer block parameters (attention, MLP): unchanged, not vocab-sized
- Activation tensors for attention/MLP layers: unchanged
- Optimizer moment tensors in CPU RAM: still full size (V, d) for wte and value_embeds, just accessed sparsely

---

## 11. Implementation Plan

### Phase 1: Core Sparse Vocabulary Layer

**Deliver**: `SparseVocabLayer` in `nanochat/sparse_embedding.py` — a unified layer that handles the embedding-lookup case (`wte`, `value_embeds`) and the projection + loss case (`lm_head`).

For the embedding case:
```python
class SparseEmbedding:
    # CPU-resident weight: (V, d)
    # Forward: gather U rows to GPU, apply F.embedding with local_idx
    # Backward: scatter sparse grad back to CPU rows
    def forward(self, local_idx: Tensor, U: Tensor) -> Tensor: ...
```

For the output head case:
```python
class SparseOutputHead:
    # CPU-resident weight: (V, d)
    # Forward: gather U rows to GPU, compute x @ W_U.T -> logits_U (B, T, |U|)
    # Returns: (logits_U, log_correction)
    def forward(self, x: Tensor, U: Tensor) -> tuple[Tensor, float]: ...
```

`compute_batch_token_set(idx, targets, V, device)` as a standalone function that returns `(U, global_to_local, local_idx, local_targets)`.

### Phase 2: Model Integration

**Changes to `gpt.py`**:

1. `GPT.forward()` in training mode (`targets is not None`) with `sparse_mode=True`:
   - Compute `U, global_to_local, local_idx, local_targets = compute_batch_token_set(idx, targets, V, device)`
   - Use `self.sparse_wte(local_idx, U)` instead of `self.transformer.wte(idx)`
   - Use `self.sparse_value_embeds[i](local_idx, U)` for each active layer
   - Use `logits_U, log_corr = self.sparse_lm_head(x, U)` instead of `self.lm_head(x)`
   - Apply softcap to `logits_U`, compute `F.cross_entropy`, add `log_corr`

2. `GPT.forward()` in inference mode (`targets is None`) or `sparse_mode=False`:
   - Unchanged. Always uses dense `wte`, `value_embeds`, `lm_head`.

3. `GPT.generate()`: No changes. Always dense.

4. `setup_optimizer()`: In sparse mode, `wte`, `value_embeds`, `lm_head` parameter groups use sparse Adam rather than AdamW.

### Phase 3: Optimizer Integration

Implement `SparseCPUAdam` in `nanochat/optim.py`:
- Moment tensors `m` and `v` allocated in CPU pinned memory at shape `(V, d)`
- Per step: given `(U, grad_U)`, fetch `m[U]` and `v[U]` to GPU, apply Adam update, write back
- Compatible with existing optimizer interface

### Phase 4: DDP Integration

Add `U_union` synchronization before the backward pass:
```python
# Before backward
U_all = [torch.zeros_like(U) for _ in range(world_size)]
dist.all_gather(U_all, U)
U_union = torch.unique(torch.cat(U_all))
```

Ensure each rank computes gradients for all tokens in `U_union` and performs dense `all_reduce` on the union-shaped gradient tensors.

### Phase 5: Validation

Run paired experiments from identical seeds comparing sparse vs dense training loss. The quantity to match is `loss_sparse + log(V / |U|)` vs `loss_dense` — they should track within noise across steps.

---

## 12. Testing Plan

### Unit Tests

| Test | What to Verify |
|---|---|
| `test_sparse_embedding_forward` | `SparseEmbedding.forward(local_idx, U)` output matches `nn.Embedding(global_idx)[:, :]` exactly |
| `test_sparse_embedding_backward` | Gradient for rows in `U` matches dense backward; rows outside `U` have zero gradient |
| `test_output_head_forward` | `SparseOutputHead.forward(x, U)` rows match corresponding rows of `x @ lm_head.weight.T` |
| `test_importance_correction_formula` | `loss_sparse + log(V/|U|)` equals `loss_dense` when `U = range(V)` (i.e. full vocab) |
| `test_importance_correction_unbiasedness` | `E[loss_sparse + log(V/|U|)]` converges to `loss_dense` as number of random batches grows |
| `test_token_set_includes_targets` | `compute_batch_token_set` result always contains all values in `targets` |
| `test_local_remapping_roundtrip` | `U[local_idx] == idx` for all positions |
| `test_softcap_with_sparse_logits` | Softcap applied to `logits_U` matches softcap on the same rows of dense `logits` |
| `test_sparse_mode_toggle` | Dense forward output is identical before and after enabling+disabling sparse mode |

### Integration Tests

| Test | What to Verify |
|---|---|
| `test_full_training_step_sparse` | Complete forward + backward + optimizer step runs without error |
| `test_loss_decreases_sparse` | Corrected loss `loss + log(V/|U|)` decreases monotonically over 500 steps |
| `test_checkpoint_roundtrip` | Sparse-trained checkpoint loads and generates identically in dense inference mode |
| `test_generate_full_vocab` | `model.generate()` can produce tokens not seen in any training batch |

### Performance Benchmarks

| Benchmark | Expected Result |
|---|---|
| Peak VRAM at 50k vocab, B=4 | ~1.5GB reduction vs dense |
| Peak VRAM at 1M vocab, B=4 | Dense OOM; sparse feasible |
| Step time at 50k vocab | Within 5% of dense |
| Step time at 500k vocab | Sparse faster due to smaller logit tensor |

---

## Summary of Design Decisions

| Decision | Rationale |
|---|---|
| `wte` and `value_embeds`: exact sparse gather | No approximation needed; gather semantics are already row-local; gradient is naturally sparse |
| `lm_head.weight = wte.weight` (weight tying) | Halves the dominant weight memory cost at large vocab; standard practice; combined gradient remains sparse on U |
| `lm_head`: batch-local softmax | Design choice: the model should learn to discriminate among tokens present in context, not tokens from the entire vocabulary that are never seen together |
| Importance weight `log(V/|U|)` correction | Makes the gradient estimator unbiased in expectation; added to the loss scalar post-softcap |
| Unified init for tied weight | `wte` (std=1.0) and `lm_head` (std=0.001) initializations are incompatible when tied; requires a single chosen init scheme, validated empirically |
| Unified LR for tied weight | `embedding_lr=0.2` and `unembedding_lr=0.004` cannot coexist on one parameter; geometric mean (~0.028) or fresh ablation required |
| Full vocab at inference | The sparse treatment is a training-time gradient shaping decision; generation must reach any token |
| Correction added to loss scalar, not logits | Adding to logits cancels in cross-entropy; correct placement is after `F.cross_entropy` |
| `SparseAdam` for vocab-sized parameters | Dense AdamW moment tensors at full V are the dominant memory cost at large vocab |
| DDP via `U_union` dense reduce | Correct and simple; sparse reduce only needed at very large V |


## 20260301 Model review

Plan: Architectural Review Report — Sparse Mode Training
TL;DR
A thorough review of every layer of the sparse-vocab pipeline — dataloader, CPU remapping, H2D prefetch, GPU forward/backward, log-correction math, VocabRowAdamW optimizer, and async D2H writeback — confirms the remapping algebra and async memory pipeline are arithmetically correct. No data-corruption or indexing bugs were found. However, six distinct issues were identified, of which three are critical and at least one is almost certainly responsible for the loss diverging from ~9.79 back toward ~10.39 (random-guess territory) after step 47.

The Run Log: What the Numbers Say
Observation	Value	Implication
log(V=32768)	10.397 nats	Exact full-vocab random-guess loss
Step-0 raw loss	10.397	Model is at exact random-guess before any update ✓
log_correction	log(32768/20200) ≈ 0.49 nats	Correction applied per step
Step-47 lose	9.79 nats	Real learning — 0.61 nats below random ✓
Step-196 loss	10.39 nats	Near-identical to random-guess again ✗
Final status	Interrupted via KeyboardInterrupt	No convergence
The loss trajectory shows genuine improvement followed by systematic divergence — a classic optimizer-/weight-decay-instability signature, not a random-guessing failure of the correction formula.

Critical Issues (likely root-causes)
Issue 1 — CRITICAL: Weight decay is catastrophically aggressive for this run length
The formula weight_decay_scaled = 0.2 * sqrt(B/B_ref) * (D_ref/D) produces 0.6711 for this run (base_train.py:386). Muon applies this as w *= (1 - lr * wd) each step with matrix_lr = 0.02. Per-step decay ≈ 0.02 × 0.671 = 0.0134.

Over the first 47 steps (before divergence begins):

Cumulative decay ≈ (1-0.0134)^47 ≈ e^{-0.63} ≈ 0.53
All initialized QKV + FFN layers are at ~53% of their initial magnitude by step 47
This is not a calibration problem — the formula is designed for runs with a token budget matching D_ref. This run is 2 000 steps on a d6 model with a much smaller D_ref, making the computed weight_decay_scaled incompatible with the run duration. The c_q, c_k, c_v, and c_fc matrices halve before they can accumulate a usable learning signal. The c_proj / mlp.c_proj are zero-init, so weight-decay actually has no effect on them—but the decayed QKV/FFN matrices mean the attention pattern and feedforward transformation collapse within 50 steps, exactly matching the observed loss upturn at step 47.

Verification step: print block.attn.c_q.weight.norm() every few steps. If it drops monotonically from ~0.9 toward 0, this is confirmed.

Issue 2 — CRITICAL: x0_lambdas learning rate is 354× the per-element update norm
The LR for x0_lambdas is scalar_lr × batch_lr_scale = 0.5 × 0.7071 = 0.354 (base_train.py:472). These are the skip-connection scalars initialized to 0.1 at gpt.py:251.

In each block: x = resid_lambdas[i] * x + x0_lambdas[i] * x0 before block application (gpt.py:550). If x0_lambdas grow (gradient consistently positive), the hidden state at each layer regresses toward the initial normalized embedding x0, progressively washing out context and making every layer output nearly identical. The gradient through these scalars is ∂L/∂λ_x0[i] = ⟨∂L/∂x_input_i, x0⟩ — for a model that's not yet contextually competent, this dot product is systematically nonzero.

With betas=(0.96, 0.95) and lr=0.354, these scalars can accumulate significant bias after 47 steps. If x0_lambdas have grown to e.g. 0.5, the information contribution of the residual stream from the transformer becomes comparable to the constant initial-embedding content, degrading contextual predictions.

Verification step: log model.x0_lambdas every 10 steps via wandb.

Issue 3 — CRITICAL: _v.fill_(0.001) inflates the Adam step size by 31.6× in the first step
In sparse_optim.py:61:

v=0.001 means sqrt(v_hat at step 1) ≈ sqrt(0.001/(1-0.95)) ≈ 0.141, not the expected sqrt(|grad|²). For wte (LR=0.05), the effective step size is 0.05 / 0.141 ≈ 0.35 per element regardless of gradient magnitude. For embedding vectors with std=1.0, this can produce updates of magnitude 0.35 on randomly-initialized vectors — a ~35% change to every active embedding in the first step alone. This explains the raw loss spike to ~17.5 nats at step 1 (logged as EMA 14.17 at step 1). The model recovers slowly from this self-inflicted first-step explosion, but the artificially displaced embeddings create a poor initialisation point for subsequent learning.

The comment says this prevents "first-update explosion on new tokens" — but the initial gradient magnitude for a near-zero lm_head (std=0.001) is already tiny. The large pre-filled _v achieves the opposite of the stated intent at step 1 because early gradients for wte are non-trivially sized (passed through large-std embeddings, scaled by token frequency).

Serious Concerns
Issue 4 — The log_correction approximation becomes less accurate as training proceeds
The correction log(V/|U_step|) is derived under the assumption that all tokens outside U_step have the same logit as the average logit within U_step. This holds near random initialization (all logits ≈ 0). As the model learns, high-frequency tokens (which are overrepresented in U_step) get higher logits, making Z_local an overestimate of |U_step|/V × Z_full. Result: the gradient landscape computed from the local CE is increasingly biased — the model is being trained on an objective that diverges from true cross-entropy more each step.

This doesn't explain divergence on its own, but it means the loss plateau will be above what a properly calibrated model would achieve, and the reported corrected loss will systematically over-estimate the true full-vocab CE as training progresses.

Issue 5 — Only 61% of vocabulary is covered per optimizer step
U_step ≈ 20,000-21,000 from 262,144 tokens across 4 micro-batches. Theoretically 262,144 tokens over a 32,768-token vocab gives ~32,766 unique tokens (birthday paradox). The observed 20K implies highly skewed token frequency distribution — ~39% of tokens receive zero gradient for the embeddings and lm_head rows this step. While this is by design (sparse vocab), it means rare tokens' representations are updated only when they happen to appear in a batch, severely limiting their learning. For the value_embeds tables (3 tables, each managed by VocabRowAdamW), this creates even more severe gradient sparsity.

Issue 6 — train_loss_f is from last micro-batch only
At base_train.py:764: train_loss = loss.detach() inside the micro-batch loop. With grad_accum_steps=4, this overwrites the variable on each iteration — the reported loss and the EMA are computed from only micro-batch index 3 out of 4. This introduces significant step-to-step variance in reported loss independent of any actual training change and makes diagnosing instability from the log harder.

Architecture Verified Correct
The following were verified arithmetically sound:

Token remapping: compute_batch_token_set in sparse_vocab.py:29 correctly maps global→local via global_to_local[U] = arange(|U|). The U_micro→U_step re-index in base_train.py:740 via torch.searchsorted(U_step, U_micro) is correct since U_micro ⊆ U_step by construction of the union.

flush_pending_writes ordering: flush is called before prefetch_to_gpu (base_train.py:700-730), guaranteeing the D2H from the prior step completes before fresh rows are read from master tables. The _d2h_event.synchronize() ensures CPU-GPU ordering.

flush_pending_writes loop body: The actual source at sparse_optim.py:123-127 correctly includes _m[key][U] and _v[key][U] updates inside the for key in self._pin_m: loop.

W_U_* row ordering alignment: W_U_wte = index_select(master_wte, 0, U_step) (sorted). local_idx values are positions in the sorted U_step array. F.embedding(local_idx, W_U_wte) correctly retrieves the embedding for the original token. ✓

Loss gradient is unaffected by log_correction: log_correction_t is a non-differentiable constant tensor (created via torch.tensor(...) without requires_grad). Adding it to the loss shifts the scalar metric but contributes zero gradient to any parameter. ✓

Gradient accumulation: Each micro-batch loss is divided by grad_accum_steps before .backward(). Gradients accumulate in W_U_wte.grad across 4 backward passes, then passed to VocabRowAdamW.step(). Correct average gradient.

GPU-path Adam numerics: Prefetch of _m/_v/w rows fires before the forward pass, runs decay on CPU master tables contemporaneously (ok since flush overwrites active rows), and the D2H writeback correctly uses a dedicated stream with a fence event. The only concern is the _v.fill_(0.001) issue (Issue 3 above).

Validation eval uses dense path: evaluate_bpb (loss_eval.py:9) calls the model without sparse_context, falling through to _cpu_safe_lm_head over the full vocab. Val bpb is therefore a true full-vocab measurement unaffected by sparse-path approximations.

Verification Steps
Decisions

Weight decay formula: designed for longer compute-optimal runs; for short sweep runs, needs a hard cap (e.g. weight_decay_scaled = min(computed_value, 0.05)) or explicit --weight-decay override
_v.fill_(0.001) should be reconsidered — either reduce to 1e-4 so initial Adam steps are more conservative, or initialize _v from a brief warmup pass
x0_lambdas LR should be decoupled from scalar_lr or kept at scalar_lr * 0.01 like resid_lambdas during initial experiments
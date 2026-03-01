"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
import queue
import threading
from dataclasses import asdict
from contextlib import nullcontext, contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.sparse_vocab import _ddp_union_tokens
from nanochat.sparse_optim import VocabRowAdamW
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb, evaluate_bpb_sparse
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()


class AsyncBatchPrefetcher:
    def __init__(self, loader, max_prefetch=2):
        self.loader = loader
        self.queue = queue.Queue(maxsize=max_prefetch)
        self._error = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            for batch in self.loader:
                if isinstance(batch, tuple) and len(batch) == 4:
                    x, y, state, sparse_context = batch
                    sparse_context_cloned = {
                        "U": sparse_context["U"].clone(),
                        "local_idx": sparse_context["local_idx"].clone(),
                        "local_targets": sparse_context["local_targets"].clone(),
                    }
                    self.queue.put((x.clone(), y.clone(), state, sparse_context_cloned))
                elif isinstance(batch, tuple) and len(batch) == 3:
                    x, y, state = batch
                    # Underlying loader reuses persistent buffers; clone so queued
                    # batches stay immutable while producer continues.
                    self.queue.put((x.clone(), y.clone(), state))
                else:
                    self.queue.put(batch)
        except BaseException as exc:
            self._error = exc
        finally:
            self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            if self._error is not None:
                raise self._error
            raise StopIteration
        return item

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
parser.add_argument("--sparse-mode", action="store_true", help="enable dynamic local-vocab sparse training path")
parser.add_argument("--sparse-ddp-union", action="store_true", help="when sparse mode is enabled, synchronize local token sets across DDP ranks")
parser.add_argument("--tie-embeddings", action="store_true", help="tie wte and lm_head weights (recommended for sparse mode)")
parser.add_argument("--no-sparse-prefetch", action="store_true", help="disable AsyncBatchPrefetcher in sparse mode")
parser.add_argument("--sparse-extra-negatives", type=int, default=0, help="number of random non-batch token rows to add to U_step each sparse step (0 = disable)")
parser.add_argument("--sparse-fp32", action="store_true", help="disable autocast in sparse mode to avoid bf16 gradient underflow")
parser.add_argument("--no-sparse-compile", action="store_true", help="disable torch.compile(dynamic=True) for sparse mode")
parser.add_argument("--sparse-optimizer-device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="device for sparse vocab optimizer arithmetic: auto uses current training device, cpu/cuda force a specific backend")
parser.add_argument("--sparse-cuda-deferred-writeback", action="store_true", help="enable deferred CUDA D2H writeback in sparse vocab optimizer")
parser.add_argument("--no-sparse-overlap-cache", action="store_true", help="disable overlap cache that keeps intersecting U_step rows resident on GPU between sparse steps")
parser.add_argument("--sparse-untied-lm-head", action="store_true", help="keep lm_head untied in sparse mode (default is tied)")
parser.add_argument("--sparse-profile", action="store_true", help="print sparse phase timings every N steps")
parser.add_argument("--sparse-profile-every", type=int, default=20, help="emit sparse phase timings every N steps when --sparse-profile is enabled")
parser.add_argument("--sparse-profile-sync", action="store_true", help="synchronize CUDA around profiled sparse phases for attribution accuracy")
parser.add_argument("--sparse-logit-chunk-size", type=int, default=4096, help="number of token positions processed per sparse logit chunk during training")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.05, help="learning rate for embedding/wte parameters (Adam); also controls wte LR in sparse tied-embedding mode")
parser.add_argument("--tied-embedding-lr", type=float, default=0.028, help="learning rate for tied embedding/unembedding when --tie-embeddings is enabled")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--value-embed-lr", type=float, default=None, help="learning rate for value_embed parameters (Adam). If unset, defaults to --embedding-lr to mirror dense-path calibration.")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--vocab-max-grad-norm", type=float, default=0.0, help="per-row gradient norm clip for VocabRowAdamW (0 = disabled)")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--eval-device-batch-size", type=int, default=-1, help="per-device batch size used only for eval (-1 = auto; sparse mode defaults to min(train_bs,16))")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging

if args.sparse_mode and (not args.tie_embeddings) and (not args.sparse_untied_lm_head):
    args.tie_embeddings = True
    print("Sparse mode: auto-enabling --tie-embeddings (use --sparse-untied-lm-head to opt out)")

# SDPA fallback (no FA3) performs poorly with alternating sliding-window patterns.
# Keep user CLI overrides intact, but choose a faster default on non-FA3 systems.
if (not HAS_FA3
    and args.window_pattern == parser.get_default("window_pattern")
    and args.window_pattern != "L"):
    args.window_pattern = "L"
    print(f"No Flash Attention 3 detected: auto-adjusting --window-pattern to '{args.window_pattern}' (override with CLI flag)")
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
sparse_autocast_ctx = nullcontext() if (args.sparse_mode and args.sparse_fp32) else autocast_ctx
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
get_max_reserved_memory = torch.cuda.max_memory_reserved if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

peak_device_memory_used = 0
def update_peak_device_memory_used():
    global peak_device_memory_used
    if device_type == "cuda":
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_mem = total_mem - free_mem
        peak_device_memory_used = max(peak_device_memory_used, used_mem)

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
if HAS_FA3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes_cpu = get_token_bytes(device="cpu")
token_bytes = token_bytes_cpu if device_type == "cpu" else token_bytes_cpu.to(device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
        sparse_mode=args.sparse_mode,
        sparse_ddp_union=args.sparse_ddp_union,
        tie_embeddings=args.tie_embeddings,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized
model.retie_embeddings()

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    model.retie_embeddings()
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> nn.Linear (shares the same weight tensor, no copy)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
if model.config.sparse_mode:
    # Pre-fetched row tensors are created once per optimizer step before the
    # gradient-accumulation loop.  The forward/backward is then graph-break-free.
    # dynamic=True is required because |U_step| (the sub-table size) varies per step.
    sparse_compile_enabled = (not args.sparse_fp32) and (not args.no_sparse_compile)
    if args.sparse_fp32:
        print0("Sparse mode: --sparse-fp32 enabled, skipping torch.compile for dtype-stable fp32 sparse path")
    elif sparse_compile_enabled:
        print0("Sparse mode: compiling with dynamic=True (pre-fetched row tensors, zero graph breaks, dim-0 of W_U tables marked dynamic)")
        model = torch.compile(model, dynamic=True)
    else:
        print0("Sparse mode: running eager (--no-sparse-compile)")
else:
    sparse_compile_enabled = False
    model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
# Cap at the user-specified value: the formula outputs λ_ref × √(B/B_ref) × (D_ref/D), which is designed
# for long compute-optimal runs where target_tokens >> D_ref (so the cap never triggers). For short sweep/debug
# runs (target_tokens << D_ref), the formula overshoots dramatically, decaying transformer matrices to near-zero
# before any useful gradient signal accumulates. Never silently exceed the user's stated intent.
if weight_decay_scaled > args.weight_decay:
    print0(f"Weight decay formula gave {weight_decay_scaled:.6f} (> user value {args.weight_decay:.6f}); capping at user value")
    weight_decay_scaled = args.weight_decay
elif weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    tied_embedding_lr=args.tied_embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer_state = optimizer_data.get("dense", optimizer_data)  # backward compat
    optimizer.load_state_dict(optimizer_state)
    del optimizer_data

# -----------------------------------------------------------------------------
# VocabRowAdamW: sparse-mode optimizer for CPU-resident vocabulary tables.
# Manages wte, value_embeds, and lm_head weights externally; these params have
# requires_grad=False and are NOT inside the MuonAdamW optimizer.
vocab_optimizer = None
vocab_tables = None   # {key: master_weight CPU f32 tensor}
if args.sparse_mode:
    dmodel_lr_scale_v = (model_config.n_embd / 768) ** -0.5
    vocab_emb_lr   = args.embedding_lr   * batch_lr_scale * dmodel_lr_scale_v
    vocab_tied_lr  = args.tied_embedding_lr * batch_lr_scale * dmodel_lr_scale_v
    vocab_unemb_lr = args.unembedding_lr * batch_lr_scale * dmodel_lr_scale_v
    # By default match dense setup_optimizer behaviour: value_embeds use embedding_lr.
    # Users can still override with --value-embed-lr for experiments.
    vocab_ve_lr    = (args.value_embed_lr if args.value_embed_lr is not None else args.embedding_lr) * batch_lr_scale * dmodel_lr_scale_v

    vocab_tables = {"wte": orig_model.wte().weight.data}
    if model_config.tie_embeddings:
        vocab_initial_lrs = {"wte": vocab_tied_lr}
    else:
        vocab_tables["lm_head"] = orig_model.lm_head.weight.data
        vocab_initial_lrs = {"wte": vocab_emb_lr, "lm_head": vocab_unemb_lr}
    for i_str, ve_module in orig_model.value_embeds.items():
        key = f"ve_{i_str}"
        vocab_tables[key] = ve_module.weight.data
        vocab_initial_lrs[key] = vocab_ve_lr

    sparse_optimizer_device = device_type if args.sparse_optimizer_device == "auto" else args.sparse_optimizer_device
    if sparse_optimizer_device == "cuda" and device_type != "cuda":
        print0("Requested --sparse-optimizer-device=cuda without CUDA runtime; falling back to cpu")
        sparse_optimizer_device = "cpu"

    vocab_optimizer = VocabRowAdamW(
        tables=vocab_tables,
        initial_lrs=vocab_initial_lrs,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-10,
        weight_decay=0.0,  # embedding rows are not weight-decayed
        device=sparse_optimizer_device,
        defer_writeback=args.sparse_cuda_deferred_writeback,
        overlap_cache=not args.no_sparse_overlap_cache,
    )
    print0(
        f"VocabRowAdamW: {len(vocab_tables)} tables, vocab_size={vocab_size:,}, "
        f"device={sparse_optimizer_device}, deferred_writeback={args.sparse_cuda_deferred_writeback}, "
        f"overlap_cache={not args.no_sparse_overlap_cache}"
    )
    print0(
        f"Sparse table LRs (scaled): wte={vocab_emb_lr:.6g}, "
        f"lm_head={'tied@' + format(vocab_tied_lr, '.6g') if model_config.tie_embeddings else f'{vocab_unemb_lr:.6g}'}, "
        f"value_embeds={vocab_ve_lr:.6g}"
    )

# Pre-allocate pinned CPU staging buffers for weight-row H2D prefetch.
# Using torch.index_select(..., out=pre_pinned_buf) writes the gathered rows
# directly into page-locked memory — no intermediate pageable allocation
# or separate pin_memory() copy, halving CPU bandwidth for each fetch.
if args.sparse_mode and device_type == "cuda":
    _pin_wte     = torch.empty_like(orig_model.wte().weight, pin_memory=True)
    _pin_lm_head = None if model_config.tie_embeddings else torch.empty_like(orig_model.lm_head.weight, pin_memory=True)
    _pin_ve      = {i_str: torch.empty_like(ve_mod.weight, pin_memory=True)
                    for i_str, ve_mod in orig_model.value_embeds.items()}
else:
    _pin_wte = _pin_lm_head = _pin_ve = None

if resuming and args.sparse_mode and vocab_optimizer is not None:
    if "vocab_optimizer" in meta_data.get("loop_state", {}):
        vocab_optimizer.load_state_dict(meta_data["loop_state"]["vocab_optimizer"])

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)

if args.eval_device_batch_size > 0:
    eval_device_batch_size = args.eval_device_batch_size
elif args.sparse_mode:
    eval_device_batch_size = min(args.device_batch_size, 16)
else:
    eval_device_batch_size = args.device_batch_size
if eval_device_batch_size != args.device_batch_size:
    print0(f"Eval micro-batch size set to {eval_device_batch_size} (train micro-batch is {args.device_batch_size})")

build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, eval_device_batch_size, args.max_seq_len, split="val", device=device)
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    args.device_batch_size,
    args.max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
    # In sparse mode, prepare per-step uniques/local maps in the dataloader
    # (on CPU), and hand them to the model for exact dynamic-shape execution.
    return_sparse_context=args.sparse_mode,
    vocab_size=vocab_size,
)
# NOTE: We do NOT wrap with AsyncBatchPrefetcher here — grad_accum_steps is not
# yet known. We'll wrap below after it's computed so we can set the right depth.
first_batch = next(train_loader) # kick off load of the very first batch of data
if args.sparse_mode:
    x, y, dataloader_state_dict, sparse_context = first_batch
else:
    x, y, dataloader_state_dict = first_batch
    sparse_context = None

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.95 over the first 300 steps)
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linearly decays to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

if args.sparse_mode:
    # Each sparse step drains (grad_accum_steps) batches from the queue per step.
    # Buffer 4× that depth so transient tokenizer hiccups (parquet reads, epoch
    # boundaries, GC) never drain the queue and stall the GPU.
    if not args.no_sparse_prefetch:
        train_loader = AsyncBatchPrefetcher(train_loader, max_prefetch=max(32, (grad_accum_steps + 1) * 4))

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        if args.sparse_mode and vocab_optimizer is not None:
            vocab_optimizer.flush_cache_to_master(vocab_tables)
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (eval_device_batch_size * args.max_seq_len * ddp_world_size)
        if args.sparse_mode and vocab_tables is not None:
            # Sparse eval: only |U_batch| vocab rows on GPU at once — no OOM risk.
            # Uses the same chunked logit path as training and includes log_correction.
            # Run under sparse_autocast_ctx (fp32 when --sparse-fp32, else bf16).
            sparse_row_dtype = torch.float32 if args.sparse_fp32 else torch.bfloat16
            with disable_fp8(orig_model), sparse_autocast_ctx:
                val_bpb = evaluate_bpb_sparse(
                    orig_model, val_loader, eval_steps, token_bytes_cpu,
                    vocab_tables, device, sparse_row_dtype=sparse_row_dtype,
                )
        else:
            with disable_fp8(model), autocast_ctx:
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model), autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        if args.sparse_mode and vocab_optimizer is not None:
            vocab_optimizer.flush_cache_to_master(vocab_tables)
        # Wrap optimizer state to include vocab_optimizer when in sparse mode
        optim_state = optimizer.state_dict()
        if args.sparse_mode and vocab_optimizer is not None:
            optim_state = {"dense": optim_state, "vocab": vocab_optimizer.state_dict()}
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optim_state,
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    t0 = time.time()
    data_wait_time = 0.0
    U_size_log = 0  # number of unique tokens in this step (logged for sparsity visibility)
    log_correction_f = 0.0  # scalar correction added by sparse objective: log(V/|U_step|)
    train_loss_sum = None  # accumulated loss across micro-batches (reset each step)
    profile_this_step = False
    profile_sync = False
    t_phase_dense_step = 0.0
    if args.sparse_mode:
        profile_this_step = args.sparse_profile and (step % max(args.sparse_profile_every, 1) == 0)
        profile_sync = profile_this_step and args.sparse_profile_sync and device_type == "cuda"
        t_phase_union = 0.0
        t_phase_prefetch = 0.0
        t_phase_remap = 0.0
        t_phase_fwdbwd = 0.0
        t_phase_dense_step = 0.0
        t_phase_vocab_step = 0.0

        # ---- 1. Collect ALL micro-batches for this optimizer step upfront. ----
        # We start from the current (x, y, sc) and fetch grad_accum_steps-1 more.
        # All data is on CPU at this point — no GPU work yet.
        all_micro_batches = [(x, y, dataloader_state_dict, sparse_context)]
        t_data0 = time.time()
        for _ in range(grad_accum_steps - 1):
            all_micro_batches.append(next(train_loader))
        data_wait_time += (time.time() - t_data0)

        # ---- 2. U_step = union of all per-micro-batch unique token sets (CPU). ----
        t_union0 = time.time()
        all_U = torch.cat([mb[3]["U"] for mb in all_micro_batches])
        U_step = torch.unique(all_U, sorted=True)
        if model_config.sparse_ddp_union:
            U_step = _ddp_union_tokens(U_step)
        if args.sparse_extra_negatives > 0 and U_step.numel() < model_config.vocab_size:
            need = min(args.sparse_extra_negatives, model_config.vocab_size - U_step.numel())
            extra = torch.randperm(model_config.vocab_size, device=U_step.device, dtype=torch.long)[:need]
            U_step = torch.unique(torch.cat([U_step, extra]), sorted=True)
        if profile_this_step:
            t_phase_union += (time.time() - t_union0)
        U_size_log = U_step.numel()   # Python int — used for logging only, never passed into compiled graph
        # Pre-compute as a GPU scalar tensor so Dynamo guards on its *shape* (always ())
        # rather than its *value*, preventing a guard miss — and therefore a recompile — every step.
        # Clamp to prevent overflow when U_size_log is very small
        log_correction_t = torch.tensor(
            math.log(model_config.vocab_size) - math.log(max(U_size_log, 1)),
            dtype=torch.float32, device=device,
        )
        log_correction_f = float(log_correction_t.item())

        optimizer_uses_cuda = vocab_optimizer is not None and getattr(vocab_optimizer, "_device", device_type) == "cuda"
        optimizer_overlap_cache = optimizer_uses_cuda and bool(getattr(vocab_optimizer, "_overlap_cache", False))

        # ---- 3. H2D: fetch exactly |U_step| rows per vocab table once. --------
        # Each W_U_* is a GPU leaf tensor with requires_grad=True.
        # Grads accumulate across all grad_accum_steps micro-steps.
        #
        # For overlap-cache mode, prefetch_to_gpu() handles pending flush +
        # overlap reconciliation internally. For non-overlap mode, flush first.
        if optimizer_uses_cuda and not optimizer_overlap_cache:
            vocab_optimizer.flush_pending_writes()

        U_size = U_step.numel()
        sparse_row_dtype = torch.float32 if args.sparse_fp32 else torch.bfloat16
        def _fetch_rows(master_w, pin_buf=None):
            if pin_buf is not None:
                # Write gathered rows directly into pre-allocated pinned memory
                # (one CPU copy instead of two; no per-step pinned allocation).
                torch.index_select(master_w, 0, U_step, out=pin_buf[:U_size])
                rows_gpu = pin_buf[:U_size].to(device, dtype=sparse_row_dtype, non_blocking=True)
            else:
                rows_cpu = master_w.index_select(0, U_step)      # CPU (pageable)
                rows_gpu = rows_cpu.pin_memory().to(device, dtype=sparse_row_dtype, non_blocking=True)
            rows_gpu.requires_grad_(True)
            # Tell Dynamo that dim 0 (|U_step|) varies across steps so it never
            # places a static size guard on it — prevents recompilation every step.
            if sparse_compile_enabled:
                torch._dynamo.mark_dynamic(rows_gpu, 0)
            return rows_gpu

        def _rows_from_prefetched(prefetched_rows):
            rows_gpu = prefetched_rows.to(dtype=sparse_row_dtype)
            rows_gpu.requires_grad_(True)
            if sparse_compile_enabled:
                torch._dynamo.mark_dynamic(rows_gpu, 0)
            return rows_gpu

        # ---- 3b. H2D: prefetch active _m/_v/weight rows for GPU-side Adam. ----
        # Uses pre-allocated pinned buffers inside VocabRowAdamW; fires all H2D
        # transfers non-blocking so they arrive during the forward + backward pass.
        if profile_sync:
            torch.cuda.synchronize()
        t_prefetch0 = time.time()
        if optimizer_uses_cuda:
            prefetched_m_gpu, prefetched_v_gpu, prefetched_w_gpu = \
                vocab_optimizer.prefetch_to_gpu(U_step, device, vocab_tables)
        else:
            prefetched_m_gpu = prefetched_v_gpu = prefetched_w_gpu = None
        if profile_sync:
            torch.cuda.synchronize()
        if profile_this_step:
            t_phase_prefetch += (time.time() - t_prefetch0)

        if optimizer_overlap_cache and prefetched_w_gpu is not None:
            W_U_wte = _rows_from_prefetched(prefetched_w_gpu["wte"])
            W_U_lm_head = W_U_wte if model_config.tie_embeddings else _rows_from_prefetched(prefetched_w_gpu["lm_head"])
            W_U_ve = {
                i_str: _rows_from_prefetched(prefetched_w_gpu[f"ve_{i_str}"])
                for i_str in orig_model.value_embeds.keys()
            }
        else:
            W_U_wte     = _fetch_rows(orig_model.wte().weight, _pin_wte)
            W_U_lm_head = W_U_wte if model_config.tie_embeddings else _fetch_rows(orig_model.lm_head.weight, _pin_lm_head)
            W_U_ve      = {i_str: _fetch_rows(ve_mod.weight, _pin_ve.get(i_str) if _pin_ve else None)
                           for i_str, ve_mod in orig_model.value_embeds.items()}

        # ---- 4a. Precompute per-micro-batch remapped index tensors (CPU → GPU). ----
        # All CPU remap work and non-blocking H2D transfers happen here, BEFORE the
        # accumulation loop, while the W_U H2D transfers from step 3 are still in
        # flight.  By the time model() executes there are no CPU stalls between
        # micro-batch forward/backward calls → GPU runs continuously at 100%.
        t_remap0 = time.time()
        mb_local_idx_gpu  = []
        mb_local_tgt_gpu  = []
        for _x_pre, _y_pre, _st_pre, sc_micro in all_micro_batches:
            U_micro  = sc_micro["U"]                             # CPU sorted
            remap    = torch.searchsorted(U_step, U_micro)       # (|U_micro|,) CPU
            l_idx    = sc_micro["local_idx"]                     # (B, T) CPU
            l_tgt    = sc_micro["local_targets"].clone()         # (B, T) CPU
            valid_t  = l_tgt >= 0
            l_tgt[valid_t] = remap[l_tgt[valid_t]]
            mb_local_idx_gpu.append(remap[l_idx].to(device, non_blocking=True))
            mb_local_tgt_gpu.append(l_tgt.to(device, non_blocking=True))
        if profile_this_step:
            t_phase_remap += (time.time() - t_remap0)

        # ---- 4b. Gradient-accumulation loop — purely GPU, zero CPU stalls. -----
        skip_sparse_step = False
        for i_mb, (x_mb, y_mb, _state, _sc_micro) in enumerate(all_micro_batches):
            sparse_ctx_step = {
                "W_U_wte":        W_U_wte,
                "W_U_ve":         W_U_ve,
                "W_U_lm_head":    W_U_lm_head,
                "local_idx":      mb_local_idx_gpu[i_mb],
                "local_targets":  mb_local_tgt_gpu[i_mb],
                "log_correction": log_correction_t,
                "logit_chunk_size": args.sparse_logit_chunk_size,
            }

            if profile_sync:
                torch.cuda.synchronize()
            t_fwdbwd0 = time.time()
            with sparse_autocast_ctx:
                loss = model(x_mb, y_mb, sparse_context=sparse_ctx_step)

            if not torch.isfinite(loss).all():
                print0(f"Non-finite sparse loss at step {step}, micro_batch {i_mb}; skipping optimizer step")
                skip_sparse_step = True
                break

            if train_loss_sum is None:
                train_loss_sum = loss.detach()
            else:
                train_loss_sum = train_loss_sum + loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            if profile_sync:
                torch.cuda.synchronize()
            if profile_this_step:
                t_phase_fwdbwd += (time.time() - t_fwdbwd0)

            # On the last micro-batch, fetch the first batch for the *next* step
            # while the final backward kernels are still executing on GPU.
            # Since next(train_loader) is a queue.get() it returns immediately
            # when the prefetch thread is keeping up, costing near-zero time.
            if i_mb == len(all_micro_batches) - 1:
                t_data0 = time.time()
                next_first = next(train_loader)
                data_wait_time += (time.time() - t_data0)
                x, y, dataloader_state_dict, sparse_context = next_first

        if skip_sparse_step:
            model.zero_grad(set_to_none=True)
            t_data0 = time.time()
            next_first = next(train_loader)
            data_wait_time += (time.time() - t_data0)
            x, y, dataloader_state_dict, sparse_context = next_first
            t1 = time.time()
            dt = t1 - t0
            update_peak_device_memory_used()
            epoch = dataloader_state_dict["epoch"]
            pct_done = 100 * step / num_iterations
            sparse_info = f" | U_step: {U_size_log:,}" if args.sparse_mode else ""
            print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: nan (skipped) | lrm: {get_lr_multiplier(step):.2f} | dt: {dt * 1000:.2f}ms | data_wait: {data_wait_time * 1000:.2f}ms{sparse_info} | epoch: {epoch}")
            first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
            step += 1
            if first_step_of_run:
                gc.collect()
                gc.freeze()
                gc.disable()
            elif step % 5000 == 0:
                gc.collect()
            continue
    else:
        for micro_step in range(grad_accum_steps):
            with sparse_autocast_ctx:
                loss = model(x, y, sparse_context=sparse_context)
            if train_loss_sum is None:
                train_loss_sum = loss.detach()
            else:
                train_loss_sum = train_loss_sum + loss.detach()
            loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
            loss.backward()
            t_data0 = time.time()
            batch_next = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
            data_wait_time += (time.time() - t_data0)
            x, y, dataloader_state_dict = batch_next
            sparse_context = None
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group.get('kind') == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if args.sparse_mode and profile_sync:
        torch.cuda.synchronize()
    t_dense_step0 = time.time()
    optimizer.step()
    if args.sparse_mode and profile_sync:
        torch.cuda.synchronize()
    if args.sparse_mode and profile_this_step:
        t_phase_dense_step += (time.time() - t_dense_step0)

    vocab_grad_max_t = None  # GPU scalar tensor; materialised below in sparse mode for wandb logging
    vocab_update_max_t = None  # GPU/CPU scalar tensor; max row-update norm across sparse vocab tables
    sparse_grad_norms: dict[str, float] = {}
    sparse_update_norms: dict[str, float] = {}

    if args.sparse_mode and vocab_optimizer is not None:
        # Collect W_U_* gradients (GPU bf16).
        # No explicit synchronize() needed: the GPU path runs all Adam arithmetic
        # in the same CUDA stream as backward, so ordering is guaranteed; the
        # D2H writeback inside step() will synchronize when it copies results to CPU.
        vocab_grads = {}
        if W_U_wte.grad is not None:
            grad_wte = W_U_wte.grad
            if ddp and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(grad_wte, op=dist.ReduceOp.AVG)
            vocab_grads["wte"] = grad_wte
        if not model_config.tie_embeddings and W_U_lm_head.grad is not None:
            grad_lm = W_U_lm_head.grad
            if ddp and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(grad_lm, op=dist.ReduceOp.AVG)
            vocab_grads["lm_head"] = grad_lm
        for i_str, W_U_ve_i in W_U_ve.items():
            if W_U_ve_i.grad is not None:
                grad_ve = W_U_ve_i.grad
                if ddp and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(grad_ve, op=dist.ReduceOp.AVG)
                vocab_grads[f"ve_{i_str}"] = grad_ve
           
        # Per-ROW gradient norm clipping (not per-table Frobenius norm).
        # Per-table Frobenius norm scales as sqrt(|U_step| * dim) ≈ sqrt(7.7M) ≈ 2775,
        # so a table-level clip at e.g. 0.3 would reduce every gradient by ~1000x,
        # making the embedding optimizer a no-op regardless of the chosen LR.
        # Per-row clip limits each token row independently, preserving relative magnitudes.
        # In-place on GPU; clamp_(max=1.0) avoids a CPU-GPU sync on every row.
        if args.vocab_max_grad_norm > 0 and vocab_grads:
            for g in vocab_grads.values():
                # g shape: (|U_step|, dim) — clip each row independently.
                row_norms = g.norm(dim=-1, keepdim=True)  # (|U_step|, 1)
                g.mul_((args.vocab_max_grad_norm / (row_norms + 1e-6)).clamp_(max=1.0))
        
        # Record the max per-row grad norm across all vocab tables (post-clip).
        # Materialised via .item() in the logging block below alongside train_loss.item().
        if vocab_grads:
            vocab_grad_max_t = torch.stack([g.norm(dim=-1).amax() for g in vocab_grads.values()]).amax()
            sparse_grad_norms = {k: float(g.norm(dim=-1).amax().item()) for k, g in vocab_grads.items()}

        t_vocab_step0 = time.time()
        if profile_sync:
            torch.cuda.synchronize()
        vocab_update_norms = vocab_optimizer.step(
            U_step, vocab_grads, vocab_tables, lr_multiplier=lrm,
            prefetched_m=prefetched_m_gpu,
            prefetched_v=prefetched_v_gpu,
            prefetched_w=prefetched_w_gpu,
        )
        if profile_sync:
            torch.cuda.synchronize()
        if profile_this_step:
            t_phase_vocab_step += (time.time() - t_vocab_step0)
        if vocab_update_norms:
            vocab_update_max_t = torch.stack(list(vocab_update_norms.values())).amax()
            sparse_update_norms = {k: float(v.item()) for k, v in vocab_update_norms.items()}
        # Free GPU row tensors — they'll be re-created next step from updated master weights.
        del W_U_wte, W_U_lm_head, W_U_ve

    model.zero_grad(set_to_none=True)
    train_loss = train_loss_sum / grad_accum_steps  # average over all micro-batches
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point

    t1 = time.time()
    dt = t1 - t0
    update_peak_device_memory_used()

    # -------------------------------------------------------------------------

    # logging (CPU action only)
    local_train_loss_f = train_loss_f - log_correction_f if args.sparse_mode else train_loss_f
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    sparse_info = f" | U_step: {U_size_log:,}" if args.sparse_mode else ""
    sparse_profile_info = ""
    if args.sparse_mode and profile_this_step:
        t_known = t_phase_union + t_phase_prefetch + t_phase_remap + t_phase_fwdbwd + t_phase_dense_step + t_phase_vocab_step
        sparse_profile_info = (
            f" | sparse(ms): union={t_phase_union*1000:.1f} prefetch={t_phase_prefetch*1000:.1f}"
            f" remap={t_phase_remap*1000:.1f} fwdbwd={t_phase_fwdbwd*1000:.1f}"
            f" dense_step={t_phase_dense_step*1000:.1f} vocab_step={t_phase_vocab_step*1000:.1f}"
            f" gap={max(dt - t_known, 0.0)*1000:.1f}"
        )
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | data_wait: {data_wait_time * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f}{sparse_info}{sparse_profile_info} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/data_wait": data_wait_time,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        if args.sparse_mode:
            log_data["sparse/U_step_size"] = U_size_log
            log_data["sparse/log_correction"] = log_correction_f
            log_data["sparse/local_ce_avg_mb"] = local_train_loss_f  # avg across all micro-batches
            if vocab_grad_max_t is not None:
                log_data["sparse/vocab_grad_norm_max"] = vocab_grad_max_t.item()
                if "wte" in sparse_grad_norms:
                    log_data["sparse/wte_grad_norm"] = sparse_grad_norms["wte"]
                ve_grad_keys = [k for k in sparse_grad_norms if k.startswith("ve_")]
                if ve_grad_keys:
                    log_data["sparse/ve_grad_norm_max"] = max(sparse_grad_norms[k] for k in ve_grad_keys)
            if vocab_update_max_t is not None:
                log_data["sparse/vocab_row_update_norm_max"] = vocab_update_max_t.item()
                if "wte" in sparse_update_norms:
                    log_data["sparse/wte_row_update_norm"] = sparse_update_norms["wte"]
                ve_update_keys = [k for k in sparse_update_norms if k.startswith("ve_")]
                if ve_update_keys:
                    log_data["sparse/ve_row_update_norm_max"] = max(sparse_update_norms[k] for k in ve_update_keys)
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
        if device_type == "cuda":
            # torch.compile(dynamic=True) keeps a large working-memory cache after
            # the first fwd+bwd trace (~19 GB spike).  Release it immediately so
            # VRAM settles back to baseline before the next eval checkpoint.
            torch.cuda.empty_cache()
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage (allocator allocated): {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Peak memory usage (allocator reserved): {get_max_reserved_memory() / 1024 / 1024:.2f}MiB")
if device_type == "cuda":
    print0(f"Peak memory usage (device used): {peak_device_memory_used / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage (allocator allocated)": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        "Peak memory usage (allocator reserved)": f"{get_max_reserved_memory() / 1024 / 1024:.2f}MiB",
        "Peak memory usage (device used)": f"{peak_device_memory_used / 1024 / 1024:.2f}MiB" if device_type == "cuda" else None,
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()

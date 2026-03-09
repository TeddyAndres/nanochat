"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
if os.environ.get("NANOCHAT_ENABLE_EXPANDABLE_SEGMENTS", "0") == "1":
    alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF", "")
    if "expandable_segments" not in alloc_conf:
        alloc_conf = f"{alloc_conf},expandable_segments:True" if alloc_conf else "expandable_segments:True"
        os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit_dynamic, tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.dynamic_vocab import DynamicVocabRuntime
from nanochat.loss_eval import evaluate_bpb_and_ece
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from nanochat.sparse_manifest import load_sparse_manifest_header, validate_sparse_manifest
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--log-every", type=int, default=1, help="print the training step log every N steps (1 = every step)")
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
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--value-embed-lr", type=float, default=-1.0, help="learning rate for value embedding parameters (Adam). -1 reuses --embedding-lr")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--sparse-mode", action="store_true", help="enable first-pass dynamic vocab training (single GPU, grad_accum_steps=1)")
parser.add_argument("--sparse-manifest", type=str, default="", help="path to a precomputed sparse manifest JSON for fixed-U hybrid sparse mode")
parser.add_argument("--sparse-cold-negative-count", type=int, default=0, help="number of sampled cold lm_head rows to append to sparse training logits for normalization correction (0 disables)")
parser.add_argument("--sparse-logit-chunk-size", type=int, default=0, help="reserved for future sparse-logit chunking work")
parser.add_argument("--sparse-empty-cache-every", type=int, default=0, help="in sparse mode, call torch.cuda.empty_cache() every N steps after writeback (0 disables)")
parser.add_argument("--sparse-max-reserved-mib", type=float, default=8192.0, help="in sparse mode, if current CUDA reserved memory exceeds this threshold after a step, trim the cache with torch.cuda.empty_cache() (0 disables)")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--grad-norm-every", type=int, default=100, help="log global gradient norm every N steps (-1 = disable)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
reset_peak_memory = torch.cuda.reset_peak_memory_stats if device_type == "cuda" else lambda: None
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
print0(f"PYTORCH_ALLOC_CONF: {os.environ.get('PYTORCH_ALLOC_CONF', '<unset>')}")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
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

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
sparse_manifest = None
hybrid_sparse = args.sparse_mode and args.sparse_manifest != ""
if hybrid_sparse:
    sparse_manifest = load_sparse_manifest_header(args.sparse_manifest)
if args.sparse_mode:
    assert not ddp, "Sparse mode is single-GPU only for now"
    if hybrid_sparse:
        print0("Sparse hybrid mode: fixed-U manifest path enabled; dense eval/sample paths materialize from CPU masters and training uses manifest-driven overlap reuse")
    else:
        print0("Sparse mode first pass: dense eval/sample paths use temporary full-vocab materialization; checkpoint save/resume is enabled and training metrics focus on the training loop and transfer timings")
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    optimizer_device = "cpu" if args.sparse_mode else None
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank, optimizer_device=optimizer_device)
    if hybrid_sparse:
        checkpoint_manifest = meta_data.get("sparse_manifest", "")
        assert checkpoint_manifest == args.sparse_manifest, (
            f"Sparse manifest mismatch on resume: checkpoint uses '{checkpoint_manifest}', current run uses '{args.sparse_manifest}'"
        )
    model.load_state_dict(model_data, strict=True, assign=True)
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

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
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
if args.sparse_mode and not hybrid_sparse:
    print0("Sparse mode enabled: skipping torch.compile in first-pass dynamic vocab path")
else:
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
sparse_step_tokens = args.device_batch_size * args.max_seq_len * ddp_world_size

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if args.sparse_mode and total_batch_size == -1:
    total_batch_size = sparse_step_tokens
    print0(f"Sparse mode forcing total batch size to one micro-batch: {total_batch_size:,} tokens")
elif total_batch_size == -1:
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

sparse_embedding_lr = args.embedding_lr * batch_lr_scale
sparse_value_embedding_lr = (args.embedding_lr if args.value_embed_lr < 0 else args.value_embed_lr) * batch_lr_scale
sparse_unembedding_lr = args.unembedding_lr * batch_lr_scale

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    value_embedding_lr=(args.embedding_lr if args.value_embed_lr < 0 else args.value_embed_lr) * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    include_vocab_tables=not args.sparse_mode,
)

dynamic_vocab = None
optimizer_data_sparse = None
if args.sparse_mode:
    assert args.sparse_cold_negative_count >= 0, "--sparse-cold-negative-count must be non-negative"
    if args.sparse_cold_negative_count > 0:
        print0(f"Sparse normalization fix: adding {args.sparse_cold_negative_count:,} sampled cold lm_head negatives per step")
    dynamic_vocab = DynamicVocabRuntime(
        orig_model,
        device=device,
        embedding_lr=sparse_embedding_lr,
        value_embedding_lr=sparse_value_embedding_lr,
        unembedding_lr=sparse_unembedding_lr,
        fixed_u_max=(None if not hybrid_sparse else int(sparse_manifest["u_max"])),
        sampled_negative_count=args.sparse_cold_negative_count,
        adam_betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=0.0,
    )
    if resuming:
        optimizer.load_state_dict(optimizer_data["base_optimizer"])
        optimizer_data_sparse = optimizer_data["dynamic_vocab"]
        dynamic_vocab.load_state_dict(optimizer_data_sparse)
    if device_type == "cuda":
        torch.cuda.empty_cache()

if resuming:
    if not args.sparse_mode:
        optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")
if args.sparse_mode:
    assert scaler is None, "Sparse mode does not support fp16 GradScaler yet; use bf16/fp32"

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
if args.sparse_mode:
    if hybrid_sparse:
        train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest(
            tokenizer,
            args.device_batch_size,
            args.max_seq_len,
            split="train",
            manifest_path=args.sparse_manifest,
            device=device,
            resume_state_dict=dataloader_resume_state_dict,
            vocab_size=vocab_size,
        )
    else:
        train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit_dynamic(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict, vocab_size=vocab_size)
else:
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
if args.sparse_mode:
    x, y, sparse_batch_meta, dataloader_state_dict = next(train_loader) # kick off the first sparse batch
else:
    x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

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


def release_eval_memory():
    """Drop cached eval allocations so training memory settles back down."""
    if device_type == "cuda":
        synchronize()
        gc.collect()
        torch.cuda.empty_cache()


def capture_peak_memory_bytes():
    return get_max_memory() if device_type == "cuda" else 0


@torch.no_grad()
def compute_global_grad_norm(optimizer):
    """Compute the global L2 norm of the averaged gradient without mutating optimizer grads."""
    total_sq = torch.tensor(0.0, dtype=torch.float64, device=device)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if world_size > 1:
                grad = grad.float().clone()
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                grad /= world_size
            total_sq += grad.float().pow(2).sum(dtype=torch.float64)
    return total_sq.sqrt().item()

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    val_ece = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    val_ece = meta_data.get("val_ece")
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

peak_memory_usage = capture_peak_memory_bytes()
peak_training_memory_usage = 0

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
if args.sparse_mode:
    assert grad_accum_steps == 1, (
        f"Sparse mode requires grad_accum_steps == 1, got {grad_accum_steps}. "
        f"Set --total-batch-size {world_tokens_per_fwdbwd} for the current settings "
        f"(device_batch_size={args.device_batch_size}, max_seq_len={args.max_seq_len}, world_size={ddp_world_size})."
    )
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
if hybrid_sparse:
    assert sparse_manifest is not None
    validate_sparse_manifest(
        sparse_manifest,
        split="train",
        vocab_size=vocab_size,
        device_batch_size=args.device_batch_size,
        max_seq_len=args.max_seq_len,
        grad_accum_steps=grad_accum_steps,
        ddp_world_size=ddp_world_size,
        num_iterations=num_iterations,
    )
    print0(f"Sparse hybrid manifest: {args.sparse_manifest} | U_max={int(sparse_manifest['u_max']):,}")

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    final_train_step = step == num_iterations - 1
    flops_so_far = num_flops_per_token * total_batch_size * step
    sparse_metrics = None
    dense_eval_model = orig_model
    h2d_gbps = 0.0
    d2h_gbps = 0.0

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        reset_peak_memory()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with torch.inference_mode():
            if args.sparse_mode:
                with dynamic_vocab.materialize_dense_params():
                    with disable_fp8(orig_model):
                        val_bpb, val_ece = evaluate_bpb_and_ece(dense_eval_model, val_loader, eval_steps, token_bytes)
            else:
                with disable_fp8(orig_model):
                    val_bpb, val_ece = evaluate_bpb_and_ece(dense_eval_model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f} | ECE: {val_ece:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
            "val/ece": val_ece,
        })
        model.train()
        peak_memory_usage = max(peak_memory_usage, capture_peak_memory_bytes())
        del val_loader
        release_eval_memory()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        reset_peak_memory()
        with torch.inference_mode():
            if args.sparse_mode:
                with dynamic_vocab.materialize_dense_params():
                    with disable_fp8(orig_model):
                        results = evaluate_core(dense_eval_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
            else:
                with disable_fp8(orig_model):
                    results = evaluate_core(dense_eval_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()
        peak_memory_usage = max(peak_memory_usage, capture_peak_memory_bytes())
        release_eval_memory()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        reset_peak_memory()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        if args.sparse_mode:
            with dynamic_vocab.materialize_dense_params():
                engine = Engine(dense_eval_model, tokenizer) # use orig_model to avoid recompilation
                for prompt in prompts:
                    tokens = tokenizer(prompt, prepend="<|bos|>")
                    with torch.inference_mode():
                        with disable_fp8(orig_model):
                            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                    print0(tokenizer.decode(sample[0]))
        else:
            engine = Engine(dense_eval_model, tokenizer) # use orig_model to avoid recompilation
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                with torch.inference_mode():
                    with disable_fp8(orig_model):
                        sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                print0(tokenizer.decode(sample[0]))
        model.train()
        peak_memory_usage = max(peak_memory_usage, capture_peak_memory_bytes())
        release_eval_memory()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        if hybrid_sparse:
            dynamic_vocab.flush_active_to_cpu()
        optimizer_payload = optimizer.state_dict()
        if args.sparse_mode:
            optimizer_payload = {
                "base_optimizer": optimizer_payload,
                "dynamic_vocab": dynamic_vocab.state_dict(),
            }
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer_payload, # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "val_ece": val_ece,
                "sparse_mode": args.sparse_mode,
                "sparse_manifest": (args.sparse_manifest if hybrid_sparse else ""),
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
    synchronize()
    reset_peak_memory()
    t0 = time.time()
    sparse_step_ctx = None
    for micro_step in range(grad_accum_steps):
        if args.sparse_mode:
            sparse_step_ctx = dynamic_vocab.prepare_step(sparse_batch_meta)
            sparse_metrics = sparse_step_ctx
            loss = model(x, y, active_vocab=sparse_step_ctx.active_vocab)
        else:
            loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if not final_train_step:
            if args.sparse_mode:
                x, y, sparse_batch_meta, dataloader_state_dict = next(train_loader) # prefetch the next sparse batch while GPU is busy with backward
            else:
                x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    should_log_grad_norm = args.grad_norm_every > 0 and (step == 0 or step == num_iterations - 1 or step % args.grad_norm_every == 0)
    grad_norm = None
    if scaler is not None:
        scaler.unscale_(optimizer)
        if should_log_grad_norm:
            grad_norm = compute_global_grad_norm(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        if should_log_grad_norm:
            grad_norm = compute_global_grad_norm(optimizer)
        optimizer.step()
    if args.sparse_mode:
        assert sparse_step_ctx is not None
        sparse_metrics = dynamic_vocab.step(sparse_step_ctx)
        sparse_step_ctx = None
    model.zero_grad(set_to_none=True)
    should_trim_sparse_cache = False
    if (
        args.sparse_mode and
        device_type == "cuda"
    ):
        over_step_interval = args.sparse_empty_cache_every > 0 and ((step + 1) % args.sparse_empty_cache_every == 0)
        over_reserved_limit = args.sparse_max_reserved_mib > 0 and torch.cuda.memory_reserved() > args.sparse_max_reserved_mib * 1024 * 1024
        should_trim_sparse_cache = over_step_interval or over_reserved_limit
        if should_trim_sparse_cache:
            torch.cuda.empty_cache()
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    step_peak_memory = capture_peak_memory_bytes()
    peak_training_memory_usage = max(peak_training_memory_usage, step_peak_memory)
    peak_memory_usage = max(peak_memory_usage, step_peak_memory)
    # -------------------------------------------------------------------------

    # logging (CPU action only)
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
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    grad_norm_str = "" if grad_norm is None else f" | grad_norm: {grad_norm:.4f}"
    sparse_str = ""
    if sparse_metrics is not None:
        sparse_str = f" | U: {sparse_metrics.unique_count:,}"
    should_print_step = (step == 0) or (step == num_iterations - 1) or (args.log_every > 0 and step % args.log_every == 0)
    if should_print_step:
        print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f}{grad_norm_str}{sparse_str} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    should_log_wandb_step = should_print_step or grad_norm is not None
    if should_log_wandb_step:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
            "train/step_peak_memory_mib": step_peak_memory / 1024 / 1024,
        }
        if sparse_metrics is not None:
            log_data.update({
                "train/u": sparse_metrics.unique_count,
            })
        if grad_norm is not None:
            log_data["train/grad_norm"] = grad_norm
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
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {peak_memory_usage / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
if val_ece is not None:
    print0(f"Final validation ece: {val_ece:.6f}")

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
        "Final validation ece": val_ece,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{peak_memory_usage / 1024 / 1024:.2f}MiB",
        "Peak training-step memory usage": f"{peak_training_memory_usage / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()

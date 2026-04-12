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
import queue
import threading
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
from nanochat.dynamic_vocab import COLD_LOGIT_BIAS_CLAMP_MAX, COLD_LOGIT_BIAS_CLAMP_MIN, DynamicVocabRuntime
from nanochat.loss_eval import evaluate_bpb_and_ece
from nanochat.sparse_analysis import SparseLossAnalysisWriter, collect_sparse_loss_topk_from_stats, merge_topk_records, select_topk_records
from nanochat.sparse_replan import SparseFutureWindowPlanner
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from nanochat.sparse_manifest import (
    load_sparse_manifest_header,
    resolve_grouping_base_manifest_path,
    resolve_sparse_manifest_grad_accum_u_max,
    validate_sequence_manifest,
    validate_sparse_manifest,
)
from scripts.base_eval import evaluate_core
print_banner()

SPARSE_RUNTIME_CAPACITY_MULTIPLE = 32


class AsyncLoaderPrefetcher:
    def __init__(self, loader, max_prefetch=2):
        self.loader = loader
        self.queue = queue.Queue(maxsize=max_prefetch)
        self._error = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            while True:
                self.queue.put(next(self.loader))
        except StopIteration:
            pass
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
parser.add_argument("--sparse-logit-scale", type=float, default=1.0, help="multiply sparse training and validation logits by this factor before CE (1.0 disables)")
parser.add_argument("--sparse-cold-bias-scale", type=float, default=0.0, help="sparse-only cold-token bias coefficient; effective magnitude also follows sparse unembedding LR, LR schedule, and total batch size")
parser.add_argument("--sparse-cold-row-decay", "--sparse-cold-row-decrement", dest="sparse_cold_row_decay", type=float, default=0.0, help="sparse-only fixed per-step CPU multiplicative decay applied to lm_head rows outside the next-step sparse table")
parser.add_argument("--sparse-cloud-max-u", type=int, default=0, help="fixed lm_head sparse capacity for step_U + warm + cold rows (0 disables cloud expansion)")
parser.add_argument("--sparse-cloud-warm-proportion", type=float, default=0.5, help="fraction of lm_head cloud capacity to allocate to warm rows; cold receives the remainder")
parser.add_argument("--sparse-unembedding-warm-lr", type=float, default=-1.0, help="lm_head LR for warm cloud rows in sparse mode; negative values reuse --unembedding-lr")
parser.add_argument("--sparse-unembedding-cold-lr", type=float, default=-1.0, help="lm_head LR for cold cloud rows in sparse mode; negative values reuse --unembedding-lr")
parser.add_argument("--sparse-cloud-router-candidate-pool", type=int, default=2048, help="top global-frequency candidate pool size for warm cloud routing on CPU")
parser.add_argument("--sparse-cloud-router-topk", type=int, default=8, help="per-source top-k candidate count aggregated into the warm cloud ranking")
parser.add_argument("--sparse-cloud-hidden-query-samples", type=int, default=32, help="number of subsampled causal positions from the next batch preview used to build warm-cloud hidden queries (0 uses the cheap preview-query path)")
parser.add_argument("--sparse-cloud-hidden-query-strategy", type=str, default="uniform", choices=["uniform", "last"], help="subsampling strategy for warm-cloud hidden-state queries")
parser.add_argument("--sparse-cloud-hidden-query-max-prefix-len", type=int, default=2048, help="maximum causal prefix length used when extracting each warm-cloud hidden query")
parser.add_argument("--sparse-logit-chunk-size", type=int, default=0, help="reserved for future sparse-logit chunking work")
parser.add_argument("--sparse-debug-timing", action="store_true", help="log detailed sparse timing breakdowns for diagnosing sparse runtime overhead")
parser.add_argument("--sparse-debug-sync-after-backward", action="store_true", help="for sparse timing diagnosis, synchronize after each backward pass to separate deferred GPU work from grad-accum bookkeeping")
parser.add_argument("--sparse-empty-cache-every", type=int, default=0, help="in sparse mode, call torch.cuda.empty_cache() every N steps after writeback (0 disables)")
parser.add_argument("--sparse-max-reserved-mib", type=float, default=8192.0, help="in sparse mode, if current CUDA reserved memory exceeds this threshold after a step, trim the cache with torch.cuda.empty_cache() (0 disables)")
parser.add_argument("--lm-head-init-std", type=float, default=-1.0, help="override lm_head init std; negative values keep the model default")
parser.add_argument("--lm-head-init-dist", type=str, default="normal", choices=["normal", "uniform"], help="lm_head initialization distribution when building a fresh model")
parser.add_argument("--sparse-unembed-warmup-steps", type=int, default=0, help="linearly ramp sparse unembedding LR from 0 to its full value over this many steps (0 = disabled)")
parser.add_argument("--sparse-first-hot-unembedding-lr", type=float, default=0.0, help="for lm_head rows only, use this LR the first time a token becomes active, then revert to the configured sparse unembedding LR (0 = disabled)")
parser.add_argument("--sparse-hot-unembed-ramp-activations", type=int, default=0, help="for lm_head rows only, ramp per-token sparse unembedding LR over this many hot activations (0 = disabled)")
parser.add_argument("--sparse-hot-unembed-ramp-start-lr", type=float, default=0.0, help="starting lm_head LR for the per-token hot-activation ramp; used with --sparse-hot-unembed-ramp-activations (0 = disabled)")
parser.add_argument("--sparse-loss-topk-enable", action="store_true", help="capture bounded per-step sparse token-loss top-K tensors for analysis and future replanning")
parser.add_argument("--sparse-loss-topk-correct", type=int, default=50, help="maximum number of under-predicted correct-token records to keep per optimizer step")
parser.add_argument("--sparse-loss-topk-incorrect", type=int, default=50, help="maximum number of over-predicted incorrect-token records to keep per optimizer step")
parser.add_argument("--sparse-loss-topk-output", type=str, default="", help="dataset-side directory for async sparse top-K analysis output (empty = default beside token cache)")
parser.add_argument("--sparse-loss-window-steps", type=int, default=20, help="rolling optimizer-step window used to accumulate sparse loss totals for replanning")
parser.add_argument("--sparse-future-replan-enable", action="store_true", help="use per-step sparse top-K tensors to build delayed future grouping overrides in memory")
parser.add_argument("--sparse-future-replan-lookahead", type=int, default=2, help="number of optimizer steps to delay sparse future-window replanning")
parser.add_argument("--sparse-future-replan-window-steps", type=int, default=8, help="number of future optimizer steps to replan per sparse analysis update")
parser.add_argument("--sparse-auto-negative-per-microstep", type=int, default=4, help="maximum number of automatic corrective cold negatives to inject per replanned microstep")
parser.add_argument("--sparse-replan-sampling-seed", type=int, default=0, help="deterministic seed offset used when sampling from rolling sparse loss lists")
parser.add_argument("--max-grad-norm", type=float, default=0.0, help="clip global gradient norm (dense params only) to this value before optimizer step; 0 = disabled")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
parser.add_argument(
    "--token-cache-dir",
    type=str,
    default="",
    help="token cache directory (empty = sibling folder next to the dataset)",
)
parser.add_argument(
    "--token-cache-shard-batches",
    type=int,
    default=256,
    help="number of tokenized document batches to store per cache shard",
)
parser.add_argument(
    "--token-cache-workers",
    type=int,
    default=0,
    help="number of worker processes to use when building a token cache (0 = auto)",
)
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=5*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--grad-norm-every", type=int, default=10, help="log global gradient norm every N steps (-1 = disable)")
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
    #num_heads = 4
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
lm_head_init_std = None if args.lm_head_init_std < 0.0 else args.lm_head_init_std
if lm_head_init_std is not None:
    print0(f"Overriding lm_head init std to {lm_head_init_std:.6f}")
if args.lm_head_init_dist != "normal":
    print0(f"Overriding lm_head init distribution to {args.lm_head_init_dist}")
model.init_weights(lm_head_init_std=lm_head_init_std, lm_head_init_dist=args.lm_head_init_dist) # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
sparse_manifest = None
sparse_base_manifest = None
sparse_base_manifest_path = ""
sparse_resolved_grad_accum_u_max = None
hybrid_sparse = args.sparse_mode and args.sparse_manifest != ""
sparse_lm_head_clouds = args.sparse_cloud_max_u > 0
if hybrid_sparse:
    sparse_manifest = load_sparse_manifest_header(args.sparse_manifest)
    if "base_manifest_path" in sparse_manifest:
        sparse_base_manifest_path = str(resolve_grouping_base_manifest_path(args.sparse_manifest, sparse_manifest))
        sparse_base_manifest = load_sparse_manifest_header(sparse_base_manifest_path)
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
        checkpoint_base_manifest = meta_data.get("sparse_base_manifest", "")
        assert checkpoint_base_manifest == sparse_base_manifest_path, (
            f"Sparse base manifest mismatch on resume: checkpoint uses '{checkpoint_base_manifest}', current run uses '{sparse_base_manifest_path}'"
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
if args.sparse_mode and not hybrid_sparse and total_batch_size == -1:
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
    assert args.sparse_logit_scale > 0.0, "--sparse-logit-scale must be positive"
    assert args.sparse_cold_bias_scale >= 0.0, "--sparse-cold-bias-scale must be non-negative"
    assert 0.0 <= args.sparse_cold_row_decay <= 1.0, "--sparse-cold-row-decay must be in [0, 1]"
    assert 0.0 <= args.sparse_cloud_warm_proportion <= 1.0, "--sparse-cloud-warm-proportion must be in [0, 1]"
    if sparse_lm_head_clouds:
        assert hybrid_sparse, "--sparse-cloud-max-u requires --sparse-manifest hybrid sparse mode"
    if args.sparse_logit_scale != 1.0:
        print0(f"Sparse logit scaling enabled: multiplying sparse train/val logits by {args.sparse_logit_scale:.4f} before CE")
    if args.sparse_cold_bias_scale > 0.0:
        print0(f"Sparse cold-token bias enabled: base_scale={args.sparse_cold_bias_scale:.4f}")
    if args.sparse_cold_row_decay > 0.0:
        print0(f"Sparse cold-row decay enabled: base_decay={args.sparse_cold_row_decay:.6f}")
    sparse_fixed_u_max = None
    sparse_lm_head_u_max = None
    sparse_grad_accum_u_max = None
    sparse_runtime_capacity_multiple = 1
    if hybrid_sparse:
        assert sparse_manifest is not None
        sparse_fixed_u_max = int(sparse_manifest["u_max"])
        sparse_resolved_grad_accum_u_max = resolve_sparse_manifest_grad_accum_u_max(args.sparse_manifest, sparse_manifest)
        sparse_grad_accum_u_max = sparse_resolved_grad_accum_u_max
        sparse_lm_head_u_max = sparse_fixed_u_max if not sparse_lm_head_clouds else int(args.sparse_cloud_max_u)
        sparse_runtime_capacity_multiple = SPARSE_RUNTIME_CAPACITY_MULTIPLE
        if sparse_lm_head_u_max < sparse_fixed_u_max:
            raise ValueError(
                f"--sparse-cloud-max-u must be at least manifest u_max={sparse_fixed_u_max}, got {sparse_lm_head_u_max}"
            )
        if sparse_lm_head_clouds:
            warm_lr = sparse_unembedding_lr if args.sparse_unembedding_warm_lr < 0.0 else float(args.sparse_unembedding_warm_lr)
            cold_lr = sparse_unembedding_lr if args.sparse_unembedding_cold_lr < 0.0 else float(args.sparse_unembedding_cold_lr)
            print0(
                f"Sparse lm_head clouds enabled: manifest_u_max={sparse_fixed_u_max:,} | "
                f"lm_head_u_max={sparse_lm_head_u_max:,} | warm_fraction={args.sparse_cloud_warm_proportion:.2f} | "
                f"warm_lr={warm_lr:.6f} | cold_lr={cold_lr:.6f}"
            )
    dynamic_vocab = DynamicVocabRuntime(
        orig_model,
        device=device,
        embedding_lr=sparse_embedding_lr,
        value_embedding_lr=sparse_value_embedding_lr,
        unembedding_lr=sparse_unembedding_lr,
        first_hot_unembedding_lr=args.sparse_first_hot_unembedding_lr,
        hot_unembedding_ramp_activations=args.sparse_hot_unembed_ramp_activations,
        hot_unembedding_ramp_start_lr=args.sparse_hot_unembed_ramp_start_lr,
        fixed_u_max=sparse_fixed_u_max,
        lm_head_u_max=sparse_lm_head_u_max,
        grad_accum_u_max=sparse_grad_accum_u_max,
        capacity_round_multiple=sparse_runtime_capacity_multiple,
        cold_bias_reference_tokens=B_REF,
        unembedding_warm_lr=(None if args.sparse_unembedding_warm_lr < 0.0 else args.sparse_unembedding_warm_lr),
        unembedding_cold_lr=(None if args.sparse_unembedding_cold_lr < 0.0 else args.sparse_unembedding_cold_lr),
        adam_betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=0.0,
    )
    if hybrid_sparse:
        print0(
            f"Sparse runtime capacity alignment: multiple={sparse_runtime_capacity_multiple} | "
            f"fixed_u_max {sparse_fixed_u_max:,}->{dynamic_vocab.fixed_u_max:,} | "
            f"lm_head_u_max {sparse_lm_head_u_max:,}->{dynamic_vocab.lm_head_u_max:,} | "
            f"grad_accum_u_max {sparse_grad_accum_u_max:,}->{dynamic_vocab.grad_accum_u_max:,}"
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
token_cache_workers = max(1, args.token_cache_workers) if args.token_cache_workers > 0 else max(1, min(8, os.cpu_count() or 1))
sparse_loss_analysis_writer = None
sparse_future_window_planner = None
if args.sparse_loss_topk_enable:
    assert args.sparse_mode, "--sparse-loss-topk-enable requires --sparse-mode"
    sparse_loss_analysis_writer = SparseLossAnalysisWriter(args.sparse_loss_topk_output, args.token_cache_dir)
if args.sparse_future_replan_enable:
    assert args.sparse_mode, "--sparse-future-replan-enable requires --sparse-mode"
    assert hybrid_sparse, "--sparse-future-replan-enable requires --sparse-manifest"
    assert args.sparse_loss_topk_enable, "--sparse-future-replan-enable requires --sparse-loss-topk-enable"
    sparse_future_window_planner = SparseFutureWindowPlanner(
        args.sparse_manifest,
        lookahead_steps=args.sparse_future_replan_lookahead,
        window_steps=args.sparse_future_replan_window_steps,
        rolling_window_steps=args.sparse_loss_window_steps,
        max_auto_negatives_per_microstep=args.sparse_auto_negative_per_microstep,
        sampling_seed=args.sparse_replan_sampling_seed,
    )
    if resuming and args.sparse_mode:
        planner_state = optimizer_data.get("sparse_planner") if isinstance(optimizer_data, dict) else None
        if isinstance(planner_state, dict):
            sparse_future_window_planner.load_state_dict(planner_state)
if args.sparse_mode:
    if hybrid_sparse:
        train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit_manifest(
            tokenizer,
            args.device_batch_size,
            args.max_seq_len,
            split="train",
            manifest_path=args.sparse_manifest,
            device="cpu",
            resume_state_dict=dataloader_resume_state_dict,
            pin_memory_output=(device_type == "cuda"),
            vocab_size=vocab_size,
            include_local_batch=sparse_lm_head_clouds,
            step_override_provider=None if sparse_future_window_planner is None else sparse_future_window_planner.get_step_override,
            token_cache_dir=args.token_cache_dir,
            token_cache_shard_batches=args.token_cache_shard_batches,
            token_cache_workers=token_cache_workers,
        )
        train_loader = AsyncLoaderPrefetcher(train_loader, max_prefetch=2)
    else:
        train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit_dynamic(
            tokenizer,
            args.device_batch_size,
            args.max_seq_len,
            split="train",
            device="cpu",
            resume_state_dict=dataloader_resume_state_dict,
            pin_memory_output=(device_type == "cuda"),
            vocab_size=vocab_size,
            token_cache_dir=args.token_cache_dir,
            token_cache_shard_batches=args.token_cache_shard_batches,
            token_cache_workers=token_cache_workers,
        )
        train_loader = AsyncLoaderPrefetcher(train_loader, max_prefetch=2)
else:
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        split="train",
        device=device,
        resume_state_dict=dataloader_resume_state_dict,
        token_cache_dir=args.token_cache_dir,
        token_cache_shard_batches=args.token_cache_shard_batches,
        token_cache_workers=token_cache_workers,
    )
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer,
    args.device_batch_size,
    args.max_seq_len,
    split="val",
    device=device,
    token_cache_dir=args.token_cache_dir,
    token_cache_shard_batches=args.token_cache_shard_batches,
    token_cache_workers=token_cache_workers,
)


def plan_sparse_batch_meta(step_meta):
    if not sparse_lm_head_clouds:
        return step_meta
    assert dynamic_vocab is not None
    return dynamic_vocab.plan_next_lm_head_cloud(
        step_meta,
        warm_proportion=args.sparse_cloud_warm_proportion,
        router_candidate_pool_size=args.sparse_cloud_router_candidate_pool,
        router_topk=args.sparse_cloud_router_topk,
        source_token_limit=args.sparse_cloud_hidden_query_samples,
        hidden_query_strategy=args.sparse_cloud_hidden_query_strategy,
        hidden_query_max_prefix_len=args.sparse_cloud_hidden_query_max_prefix_len,
    )


def stage_batch_to_device(batch_x, batch_y):
    if torch.device(device).type == "cpu":
        return batch_x, batch_y
    return batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)


if args.sparse_mode:
    pass

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target FLOPs (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
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


def get_sparse_cold_bias_scale(it):
    if not args.sparse_mode or args.sparse_cold_bias_scale <= 0.0:
        return 0.0
    return args.sparse_cold_bias_scale


def get_sparse_cold_row_decay(it):
    if not args.sparse_mode or args.sparse_cold_row_decay <= 0.0:
        return 0.0
    return args.sparse_cold_row_decay * get_lr_multiplier(it)


if args.sparse_mode and args.sparse_cold_bias_scale > 0.0:
    bias_examples = []
    for cold_steps in (1, 10, 100):
        raw_bias = args.sparse_cold_bias_scale * math.log1p(cold_steps * total_batch_size / B_REF)
        clamped_bias = max(min(raw_bias, COLD_LOGIT_BIAS_CLAMP_MAX), COLD_LOGIT_BIAS_CLAMP_MIN)
        bias_examples.append(f"{cold_steps}: raw={raw_bias:.2f}, clamped={clamped_bias:.2f}")
    print0(
        f"Sparse cold-token bias: scale={args.sparse_cold_bias_scale:.4f}, "
        f"first_seen_bias=0, reference_tokens={B_REF:,}, total_batch={total_batch_size:,}, "
        f"clamp=[{COLD_LOGIT_BIAS_CLAMP_MIN:.0f}, {COLD_LOGIT_BIAS_CLAMP_MAX:.0f}], "
        f"bias_examples(steps -> raw/clamped): {'; '.join(bias_examples)}"
    )

if args.sparse_mode and args.sparse_cold_row_decay > 0.0:
    decay_start = get_sparse_cold_row_decay(0)
    decay_mid = get_sparse_cold_row_decay(max(num_iterations // 2, 0))
    decay_end = get_sparse_cold_row_decay(max(num_iterations - 1, 0))
    print0(
        f"Sparse cold-row decay schedule: effective_decay(start/mid/end)={decay_start:.6f}/{decay_mid:.6f}/{decay_end:.6f}, "
        f"effective_multiplier(start/mid/end)={1.0 - decay_start:.6f}/{1.0 - decay_mid:.6f}/{1.0 - decay_end:.6f}"
    )

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
def iter_grad_params(optimizer, extra_params=None):
    seen = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            yield param
    if extra_params is None:
        return
    for param in extra_params:
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        yield param


def compute_global_grad_norm(optimizer, extra_params=None):
    """Compute the global L2 norm of the averaged gradient without mutating optimizer grads."""
    total_sq = torch.tensor(0.0, dtype=torch.float64, device=device)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    for param in iter_grad_params(optimizer, extra_params=extra_params):
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
    if not hybrid_sparse:
        assert grad_accum_steps == 1, (
            f"First-pass dynamic sparse mode requires grad_accum_steps == 1, got {grad_accum_steps}. "
            f"Set --total-batch-size {world_tokens_per_fwdbwd} for the current settings "
            f"(device_batch_size={args.device_batch_size}, max_seq_len={args.max_seq_len}, world_size={ddp_world_size})."
        )
    if sparse_lm_head_clouds:
        assert grad_accum_steps == 1, "lm_head cloud expansion currently requires grad_accum_steps == 1"
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
if hybrid_sparse:
    assert sparse_manifest is not None
    resolved_grad_accum_u_max = int(sparse_resolved_grad_accum_u_max if sparse_resolved_grad_accum_u_max is not None else resolve_sparse_manifest_grad_accum_u_max(args.sparse_manifest, sparse_manifest))
    manifest_num_steps = int(sparse_manifest["num_steps"])
    if num_iterations > manifest_num_steps:
        raise ValueError(
            f"Requested num_iterations={num_iterations} exceeds sparse manifest length {manifest_num_steps}. "
            f"Shorter runs are allowed; longer runs require a longer manifest."
        )
    validate_sparse_manifest(
        sparse_manifest,
        split="train",
        vocab_size=vocab_size,
        device_batch_size=args.device_batch_size,
        max_seq_len=args.max_seq_len,
        grad_accum_steps=grad_accum_steps,
        ddp_world_size=ddp_world_size,
    )
    if sparse_base_manifest is not None:
        validate_sequence_manifest(
            sparse_base_manifest,
            split="train",
            vocab_size=vocab_size,
            device_batch_size=args.device_batch_size,
            max_seq_len=args.max_seq_len,
            ddp_world_size=ddp_world_size,
        )
    print0(
        f"Sparse hybrid manifest: {args.sparse_manifest} | "
        f"manifest_U_max={int(sparse_manifest['u_max']):,} | "
        f"model_U_max={dynamic_vocab.fixed_u_max:,} | "
        f"model_lm_head_U_max={dynamic_vocab.lm_head_u_max:,} | "
        f"manifest_grad_accum_U_max={resolved_grad_accum_u_max:,} | "
        f"model_grad_accum_U_max={dynamic_vocab.grad_accum_u_max:,}"
    )
    if sparse_base_manifest is not None:
        print0(
            f"Sparse base sequence manifest: {sparse_base_manifest_path} | "
            f"sequence_units={int(sparse_base_manifest['num_sequence_units']):,}"
        )

if args.sparse_mode:
    startup_fetch_t0 = time.perf_counter()
    x, y, sparse_batch_meta, dataloader_state_dict = next(train_loader) # kick off the first sparse batch
    sparse_batch_meta = plan_sparse_batch_meta(sparse_batch_meta)
    x, y = stage_batch_to_device(x, y)
    if args.sparse_debug_timing:
        print0(f"Sparse startup first-batch fetch: {(time.perf_counter() - startup_fetch_t0) * 1000.0:.2f}ms")
else:
    x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

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
                        val_bpb, val_ece = evaluate_bpb_and_ece(
                            dense_eval_model,
                            val_loader,
                            eval_steps,
                            token_bytes,
                            logit_scale=args.sparse_logit_scale,
                        )
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
            if sparse_future_window_planner is not None:
                optimizer_payload["sparse_planner"] = sparse_future_window_planner.state_dict()
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
                "sparse_base_manifest": (sparse_base_manifest_path if hybrid_sparse else ""),
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
    sparse_window_metrics = None
    train_loss_accum = None
    sparse_prepare_ms = 0.0
    sparse_prep_writeback_wait_ms = 0.0
    sparse_prep_prefetch_wait_ms = 0.0
    sparse_prep_cpu_gather_ms = 0.0
    sparse_prep_h2d_enqueue_ms = 0.0
    sparse_prep_h2d_tensor_count = 0
    sparse_prep_h2d_bytes = 0
    sparse_prep_prefetch_hits = 0
    sparse_fwdbwd_ms = 0.0
    sparse_input_clone_ms = 0.0
    sparse_forward_call_ms = 0.0
    sparse_backward_call_ms = 0.0
    sparse_next_fetch_ms = 0.0
    sparse_loader_fetch_ms = 0.0
    sparse_plan_meta_ms = 0.0
    sparse_row_prefetch_launch_ms = 0.0
    sparse_batch_h2d_ms = 0.0
    sparse_backward_sync_ms = 0.0
    sparse_accum_call_ms = 0.0
    sparse_accum_start_ms = 0.0
    sparse_accum_flush_ms = 0.0
    sparse_accum_queue_ms = 0.0
    sparse_accum_rows_queued = 0
    sparse_apply_call_ms = 0.0
    sparse_cold_bias = get_sparse_cold_bias_scale(step)
    sparse_cold_row_decay = get_sparse_cold_row_decay(step)
    step_correct_scores_all = None
    step_correct_records_all = None
    step_incorrect_scores_all = None
    step_incorrect_records_all = None
    next_x = None
    next_y = None
    next_sparse_batch_meta = None
    next_dataloader_state_dict = None
    for micro_step in range(grad_accum_steps):
        micro_t0 = time.perf_counter()
        clone_t0 = time.perf_counter()
        current_x = x.clone()
        current_y = y.clone()
        sparse_input_clone_ms += (time.perf_counter() - clone_t0) * 1000.0
        if args.sparse_mode:
            prepare_t0 = time.perf_counter()
            sparse_step_ctx = dynamic_vocab.prepare_step(
                sparse_batch_meta,
                cold_bias_scale=sparse_cold_bias,
                cold_row_decay=sparse_cold_row_decay,
                cold_bias_tokens_per_step=total_batch_size,
            )
            sparse_prepare_ms += (time.perf_counter() - prepare_t0) * 1000.0
            sparse_prep_writeback_wait_ms += sparse_step_ctx.prep_writeback_wait_ms
            sparse_prep_prefetch_wait_ms += sparse_step_ctx.prep_prefetch_wait_ms
            sparse_prep_cpu_gather_ms += sparse_step_ctx.prep_cpu_gather_ms
            sparse_prep_h2d_enqueue_ms += sparse_step_ctx.prep_h2d_enqueue_ms
            sparse_prep_h2d_tensor_count += sparse_step_ctx.prep_h2d_tensor_count
            sparse_prep_h2d_bytes += sparse_step_ctx.prep_h2d_bytes
            sparse_prep_prefetch_hits += sparse_step_ctx.prep_prefetch_hit
            sparse_metrics = sparse_step_ctx
            x_for_model = sparse_step_ctx.union_inputs if sparse_step_ctx.union_inputs is not None else current_x
            y_for_loss = sparse_step_ctx.union_targets if sparse_step_ctx.union_targets is not None else y
            analysis_logsumexp = None
            analysis_top2_logits = None
            analysis_top2_local = None
            analysis_target_logits = None
            forward_t0 = time.perf_counter()
            model_result = model(
                x_for_model,
                y_for_loss,
                active_vocab=sparse_step_ctx.active_vocab,
                logit_scale=args.sparse_logit_scale,
                return_sparse_analysis=args.sparse_loss_topk_enable,
            )
            sparse_forward_call_ms += (time.perf_counter() - forward_t0) * 1000.0
            if args.sparse_loss_topk_enable:
                loss, analysis_logsumexp, analysis_top2_logits, analysis_top2_local, analysis_target_logits = model_result
            else:
                loss = model_result
            if args.sparse_loss_topk_enable:
                analysis_active_ids_cpu = sparse_step_ctx.lm_head_active_ids_cpu if sparse_step_ctx.lm_head_active_ids_cpu is not None else sparse_step_ctx.active_ids_cpu
                assert analysis_logsumexp is not None
                assert analysis_top2_logits is not None
                assert analysis_top2_local is not None
                assert analysis_target_logits is not None
                analysis_token_losses = analysis_logsumexp.detach() - analysis_target_logits.detach().to(dtype=torch.float32)
                with torch.no_grad():
                    analysis_payload = collect_sparse_loss_topk_from_stats(
                        y_for_loss,
                        analysis_active_ids_cpu,
                        topk_correct=None,
                        topk_incorrect=None,
                        step=step,
                        micro_step=micro_step,
                        sequence_id=int(sparse_batch_meta.get("sequence_id", -1)),
                        losses=analysis_token_losses.detach(),
                        top2_logits=analysis_top2_logits.detach(),
                        top2_local=analysis_top2_local.detach(),
                        target_logits=analysis_target_logits.detach(),
                    )
                step_correct_scores_all, step_correct_records_all = merge_topk_records(
                    step_correct_scores_all,
                    step_correct_records_all,
                    analysis_payload["correct_scores"],
                    analysis_payload["correct_records"],
                    topk=None,
                )
                step_incorrect_scores_all, step_incorrect_records_all = merge_topk_records(
                    step_incorrect_scores_all,
                    step_incorrect_records_all,
                    analysis_payload["incorrect_scores"],
                    analysis_payload["incorrect_records"],
                    topk=None,
                )
        else:
            forward_t0 = time.perf_counter()
            loss = model(current_x, current_y)
            sparse_forward_call_ms += (time.perf_counter() - forward_t0) * 1000.0
        micro_loss = loss.detach()
        train_loss_accum = micro_loss if train_loss_accum is None else (train_loss_accum + micro_loss)
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        backward_t0 = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        sparse_backward_call_ms += (time.perf_counter() - backward_t0) * 1000.0
        if not final_train_step:
            fetch_t0 = time.perf_counter()
            if args.sparse_mode:
                loader_fetch_t0 = time.perf_counter()
                next_x, next_y, next_sparse_batch_meta, next_dataloader_state_dict = next(train_loader)
                sparse_loader_fetch_ms += (time.perf_counter() - loader_fetch_t0) * 1000.0
                plan_t0 = time.perf_counter()
                next_sparse_batch_meta = plan_sparse_batch_meta(next_sparse_batch_meta)
                sparse_plan_meta_ms += (time.perf_counter() - plan_t0) * 1000.0
                h2d_t0 = time.perf_counter()
                next_x, next_y = stage_batch_to_device(next_x, next_y)
                sparse_batch_h2d_ms += (time.perf_counter() - h2d_t0) * 1000.0
                prefetch_launch_t0 = time.perf_counter()
                dynamic_vocab.prefetch_step(next_sparse_batch_meta)
                sparse_row_prefetch_launch_ms += (time.perf_counter() - prefetch_launch_t0) * 1000.0
            else:
                loader_fetch_t0 = time.perf_counter()
                next_x, next_y, next_dataloader_state_dict = next(train_loader)
                sparse_loader_fetch_ms += (time.perf_counter() - loader_fetch_t0) * 1000.0
            sparse_next_fetch_ms += (time.perf_counter() - fetch_t0) * 1000.0
        if args.sparse_mode and args.sparse_debug_sync_after_backward and device_type == "cuda":
            backward_sync_t0 = time.perf_counter()
            synchronize()
            sparse_backward_sync_ms += (time.perf_counter() - backward_sync_t0) * 1000.0
        sparse_fwdbwd_ms += (time.perf_counter() - micro_t0) * 1000.0
        if args.sparse_mode and grad_accum_steps > 1:
            assert sparse_step_ctx is not None
            accum_t0 = time.perf_counter()
            sparse_window_metrics = dynamic_vocab.accumulate_gradients(sparse_step_ctx)
            sparse_accum_call_ms += (time.perf_counter() - accum_t0) * 1000.0
            sparse_accum_start_ms += sparse_window_metrics.grad_accum_start_ms
            sparse_accum_flush_ms += sparse_window_metrics.grad_accum_flush_ms
            sparse_accum_queue_ms += sparse_window_metrics.grad_accum_queue_ms
            sparse_accum_rows_queued += sparse_window_metrics.grad_accum_queue_count
            sparse_step_ctx = None
        if not final_train_step:
            if args.sparse_mode:
                x = next_x
                y = next_y
                sparse_batch_meta = next_sparse_batch_meta
                dataloader_state_dict = next_dataloader_state_dict
            else:
                x = next_x
                y = next_y
                dataloader_state_dict = next_dataloader_state_dict
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if args.sparse_mode:
        dynamic_vocab.table_specs["wte"]["lr"] = sparse_embedding_lr * lrm
        for name in dynamic_vocab.table_specs:
            if name.startswith("value_embeds."):
                dynamic_vocab.table_specs[name]["lr"] = sparse_value_embedding_lr * lrm
        scheduled_unembedding_lr = sparse_unembedding_lr * lrm
        if args.sparse_unembed_warmup_steps > 0:
            warmup_scale = min(1.0, (step + 1) / args.sparse_unembed_warmup_steps)
            scheduled_unembedding_lr *= warmup_scale
        dynamic_vocab.table_specs["lm_head"]["lr"] = scheduled_unembedding_lr
        warm_unembedding_lr = sparse_unembedding_lr if args.sparse_unembedding_warm_lr < 0.0 else args.sparse_unembedding_warm_lr * batch_lr_scale
        cold_unembedding_lr = sparse_unembedding_lr if args.sparse_unembedding_cold_lr < 0.0 else args.sparse_unembedding_cold_lr * batch_lr_scale
        dynamic_vocab.table_specs["lm_head"]["warm_lr"] = warm_unembedding_lr * lrm
        dynamic_vocab.table_specs["lm_head"]["cold_lr"] = cold_unembedding_lr * lrm
    should_log_grad_norm = args.grad_norm_every > 0 and (step == 0 or step == num_iterations - 1 or step % args.grad_norm_every == 0)
    grad_norm = None
    sparse_grad_params = None if not args.sparse_mode else dynamic_vocab.fixed_params.values()
    if scaler is not None:
        scaler.unscale_(optimizer)
        if should_log_grad_norm:
            grad_norm = compute_global_grad_norm(optimizer, extra_params=sparse_grad_params)
        if args.max_grad_norm > 0.0:
            params_to_clip = [p for p in iter_grad_params(optimizer, extra_params=sparse_grad_params) if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
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
            grad_norm = compute_global_grad_norm(optimizer, extra_params=sparse_grad_params)
        if args.max_grad_norm > 0.0:
            params_to_clip = [p for p in iter_grad_params(optimizer, extra_params=sparse_grad_params) if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        optimizer.step()
    if args.sparse_mode:
        if grad_accum_steps > 1:
            sparse_apply_t0 = time.perf_counter()
            sparse_metrics = dynamic_vocab.apply_accumulated_gradients()
            sparse_apply_call_ms += (time.perf_counter() - sparse_apply_t0) * 1000.0
            sparse_step_ctx = None
        else:
            assert sparse_step_ctx is not None
            sparse_apply_t0 = time.perf_counter()
            sparse_metrics = dynamic_vocab.step(sparse_step_ctx)
            sparse_apply_call_ms += (time.perf_counter() - sparse_apply_t0) * 1000.0
            sparse_step_ctx = None
    if args.sparse_loss_topk_enable and step_correct_scores_all is not None and step_correct_records_all is not None and step_incorrect_scores_all is not None and step_incorrect_records_all is not None:
        step_correct_scores_cpu = step_correct_scores_all.detach().to(device="cpu")
        step_correct_records_cpu = step_correct_records_all.detach().to(device="cpu")
        step_incorrect_scores_cpu = step_incorrect_scores_all.detach().to(device="cpu")
        step_incorrect_records_cpu = step_incorrect_records_all.detach().to(device="cpu")
        bounded_correct_scores, bounded_correct_records = select_topk_records(
            step_correct_scores_cpu,
            step_correct_records_cpu,
            topk=args.sparse_loss_topk_correct,
        )
        bounded_incorrect_scores, bounded_incorrect_records = select_topk_records(
            step_incorrect_scores_cpu,
            step_incorrect_records_cpu,
            topk=args.sparse_loss_topk_incorrect,
        )
        analysis_cpu_payload = {
            "step": step,
            "correct_scores": bounded_correct_scores,
            "correct_records": bounded_correct_records,
            "incorrect_scores": bounded_incorrect_scores,
            "incorrect_records": bounded_incorrect_records,
        }
        if sparse_loss_analysis_writer is not None:
            sparse_loss_analysis_writer.submit(step, analysis_cpu_payload)
        if sparse_future_window_planner is not None:
            sparse_future_window_planner.update_from_step_payload(
                step,
                correct_scores=step_correct_scores_cpu,
                correct_records=step_correct_records_cpu,
                incorrect_scores=step_incorrect_scores_cpu,
                incorrect_records=step_incorrect_records_cpu,
            )
            sparse_future_window_planner.prune_consumed(step)
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
    assert train_loss_accum is not None
    train_loss_f = (train_loss_accum / grad_accum_steps).item() # .item() is a CPU-GPU sync point
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
        step_u = sparse_metrics.step_u_count if sparse_metrics.step_u_count > 0 else sparse_metrics.unique_count
        live_u = sparse_metrics.live_count if sparse_metrics.live_count > 0 else sparse_metrics.unique_count
        if sparse_metrics.u_capacity > 0 and sparse_metrics.u_capacity != step_u:
            sparse_str = f" | U_step: {step_u:,} | U_lm: {live_u:,}/{sparse_metrics.u_capacity:,} | stage: {sparse_metrics.stage_count:,}"
        elif sparse_metrics.unique_count != live_u:
            sparse_str = f" | U_live: {live_u:,} | U_union: {sparse_metrics.unique_count:,} | stage: {sparse_metrics.stage_count:,}"
        else:
            sparse_str = f" | U: {live_u:,} | stage: {sparse_metrics.stage_count:,}"
        warm_count = 0 if sparse_metrics.warm_ids_cpu is None else int(sparse_metrics.warm_ids_cpu.numel())
        cold_count = 0 if sparse_metrics.cold_ids_cpu is None else int(sparse_metrics.cold_ids_cpu.numel())
        if warm_count > 0 or cold_count > 0:
            sparse_str += (
                f" | warm: {warm_count:,}/{sparse_metrics.warm_budget_target:,}"
                f" | cold: {cold_count:,}/{sparse_metrics.cold_budget_target:,}"
            )
        if sparse_metrics.cold_bias_abs_max > 0.0 or sparse_metrics.cold_bias_clamped_count > 0:
            sparse_str += f" | cold_absmax: {sparse_metrics.cold_bias_abs_max:.2f}"
            if sparse_metrics.cold_bias_clamped_count > 0:
                sparse_str += f" | cold_clamped: {sparse_metrics.cold_bias_clamped_count:,}"
        if args.sparse_debug_timing and sparse_metrics.cloud_plan_ms > 0.0:
            sparse_str += (
                f" | cloud_ms plan: {sparse_metrics.cloud_plan_ms:.2f}"
                f" select: {sparse_metrics.cloud_selection_ms:.2f}"
                f" hidden: {sparse_metrics.cloud_hidden_query_ms:.2f}"
                f" | residual: {sparse_metrics.cloud_residual_capacity:,}"
                f" candidates: {sparse_metrics.warm_candidate_count:,}"
            )
        if args.sparse_debug_timing and grad_accum_steps == 1:
            sparse_str += (
                f" | step_ms prep: {sparse_prepare_ms:.2f}"
                f" fwdbwd: {sparse_fwdbwd_ms:.2f}"
                f" apply_call: {sparse_apply_call_ms:.2f}"
            )
        if args.sparse_debug_timing and grad_accum_steps > 1:
            sparse_str += (
                f" | step_ms prep: {sparse_prepare_ms:.2f}"
                f" fwdbwd: {sparse_fwdbwd_ms:.2f}"
                f" (clone: {sparse_input_clone_ms:.2f}"
                f" fwd: {sparse_forward_call_ms:.2f}"
                f" bwd: {sparse_backward_call_ms:.2f}"
                f" fetch: {sparse_next_fetch_ms:.2f})"
                f" bw_sync: {sparse_backward_sync_ms:.2f}"
                f" accum_start: {sparse_accum_start_ms:.2f}"
                f" accum_flush: {sparse_accum_flush_ms:.2f}"
                f" accum_queue: {sparse_accum_queue_ms:.2f}"
                f" accum: {sparse_accum_call_ms:.2f}"
                f" apply_call: {sparse_apply_call_ms:.2f}"
                f" | sparse_ms flush: {sparse_metrics.grad_accum_flush_ms:.2f}"
                f" stage: {sparse_metrics.grad_accum_stage_ms:.2f}"
                f" apply: {sparse_metrics.grad_accum_apply_ms:.2f}"
                f" restore: {sparse_metrics.grad_accum_restore_ms:.2f}"
                f" writeback: {sparse_metrics.grad_accum_writeback_ms:.2f}"
                f" (launch: {sparse_metrics.d2h_launch_ms:.2f}"
                f" sync: {sparse_metrics.d2h_sync_ms:.2f}"
                f" cpu: {sparse_metrics.cpu_writeback_ms:.2f})"
                f" | union_rows buffered: {sparse_metrics.grad_accum_queue_count:,}"
                f" resident: {sparse_metrics.grad_accum_resident_count:,}"
                f" | accum_rows queued: {sparse_accum_rows_queued:,}"
            )
            if sparse_prep_cpu_gather_ms > 0.0 or sparse_prep_h2d_enqueue_ms > 0.0 or sparse_prep_writeback_wait_ms > 0.0 or sparse_prep_prefetch_wait_ms > 0.0 or sparse_prep_prefetch_hits > 0:
                sparse_str += (
                    f" | prep_xfer_ms wait: {sparse_prep_writeback_wait_ms:.2f}"
                    f" prefetch_wait: {sparse_prep_prefetch_wait_ms:.2f}"
                    f" gather: {sparse_prep_cpu_gather_ms:.2f}"
                    f" h2d: {sparse_prep_h2d_enqueue_ms:.2f}"
                    f" | prefetch_hit: {sparse_prep_prefetch_hits:,}"
                    f" | h2d_tensors: {sparse_prep_h2d_tensor_count:,}"
                    f" h2d_mb: {sparse_prep_h2d_bytes / (1024 * 1024):.2f}"
                )
            if sparse_next_fetch_ms > 0.0:
                sparse_str += (
                    f" | fetch_ms loader: {sparse_loader_fetch_ms:.2f}"
                    f" plan: {sparse_plan_meta_ms:.2f}"
                    f" h2d: {sparse_batch_h2d_ms:.2f}"
                    f" prefetch_launch: {sparse_row_prefetch_launch_ms:.2f}"
                )
            if sparse_metrics.d2h_segment_count > 0 or sparse_metrics.d2h_bytes > 0:
                sparse_str += (
                    f" | d2h_segments: {sparse_metrics.d2h_segment_count:,}"
                    f" rows: {sparse_metrics.d2h_row_count:,}"
                    f" d2h_mb: {sparse_metrics.d2h_bytes / (1024 * 1024):.2f}"
                )
            if sparse_window_metrics is not None:
                sparse_str += (
                    f" | micro_ms start: {sparse_window_metrics.grad_accum_start_ms:.2f}"
                    f" flush: {sparse_window_metrics.grad_accum_flush_ms:.2f}"
                    f" queue: {sparse_window_metrics.grad_accum_queue_ms:.2f}"
                    f" | micro_rows queued: {sparse_window_metrics.grad_accum_queue_count:,}"
                    f" resident: {sparse_window_metrics.grad_accum_resident_count:,}"
                )
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
            warm_count = 0 if sparse_metrics.warm_ids_cpu is None else int(sparse_metrics.warm_ids_cpu.numel())
            cold_count = 0 if sparse_metrics.cold_ids_cpu is None else int(sparse_metrics.cold_ids_cpu.numel())
            log_data.update({
                "train/u": sparse_metrics.unique_count,
                "train/u_step": sparse_metrics.step_u_count if sparse_metrics.step_u_count > 0 else sparse_metrics.unique_count,
                "train/u_live": sparse_metrics.live_count if sparse_metrics.live_count > 0 else sparse_metrics.unique_count,
                "train/u_capacity": sparse_metrics.u_capacity,
                "train/u_stage": sparse_metrics.stage_count,
                "train/u_writeback": sparse_metrics.writeback_count,
                "train/u_warm": warm_count,
                "train/u_cold": cold_count,
                "train/u_residual": sparse_metrics.cloud_residual_capacity,
                "train/u_warm_target": sparse_metrics.warm_budget_target,
                "train/u_cold_target": sparse_metrics.cold_budget_target,
                "train/u_warm_candidates": sparse_metrics.warm_candidate_count,
                "train/cloud_plan_ms": sparse_metrics.cloud_plan_ms,
                "train/cloud_hidden_query_ms": sparse_metrics.cloud_hidden_query_ms,
                "train/cloud_selection_ms": sparse_metrics.cloud_selection_ms,
                "train/cold_bias_clamped": sparse_metrics.cold_bias_clamped_count,
                "train/cold_bias_absmax": sparse_metrics.cold_bias_abs_max,
                "train/step_prepare_ms": sparse_prepare_ms,
                "train/step_fwdbwd_ms": sparse_fwdbwd_ms,
                "train/step_apply_call_ms": sparse_apply_call_ms,
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
if sparse_loss_analysis_writer is not None:
    sparse_loss_analysis_writer.close()
wandb_run.finish() # wandb run finish
compute_cleanup()

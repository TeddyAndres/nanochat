"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist


def _get_valid_targets_and_num_bytes(targets, token_bytes):
    """Return safe targets, byte counts, and a mask of valid non-special tokens."""
    if (targets.int() < 0).any():  # mps does not currently have kernel for < 0 for int64, only int32
        valid = targets >= 0
        targets_safe = torch.where(valid, targets, torch.zeros_like(targets))
    else:
        valid = torch.ones_like(targets, dtype=torch.bool)
        targets_safe = targets
    num_bytes = torch.where(
        valid,
        token_bytes[targets_safe],
        torch.zeros_like(targets, dtype=token_bytes.dtype),
    )
    valid = valid & (num_bytes > 0)
    return targets_safe, num_bytes, valid


def _can_chunk_logits(model):
    return hasattr(model, "forward_features") and hasattr(model, "compute_logits")


def _accumulate_bpb_and_ece_from_logits(logits, targets, token_bytes, total_nats, total_bytes, bin_counts, bin_confidence_sums, bin_correct_sums):
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    targets_safe, num_bytes, valid = _get_valid_targets_and_num_bytes(flat_targets, token_bytes)
    if valid.any():
        flat_logits = flat_logits[valid]
        targets_safe = targets_safe[valid]
        num_bytes = num_bytes[valid]
        log_denom = torch.logsumexp(flat_logits, dim=-1)
        target_logits = flat_logits.gather(1, targets_safe.unsqueeze(1)).squeeze(1)
        total_nats += (log_denom - target_logits).sum()
        total_bytes += num_bytes.sum()

        max_logits, predictions = flat_logits.max(dim=-1)
        confidences = torch.exp(max_logits - log_denom)
        correctness = (predictions == targets_safe).to(torch.float64)
        num_bins = bin_counts.numel()
        bin_indices = torch.clamp((confidences * num_bins).to(torch.long), max=num_bins - 1)
        bin_counts += torch.bincount(bin_indices, minlength=num_bins).to(torch.float64)
        bin_confidence_sums += torch.bincount(bin_indices, weights=confidences.to(torch.float64), minlength=num_bins)
        bin_correct_sums += torch.bincount(bin_indices, weights=correctness, minlength=num_bins)


def _accumulate_bpb_from_logits(logits, targets, token_bytes, total_nats, total_bytes):
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    targets_safe, num_bytes, valid = _get_valid_targets_and_num_bytes(flat_targets, token_bytes)
    if valid.any():
        flat_logits = flat_logits[valid]
        targets_safe = targets_safe[valid]
        num_bytes = num_bytes[valid]
        log_denom = torch.logsumexp(flat_logits, dim=-1)
        target_logits = flat_logits.gather(1, targets_safe.unsqueeze(1)).squeeze(1)
        total_nats += (log_denom - target_logits).sum()
        total_bytes += num_bytes.sum()


@torch.no_grad()
def evaluate_bpb_and_ece(model, batches, steps, token_bytes, token_chunk_size=64, num_bins=15):
    """
    Evaluate validation bits-per-byte and token-level expected calibration error.

    ECE is computed over next-token predictions using the same token masking semantics
    as BPB: ignore_index targets and zero-byte special tokens are excluded.
    """
    device = model.get_device()
    total_nats = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    bin_counts = torch.zeros(num_bins, dtype=torch.float64, device=device)
    bin_confidence_sums = torch.zeros(num_bins, dtype=torch.float64, device=device)
    bin_correct_sums = torch.zeros(num_bins, dtype=torch.float64, device=device)

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        if _can_chunk_logits(model):
            features = model.forward_features(x)
            for start in range(0, x.size(1), token_chunk_size):
                end = min(start + token_chunk_size, x.size(1))
                logits = model.compute_logits(features[:, start:end])
                _accumulate_bpb_and_ece_from_logits(
                    logits,
                    y[:, start:end],
                    token_bytes,
                    total_nats,
                    total_bytes,
                    bin_counts,
                    bin_confidence_sums,
                    bin_correct_sums,
                )
                del logits
            del features
        else:
            logits = model(x)
            _accumulate_bpb_and_ece_from_logits(
                logits,
                y,
                token_bytes,
                total_nats,
                total_bytes,
                bin_counts,
                bin_confidence_sums,
                bin_correct_sums,
            )
            del logits
        del x, y

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(bin_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(bin_confidence_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(bin_correct_sums, op=dist.ReduceOp.SUM)

    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf'), float('nan')

    bpb = total_nats / (math.log(2) * total_bytes)

    total_count = bin_counts.sum().item()
    if total_count == 0:
        ece = float('nan')
    else:
        nonzero = bin_counts > 0
        avg_confidence = torch.zeros_like(bin_confidence_sums)
        avg_accuracy = torch.zeros_like(bin_correct_sums)
        avg_confidence[nonzero] = bin_confidence_sums[nonzero] / bin_counts[nonzero]
        avg_accuracy[nonzero] = bin_correct_sums[nonzero] / bin_counts[nonzero]
        ece = ((avg_accuracy[nonzero] - avg_confidence[nonzero]).abs() * (bin_counts[nonzero] / total_count)).sum().item()

    return bpb, ece

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        if _can_chunk_logits(model):
            features = model.forward_features(x)
            for start in range(0, x.size(1), 64):
                end = min(start + 64, x.size(1))
                logits = model.compute_logits(features[:, start:end])
                _accumulate_bpb_from_logits(logits, y[:, start:end], token_bytes, total_nats, total_bytes)
                del logits
            del features
        else:
            loss2d = model(x, y, loss_reduction='none') # (B, T)
            loss2d = loss2d.view(-1) # flatten
            y = y.view(-1) # flatten
            _, num_bytes2d, valid = _get_valid_targets_and_num_bytes(y, token_bytes)
            total_nats += (loss2d * valid).sum()
            total_bytes += num_bytes2d.sum()
            del loss2d
        del x, y
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb

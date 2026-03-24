"""
Tests for validation loss / calibration helpers.

Run:
python -m pytest tests/test_loss_eval.py -v
"""

import math

import torch

from nanochat.core_eval import forward_model
from nanochat.loss_eval import evaluate_bpb, evaluate_bpb_and_ece


class DummyEvalModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("_logits", logits)

    def get_device(self):
        return self._logits.device

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', logit_scale=1.0, logit_bias=None):
        logits = self._logits[:idx.size(0), :idx.size(1)]
        if logit_bias is not None:
            logits = logits + logit_bias.view(1, 1, -1)
        if logit_scale != 1.0:
            logits = logits * logit_scale
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        return loss


class ChunkedEvalModel(DummyEvalModel):
    def forward_features(self, idx, kv_cache=None):
        return self._logits[:idx.size(0), :idx.size(1)]

    def compute_logits(self, x, logit_scale=1.0, logit_bias=None):
        if logit_bias is not None:
            x = x + logit_bias.view(1, 1, -1)
        if logit_scale != 1.0:
            return x * logit_scale
        return x

    def iter_logits(self, x, active_vocab=None, logit_scale=1.0, logit_bias=None, force_float=True, vocab_chunk_size=None):
        total_vocab = x.size(-1)
        chunk_size = total_vocab if vocab_chunk_size is None or vocab_chunk_size <= 0 else vocab_chunk_size
        for start in range(0, total_vocab, chunk_size):
            end = min(start + chunk_size, total_vocab)
            logits = x[..., start:end]
            if logit_bias is not None:
                logits = logits + logit_bias[start:end].view(1, -1)
            if logit_scale != 1.0:
                logits = logits * logit_scale
            yield start, end, logits.float() if force_float else logits


def test_evaluate_bpb_and_ece_perfectly_calibrated_bucket():
    conf = 0.8
    other = 0.2
    margin = math.log(conf / other)
    logits = torch.tensor([
        [[margin, 0.0], [margin, 0.0], [margin, 0.0], [margin, 0.0], [margin, 0.0]],
    ], dtype=torch.float32)
    targets = torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.long)
    inputs = torch.zeros_like(targets)
    model = DummyEvalModel(logits)
    token_bytes = torch.tensor([1, 1], dtype=torch.int64)

    bpb, ece = evaluate_bpb_and_ece(model, [(inputs, targets)], steps=1, token_bytes=token_bytes, num_bins=10, token_chunk_size=2)

    expected_nats = 4 * (-math.log(conf)) + (-math.log(other))
    expected_bpb = expected_nats / (math.log(2) * 5)
    assert math.isclose(bpb, expected_bpb, rel_tol=1e-6)
    assert ece < 1e-6


def test_evaluate_bpb_and_ece_excludes_ignored_and_special_tokens():
    logits = torch.tensor([
        [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]],
    ], dtype=torch.float32)
    targets = torch.tensor([[0, -1, 1, 1]], dtype=torch.long)
    inputs = torch.zeros_like(targets)
    model = DummyEvalModel(logits)
    token_bytes = torch.tensor([1, 0], dtype=torch.int64)

    bpb, ece = evaluate_bpb_and_ece(model, [(inputs, targets)], steps=1, token_bytes=token_bytes, num_bins=10, token_chunk_size=4)
    legacy_bpb = evaluate_bpb(model, [(inputs, targets)], steps=1, token_bytes=token_bytes)

    valid_conf = math.exp(2.0) / (math.exp(2.0) + 1.0)
    expected_bpb = (-math.log(valid_conf)) / math.log(2)
    assert math.isclose(bpb, expected_bpb, rel_tol=1e-6)
    assert math.isclose(legacy_bpb, expected_bpb, rel_tol=1e-6)
    assert math.isclose(ece, 1.0 - valid_conf, rel_tol=1e-6)


def test_chunked_eval_path_matches_fallback_path():
    torch.manual_seed(0)
    logits = torch.randn(2, 7, 5, dtype=torch.float32)
    targets = torch.tensor([
        [0, 1, 2, 3, 4, -1, 1],
        [4, 3, 2, 1, 0, 2, 3],
    ], dtype=torch.long)
    inputs = torch.zeros_like(targets)
    token_bytes = torch.tensor([1, 1, 1, 0, 1], dtype=torch.int64)

    fallback_model = DummyEvalModel(logits)
    chunked_model = ChunkedEvalModel(logits)

    fallback_bpb, fallback_ece = evaluate_bpb_and_ece(
        fallback_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
        token_chunk_size=3,
        num_bins=20,
    )
    chunked_bpb, chunked_ece = evaluate_bpb_and_ece(
        chunked_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
        token_chunk_size=3,
        num_bins=20,
    )
    fallback_bpb_only = evaluate_bpb(fallback_model, [(inputs, targets)], steps=1, token_bytes=token_bytes)
    chunked_bpb_only = evaluate_bpb(chunked_model, [(inputs, targets)], steps=1, token_bytes=token_bytes)

    assert math.isclose(chunked_bpb, fallback_bpb, rel_tol=1e-6)
    assert math.isclose(chunked_ece, fallback_ece, rel_tol=1e-6)
    assert math.isclose(chunked_bpb_only, fallback_bpb_only, rel_tol=1e-6)


def test_logit_scale_matches_manual_scaled_logits_in_eval_paths():
    logits = torch.tensor([
        [[4.0, 1.0, -2.0], [0.5, 2.0, -1.0], [3.0, 0.0, -1.0]],
        [[1.0, 3.0, -0.5], [2.5, 0.5, -2.0], [0.0, 1.5, 2.0]],
    ], dtype=torch.float32)
    targets = torch.tensor([
        [0, 1, 0],
        [1, 0, 2],
    ], dtype=torch.long)
    inputs = torch.zeros_like(targets)
    token_bytes = torch.tensor([1, 1, 1], dtype=torch.int64)
    scale = 0.5

    fallback_model = DummyEvalModel(logits)
    fallback_scaled_model = DummyEvalModel(logits * scale)
    chunked_model = ChunkedEvalModel(logits)
    chunked_scaled_model = ChunkedEvalModel(logits * scale)

    actual_bpb, actual_ece = evaluate_bpb_and_ece(
        fallback_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
        logit_scale=scale,
    )
    expected_bpb, expected_ece = evaluate_bpb_and_ece(
        fallback_scaled_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
    )
    actual_bpb_only = evaluate_bpb(
        fallback_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
        logit_scale=scale,
    )
    expected_bpb_only = evaluate_bpb(
        fallback_scaled_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
    )
    chunked_actual_bpb, chunked_actual_ece = evaluate_bpb_and_ece(
        chunked_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
        logit_scale=scale,
    )
    chunked_expected_bpb, chunked_expected_ece = evaluate_bpb_and_ece(
        chunked_scaled_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
    )
    chunked_actual_bpb_only = evaluate_bpb(
        chunked_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
        logit_scale=scale,
    )
    chunked_expected_bpb_only = evaluate_bpb(
        chunked_scaled_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
    )

    assert math.isclose(actual_bpb, expected_bpb, rel_tol=1e-6)
    assert math.isclose(actual_ece, expected_ece, rel_tol=1e-6)
    assert math.isclose(actual_bpb_only, expected_bpb_only, rel_tol=1e-6)
    assert math.isclose(chunked_actual_bpb, chunked_expected_bpb, rel_tol=1e-6)
    assert math.isclose(chunked_actual_ece, chunked_expected_ece, rel_tol=1e-6)
    assert math.isclose(chunked_actual_bpb_only, chunked_expected_bpb_only, rel_tol=1e-6)


def test_logit_bias_matches_manual_shifted_logits_in_eval_paths():
    logits = torch.tensor([
        [[4.0, 1.0, -2.0], [0.5, 2.0, -1.0], [3.0, 0.0, -1.0]],
        [[1.0, 3.0, -0.5], [2.5, 0.5, -2.0], [0.0, 1.5, 2.0]],
    ], dtype=torch.float32)
    targets = torch.tensor([
        [0, 1, 0],
        [1, 0, 2],
    ], dtype=torch.long)
    inputs = torch.zeros_like(targets)
    token_bytes = torch.tensor([1, 1, 1], dtype=torch.int64)
    bias = torch.tensor([-0.2, 0.1, -0.4], dtype=torch.float32)

    fallback_model = DummyEvalModel(logits)
    fallback_biased_model = DummyEvalModel(logits + bias.view(1, 1, -1))
    chunked_model = ChunkedEvalModel(logits)
    chunked_biased_model = ChunkedEvalModel(logits + bias.view(1, 1, -1))

    actual_bpb, actual_ece = evaluate_bpb_and_ece(
        fallback_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
        logit_bias=bias,
    )
    expected_bpb, expected_ece = evaluate_bpb_and_ece(
        fallback_biased_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
    )
    actual_bpb_only = evaluate_bpb(
        fallback_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
        logit_bias=bias,
    )
    expected_bpb_only = evaluate_bpb(
        fallback_biased_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
    )
    chunked_actual_bpb, chunked_actual_ece = evaluate_bpb_and_ece(
        chunked_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
        logit_bias=bias,
    )
    chunked_expected_bpb, chunked_expected_ece = evaluate_bpb_and_ece(
        chunked_biased_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
    )
    chunked_actual_bpb_only = evaluate_bpb(
        chunked_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
        logit_bias=bias,
    )
    chunked_expected_bpb_only = evaluate_bpb(
        chunked_biased_model,
        [(inputs, targets)],
        steps=1,
        token_bytes=token_bytes,
    )

    assert math.isclose(actual_bpb, expected_bpb, rel_tol=1e-6)
    assert math.isclose(actual_ece, expected_ece, rel_tol=1e-6)
    assert math.isclose(actual_bpb_only, expected_bpb_only, rel_tol=1e-6)
    assert math.isclose(chunked_actual_bpb, chunked_expected_bpb, rel_tol=1e-6)
    assert math.isclose(chunked_actual_ece, chunked_expected_ece, rel_tol=1e-6)
    assert math.isclose(chunked_actual_bpb_only, chunked_expected_bpb_only, rel_tol=1e-6)


def test_core_forward_model_streaming_matches_dense_path():
    torch.manual_seed(0)
    logits = torch.randn(3, 6, 7, dtype=torch.float32)
    input_ids = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 0],
        [1, 3, 5, 0, 2, 4],
    ], dtype=torch.long)

    dense_model = DummyEvalModel(logits)
    chunked_model = ChunkedEvalModel(logits)

    dense_losses, dense_predictions = forward_model(dense_model, input_ids)
    chunked_losses, chunked_predictions = forward_model(chunked_model, input_ids)

    assert torch.allclose(chunked_predictions, dense_predictions)
    assert torch.allclose(chunked_losses[:, :-1], dense_losses[:, :-1], atol=1e-6, rtol=1e-6)
    assert torch.isnan(chunked_losses[:, -1]).all()
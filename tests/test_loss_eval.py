"""
Tests for validation loss / calibration helpers.

Run:
python -m pytest tests/test_loss_eval.py -v
"""

import math

import torch

from nanochat.loss_eval import evaluate_bpb, evaluate_bpb_and_ece


class DummyEvalModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("_logits", logits)

    def get_device(self):
        return self._logits.device

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        logits = self._logits[:idx.size(0), :idx.size(1)]
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

    def compute_logits(self, x):
        return x


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
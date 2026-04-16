"""
Tests for first-pass dynamic vocab runtime.

Run:
python -m pytest tests/test_dynamic_vocab.py -v
"""

import torch
import torch.nn as nn
import pytest
from concurrent.futures import Future

from nanochat.dynamic_vocab import COLD_LOGIT_BIAS_CLAMP_MAX, DynamicVocabRuntime, round_capacity_up
from nanochat.gpt import GPT, GPTConfig


def build_tiny_model(vocab_size=8):
    config = GPTConfig(
        sequence_len=4,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=1,
        n_kv_head=1,
        n_embd=32,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    return model


def build_fixed_step_meta(
    slot_to_global,
    stage_slots,
    stage_ids,
    writeback_slots,
    writeback_ids,
    warm_ids=None,
    cold_ids=None,
    inputs_cpu_local=None,
    inputs_union_cpu_local=None,
    targets_cpu_local=None,
    is_last_step=False,
    grad_accum_ids=None,
    grad_accum_steps=1,
    grad_accum_micro_step=0,
    is_grad_accum_boundary=True,
):
    slot_to_global = torch.tensor(slot_to_global, dtype=torch.long)
    active_mask = slot_to_global >= 0
    active_slot_ids = torch.nonzero(active_mask, as_tuple=False).flatten()
    active_ids = slot_to_global[active_slot_ids]
    if grad_accum_ids is None:
        grad_accum_ids = active_ids.tolist()
    step_meta = {
        "mode": "fixed-u",
        "active_ids_cpu": active_ids,
        "active_slot_ids_cpu": active_slot_ids,
        "active_mask_cpu": active_mask,
        "slot_to_global_cpu": slot_to_global,
        "stage_ids_cpu": torch.tensor(stage_ids, dtype=torch.long),
        "stage_slot_ids_cpu": torch.tensor(stage_slots, dtype=torch.long),
        "writeback_ids_cpu": torch.tensor(writeback_ids, dtype=torch.long),
        "writeback_slot_ids_cpu": torch.tensor(writeback_slots, dtype=torch.long),
        "grad_accum_ids_cpu": torch.tensor(grad_accum_ids, dtype=torch.long),
        "grad_accum_steps": grad_accum_steps,
        "grad_accum_micro_step": grad_accum_micro_step,
        "is_grad_accum_boundary": is_grad_accum_boundary,
        "is_last_step": is_last_step,
    }
    if warm_ids is not None:
        step_meta["warm_ids_cpu"] = torch.tensor(warm_ids, dtype=torch.long)
    if cold_ids is not None:
        step_meta["cold_ids_cpu"] = torch.tensor(cold_ids, dtype=torch.long)
    if inputs_cpu_local is not None:
        step_meta["inputs_cpu_local"] = torch.tensor(inputs_cpu_local, dtype=torch.long)
    if inputs_union_cpu_local is not None:
        step_meta["inputs_union_cpu_local"] = torch.tensor(inputs_union_cpu_local, dtype=torch.long)
    if targets_cpu_local is not None:
        step_meta["targets_cpu_local"] = torch.tensor(targets_cpu_local, dtype=torch.long)
    return step_meta


def set_zero_sparse_grads(step_ctx):
    step_ctx.active_vocab["wte"].grad = torch.zeros_like(step_ctx.active_vocab["wte"])
    step_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step_ctx.active_vocab["lm_head"])
    for value_embed in step_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)


def test_dynamic_vocab_forward_matches_dense_when_u_equals_v():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(model, device="cpu", embedding_lr=0.01, value_embedding_lr=0.01, unembedding_lr=0.01)
    active_ids = torch.arange(model.config.vocab_size, dtype=torch.long)
    step_ctx = runtime.prepare_step(active_ids)
    idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    dense_logits = model(idx)
    sparse_logits = model(idx, active_vocab=step_ctx.active_vocab)
    dense_loss = model(idx, targets)
    sparse_loss = model(idx, targets, active_vocab=step_ctx.active_vocab)

    assert torch.allclose(sparse_logits, dense_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sparse_loss, dense_loss, atol=1e-5, rtol=1e-5)


def test_dynamic_vocab_runtime_updates_only_active_rows():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(model, device="cpu", embedding_lr=0.05, value_embedding_lr=0.04, unembedding_lr=0.03)
    active_ids = torch.tensor([1, 3, 7], dtype=torch.long)
    inactive_id = 0
    wte = model.transformer["wte"]
    assert isinstance(wte, nn.Embedding)
    original_wte_inactive = wte.weight[inactive_id].clone()
    original_wte_active = wte.weight[active_ids].clone()

    step_ctx = runtime.prepare_step(active_ids)
    assert step_ctx.active_vocab is not None
    step_ctx.active_vocab["wte"].grad = torch.ones_like(step_ctx.active_vocab["wte"])
    step_ctx.active_vocab["lm_head"].grad = 2 * torch.ones_like(step_ctx.active_vocab["lm_head"])
    for value_embed in step_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = 3 * torch.ones_like(value_embed)

    runtime.step(step_ctx)

    assert step_ctx.active_vocab is None
    assert step_ctx.optimizer_state is None
    assert not torch.allclose(wte.weight[active_ids], original_wte_active)
    assert torch.allclose(wte.weight[inactive_id], original_wte_inactive)


def test_fixed_overlap_reuse_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("NANOCHAT_DISABLE_FIXED_OVERLAP_REUSE", raising=False)
    runtime = DynamicVocabRuntime(build_tiny_model(), device="cpu", embedding_lr=0.01, value_embedding_lr=0.01, unembedding_lr=0.01)
    assert runtime.disable_fixed_overlap_reuse is False


def test_fixed_overlap_reuse_can_be_disabled_via_env(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DISABLE_FIXED_OVERLAP_REUSE", "1")
    runtime = DynamicVocabRuntime(build_tiny_model(), device="cpu", embedding_lr=0.01, value_embedding_lr=0.01, unembedding_lr=0.01)
    assert runtime.disable_fixed_overlap_reuse is True


def test_dynamic_vocab_runtime_state_dict_round_trip():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(model, device="cpu", embedding_lr=0.05, value_embedding_lr=0.04, unembedding_lr=0.03)
    active_ids = torch.tensor([1, 3, 7], dtype=torch.long)
    step_ctx = runtime.prepare_step(active_ids)
    assert step_ctx.active_vocab is not None
    step_ctx.active_vocab["wte"].grad = torch.ones_like(step_ctx.active_vocab["wte"])
    step_ctx.active_vocab["lm_head"].grad = torch.ones_like(step_ctx.active_vocab["lm_head"])
    for value_embed in step_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.ones_like(value_embed)
    runtime.step(step_ctx)

    saved_state = runtime.state_dict()
    restored = DynamicVocabRuntime(model, device="cpu", embedding_lr=0.05, value_embedding_lr=0.04, unembedding_lr=0.03)
    restored.load_state_dict(saved_state)

    for name, spec in runtime.table_specs.items():
        param = spec["param"]
        restored_param = restored.table_specs[name]["param"]
        runtime_state = runtime.state[param]
        restored_state = restored.state[restored_param]
        assert runtime_state["step"] == restored_state["step"]
        assert torch.allclose(runtime_state["exp_avg"], restored_state["exp_avg"])
        assert torch.allclose(runtime_state["exp_avg_sq"], restored_state["exp_avg_sq"])
    assert runtime.runtime_step == restored.runtime_step
    assert torch.equal(runtime.last_seen_step_cpu, restored.last_seen_step_cpu)
    assert torch.equal(runtime.token_event_step_count_cpu, restored.token_event_step_count_cpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dynamic_prepare_step_stages_lm_head_optimizer_state_without_aliasing():
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(
            sequence_len=8,
            vocab_size=2048,
            n_layer=2,
            n_head=1,
            n_kv_head=1,
            n_embd=256,
            window_pattern="L",
        )
    )
    model.init_weights()


    model = model.to("cuda")
    runtime = DynamicVocabRuntime(
        model,
        device="cuda",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
    )
    active_ids = torch.arange(1024, dtype=torch.long)
    idx = torch.randint(0, 1024, (4, 8), device="cuda")
    targets = torch.randint(0, 1024, (4, 8), device="cuda")

    step0 = runtime.prepare_step(active_ids)
    loss0 = model(idx, targets, active_vocab=step0.active_vocab)
    loss0.backward()
    runtime.step(step0)

    step1 = runtime.prepare_step(active_ids)
    lm_head = runtime.table_specs["lm_head"]["param"]
    expected_exp_avg = runtime.state[lm_head]["exp_avg"].index_select(0, active_ids)

    assert torch.allclose(
        step1.optimizer_state["lm_head"]["exp_avg"].detach().cpu(),
        expected_exp_avg,
        atol=0.0,
        rtol=0.0,
    )


def test_sparse_step_uses_token_local_event_count_instead_of_table_age():
    torch.manual_seed(0)
    model_a = build_tiny_model(vocab_size=10)
    model_b = build_tiny_model(vocab_size=10)
    model_b.load_state_dict(model_a.state_dict())

    runtime_a = DynamicVocabRuntime(model_a, device="cpu", embedding_lr=0.05, value_embedding_lr=0.04, unembedding_lr=0.03)
    runtime_b = DynamicVocabRuntime(model_b, device="cpu", embedding_lr=0.05, value_embedding_lr=0.04, unembedding_lr=0.03)

    token_id = torch.tensor([3], dtype=torch.long)
    for runtime in (runtime_a, runtime_b):
        runtime.token_event_step_count_cpu[token_id] = 1

    for name in runtime_a.table_specs:
        param_a = runtime_a.table_specs[name]["param"]
        param_b = runtime_b.table_specs[name]["param"]
        runtime_a.state[param_a]["step"] = 100
        runtime_b.state[param_b]["step"] = 0

    step_a = runtime_a.prepare_step(token_id)
    step_b = runtime_b.prepare_step(token_id)

    step_a.active_vocab["wte"].grad = torch.ones_like(step_a.active_vocab["wte"])
    step_b.active_vocab["wte"].grad = torch.ones_like(step_b.active_vocab["wte"])
    step_a.active_vocab["lm_head"].grad = 2 * torch.ones_like(step_a.active_vocab["lm_head"])
    step_b.active_vocab["lm_head"].grad = 2 * torch.ones_like(step_b.active_vocab["lm_head"])
    for value_embed_a, value_embed_b in zip(step_a.active_vocab["value_embeds"].values(), step_b.active_vocab["value_embeds"].values()):
        value_embed_a.grad = 3 * torch.ones_like(value_embed_a)
        value_embed_b.grad = 3 * torch.ones_like(value_embed_b)

    runtime_a.step(step_a)
    runtime_b.step(step_b)

    for name in runtime_a.table_specs:
        param_a = runtime_a.table_specs[name]["param"]
        param_b = runtime_b.table_specs[name]["param"]
        assert torch.allclose(param_a[token_id], param_b[token_id])
        assert torch.allclose(runtime_a.state[param_a]["exp_avg"][token_id], runtime_b.state[param_b]["exp_avg"][token_id])
        assert torch.allclose(runtime_a.state[param_a]["exp_avg_sq"][token_id], runtime_b.state[param_b]["exp_avg_sq"][token_id])

    assert runtime_a.token_event_step_count_cpu[token_id].item() == 2
    assert runtime_b.token_event_step_count_cpu[token_id].item() == 2


def test_dynamic_vocab_dense_materialization_round_trip():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(model, device="cpu", embedding_lr=0.05, value_embedding_lr=0.04, unembedding_lr=0.03)
    idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    wte_before = runtime.table_specs["wte"]["param"].data
    logits_before = None
    with runtime.materialize_dense_params() as dense_model:
        logits_before = dense_model(idx)
        assert runtime.table_specs["wte"]["param"].device.type == "cpu"
        assert logits_before.shape[-1] == model.config.vocab_size

    assert runtime.table_specs["wte"]["param"].data.device.type == "cpu"
    assert runtime.table_specs["wte"]["param"].data.data_ptr() == wte_before.data_ptr()


def test_round_capacity_up_and_runtime_alignment_are_opt_in():
    assert round_capacity_up(None, 128) is None
    assert round_capacity_up(0, 128) == 0
    assert round_capacity_up(129, 128) == 256
    assert round_capacity_up(256, 128) == 256

    model = build_tiny_model(vocab_size=16)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=3,
        lm_head_u_max=5,
        grad_accum_u_max=4,
        capacity_round_multiple=8,
    )

    assert runtime.fixed_u_max == 8
    assert runtime.lm_head_u_max == 8
    assert runtime.grad_accum_u_max == 8

    reference_runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=3,
        lm_head_u_max=5,
        grad_accum_u_max=4,
    )

    assert reference_runtime.fixed_u_max == 3
    assert reference_runtime.lm_head_u_max == 5
    assert reference_runtime.grad_accum_u_max == 4


def test_fixed_u_runtime_delays_common_writeback_until_final_step():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=6,
    )
    wte = model.transformer["wte"]
    original_rows = wte.weight[[1, 3, 7]].clone()

    step0 = build_fixed_step_meta(
        slot_to_global=[1, 3, 7, -1, -1, -1],
        stage_slots=[0, 1, 2],
        stage_ids=[1, 3, 7],
        writeback_slots=[0],
        writeback_ids=[1],
        is_last_step=False,
    )
    step0_ctx = runtime.prepare_step(step0)
    active_slots0 = step0_ctx.active_slot_ids_cpu
    assert active_slots0 is not None
    step0_ctx.active_vocab["wte"].grad = torch.zeros_like(step0_ctx.active_vocab["wte"])
    step0_ctx.active_vocab["wte"].grad[active_slots0] = 1
    step0_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step0_ctx.active_vocab["lm_head"])
    step0_ctx.active_vocab["lm_head"].grad[active_slots0] = 2
    for value_embed in step0_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)
        value_embed.grad[active_slots0] = 3
    runtime.step(step0_ctx)

    assert not torch.allclose(wte.weight[1], original_rows[0])
    assert torch.allclose(wte.weight[3], original_rows[1])
    assert torch.allclose(wte.weight[7], original_rows[2])

    step1 = build_fixed_step_meta(
        slot_to_global=[-1, 3, 7, -1, -1, -1],
        stage_slots=[],
        stage_ids=[],
        writeback_slots=[],
        writeback_ids=[],
        is_last_step=True,
    )
    step1_ctx = runtime.prepare_step(step1)
    active_slots1 = step1_ctx.active_slot_ids_cpu
    assert active_slots1 is not None
    step1_ctx.active_vocab["wte"].grad = torch.zeros_like(step1_ctx.active_vocab["wte"])
    step1_ctx.active_vocab["wte"].grad[active_slots1] = 1
    step1_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step1_ctx.active_vocab["lm_head"])
    step1_ctx.active_vocab["lm_head"].grad[active_slots1] = 2
    for value_embed in step1_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)
        value_embed.grad[active_slots1] = 3
    runtime.step(step1_ctx)

    assert not torch.allclose(wte.weight[3], original_rows[1])
    assert not torch.allclose(wte.weight[7], original_rows[2])


def test_flush_pending_cpu_writeback_skips_disjoint_ids():
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=6,
    )

    class FakeFuture:
        def __init__(self):
            self.result_calls = 0

        def done(self):
            return False

        def result(self):
            self.result_calls += 1
            return None

    fake_future = FakeFuture()
    runtime._pending_cpu_writeback_future = fake_future
    pending_mask = torch.zeros(model.config.vocab_size, dtype=torch.bool)
    pending_mask[1] = True
    runtime._pending_cpu_writeback_mask_cpu = pending_mask

    runtime._flush_pending_cpu_writeback(torch.tensor([2, 3], dtype=torch.long))

    assert fake_future.result_calls == 0
    assert runtime._pending_cpu_writeback_future is fake_future
    assert runtime._pending_cpu_writeback_mask_cpu is pending_mask


def test_flush_pending_cpu_writeback_waits_for_overlapping_ids():
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=6,
    )

    class FakeFuture:
        def __init__(self):
            self.result_calls = 0

        def done(self):
            return False

        def result(self):
            self.result_calls += 1
            return None

    fake_future = FakeFuture()
    runtime._pending_cpu_writeback_future = fake_future
    pending_mask = torch.zeros(model.config.vocab_size, dtype=torch.bool)
    pending_mask[1] = True
    runtime._pending_cpu_writeback_mask_cpu = pending_mask

    runtime._flush_pending_cpu_writeback(torch.tensor([1, 3], dtype=torch.long))

    assert fake_future.result_calls == 1
    assert runtime._pending_cpu_writeback_future is None
    assert runtime._pending_cpu_writeback_mask_cpu is None


def test_cpu_receive_buffer_allocates_fresh_storage_while_writeback_is_pending():
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=6,
    )

    first_buffer = runtime._get_cpu_receive_buffer("rows:wte:fixed", (2, model.config.n_embd), torch.float32)

    class FakeFuture:
        def done(self):
            return False

        def result(self):
            return None

    runtime._pending_cpu_writeback_future = FakeFuture()
    runtime._pending_cpu_writeback_mask_cpu = torch.zeros(model.config.vocab_size, dtype=torch.bool)

    second_buffer = runtime._get_cpu_receive_buffer(
        "rows:wte:fixed",
        (2, model.config.n_embd),
        torch.float32,
        block_reuse_while_pending=True,
    )

    assert second_buffer.data_ptr() != first_buffer.data_ptr()


def test_fixed_u_masked_logits_hide_inactive_slots():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=6,
    )
    step_meta = build_fixed_step_meta(
        slot_to_global=[0, 1, 2, 3, -1, -1],
        stage_slots=[0, 1, 2, 3],
        stage_ids=[0, 1, 2, 3],
        writeback_slots=[0, 1, 2, 3],
        writeback_ids=[0, 1, 2, 3],
        is_last_step=True,
    )
    step_ctx = runtime.prepare_step(step_meta)
    idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)

    logits = model(idx, active_vocab=step_ctx.active_vocab)
    scaled_logits = model(idx, active_vocab=step_ctx.active_vocab, logit_scale=0.5)
    loss = model(idx, targets, active_vocab=step_ctx.active_vocab)

    assert torch.isfinite(loss)
    assert torch.all(logits[..., 4:] < -1e8)
    assert torch.all(scaled_logits[..., 4:] < -1e8)


def test_plan_next_lm_head_cloud_excludes_step_ids_and_full_warm_pool():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.1,
        fixed_u_max=3,
        lm_head_u_max=5,
    )
    with torch.no_grad():
        runtime.table_specs["wte"]["param"].zero_()
        runtime.table_specs["lm_head"]["param"].zero_()
        runtime.table_specs["wte"]["param"][0, 0] = 1.0
        runtime.table_specs["wte"]["param"][1, 1] = 1.0
        runtime.table_specs["lm_head"]["param"][5, 0] = 2.0
        runtime.table_specs["lm_head"]["param"][6, 1] = 1.5
    runtime.global_token_count_cpu[torch.tensor([4, 5, 6, 7])] = torch.tensor([9, 20, 18, 7], dtype=torch.long)

    step_meta = build_fixed_step_meta(
        slot_to_global=[0, 1, -1],
        stage_slots=[0, 1],
        stage_ids=[0, 1],
        writeback_slots=[0, 1],
        writeback_ids=[0, 1],
        inputs_cpu_local=[[0, 1, 0, 1]],
        is_last_step=True,
    )
    planned = runtime.plan_next_lm_head_cloud(
        step_meta,
        warm_proportion=0.5,
        router_candidate_pool_size=4,
        router_topk=2,
        source_token_limit=4,
    )

    warm_ids = planned["warm_ids_cpu"]
    cold_ids = planned["cold_ids_cpu"]
    warm_candidate_ids = planned["warm_candidate_ids_cpu"]
    assert warm_ids.numel() == 2
    assert cold_ids.numel() == 1
    assert 0 not in warm_ids.tolist() and 1 not in warm_ids.tolist()
    assert 0 not in cold_ids.tolist() and 1 not in cold_ids.tolist()
    assert all(token not in warm_candidate_ids.tolist() for token in cold_ids.tolist())


def test_select_warm_cloud_aggregates_per_position_router_scores():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.1,
        fixed_u_max=3,
        lm_head_u_max=6,
    )
    with torch.no_grad():
        runtime.table_specs["wte"]["param"].zero_()
        runtime.table_specs["lm_head"]["param"].zero_()
        runtime.table_specs["wte"]["param"][0, 0] = 1.0
        runtime.table_specs["wte"]["param"][1, 1] = 1.0
        runtime.table_specs["lm_head"]["param"][5, 0] = 1.0
        runtime.table_specs["lm_head"]["param"][6, 1] = 1.0
        runtime.table_specs["lm_head"]["param"][7, 2] = 1.0
    runtime.global_token_count_cpu[torch.tensor([4, 5, 6, 7])] = torch.tensor([5, 10, 9, 8], dtype=torch.long)

    warm_ids, ranked_ids = runtime.select_warm_cloud(
        torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        max_warm=2,
        topk_per_position=2,
        shortlist_limit=4,
    )

    assert warm_ids.tolist() == [5, 6]
    assert ranked_ids[:2].tolist() == [5, 6]


def test_get_subsampled_hidden_queries_returns_bounded_hidden_vectors():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.1,
        fixed_u_max=4,
        lm_head_u_max=6,
    )
    queries = runtime.get_subsampled_hidden_queries(
        torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        num_samples=2,
        max_prefix_len=3,
    )

    assert queries.shape == (2, model.config.n_embd)
    assert torch.isfinite(queries).all()


def test_get_subsampled_hidden_queries_matches_single_forward_causal_positions():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.1,
        fixed_u_max=4,
        lm_head_u_max=6,
    )
    preview_tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    positions = runtime._select_hidden_query_positions(preview_tokens.shape[1], "uniform", 2)
    queries = runtime.get_subsampled_hidden_queries(
        preview_tokens,
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        num_samples=2,
        max_prefix_len=4,
    )
    active_vocab = runtime._build_hidden_query_active_vocab(
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )
    with torch.inference_mode():
        hidden = model.forward_features(preview_tokens, active_vocab=active_vocab)
        expected = hidden.index_select(1, positions).mean(dim=0).to(dtype=torch.float32)

    assert torch.allclose(queries, expected, atol=1e-5, rtol=1e-5)


def test_select_warm_cloud_uses_hidden_queries_when_preview_active_vocab_is_available(monkeypatch):
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.1,
        fixed_u_max=4,
        lm_head_u_max=6,
    )
    with torch.no_grad():
        runtime.table_specs["lm_head"]["param"].zero_()
        runtime.table_specs["lm_head"]["param"][5, 0] = 1.0
        runtime.table_specs["lm_head"]["param"][6, 1] = 1.0
    runtime.global_token_count_cpu[torch.tensor([4, 5, 6, 7])] = torch.tensor([4, 10, 9, 8], dtype=torch.long)

    def fake_hidden_queries(*args, **kwargs):
        return torch.tensor([[1.0] + [0.0] * (model.config.n_embd - 1)], dtype=torch.float32)

    monkeypatch.setattr(runtime, "get_subsampled_hidden_queries", fake_hidden_queries)
    warm_ids, ranked_ids = runtime.select_warm_cloud(
        torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        max_warm=1,
        topk_per_position=2,
        shortlist_limit=4,
        max_positions=2,
        preview_active_ids_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        preview_active_slot_ids_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )

    assert warm_ids.tolist() == [5]
    assert ranked_ids[0].item() == 5


def test_select_warm_cloud_falls_back_to_preview_queries_when_hidden_queries_disabled(monkeypatch):
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.1,
        fixed_u_max=4,
        lm_head_u_max=6,
    )

    def hidden_queries_should_not_run(*args, **kwargs):
        raise AssertionError("hidden-query path should be disabled when max_positions == 0")

    monkeypatch.setattr(runtime, "get_subsampled_hidden_queries", hidden_queries_should_not_run)
    warm_ids, ranked_ids = runtime.select_warm_cloud(
        torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        max_warm=1,
        topk_per_position=2,
        shortlist_limit=4,
        max_positions=0,
        preview_active_ids_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        preview_active_slot_ids_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )

    assert warm_ids.numel() == 1
    assert ranked_ids.numel() >= 1


def test_plan_next_lm_head_cloud_fills_warm_budget_beyond_shortlist_limit():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=32)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.1,
        fixed_u_max=4,
        lm_head_u_max=20,
    )
    runtime.global_token_count_cpu.copy_(torch.arange(32, dtype=torch.long))

    step_meta = build_fixed_step_meta(
        slot_to_global=[0, 1, 2, 3],
        stage_slots=[0, 1, 2, 3],
        stage_ids=[0, 1, 2, 3],
        writeback_slots=[0, 1, 2, 3],
        writeback_ids=[0, 1, 2, 3],
        inputs_cpu_local=[[0, 1, 2, 3]],
        is_last_step=True,
    )
    planned = runtime.plan_next_lm_head_cloud(
        step_meta,
        warm_proportion=0.5,
        router_candidate_pool_size=3,
        router_topk=2,
        source_token_limit=0,
    )

    assert planned["warm_budget_target"] == 8
    assert planned["cold_budget_target"] == 8
    assert planned["warm_ids_cpu"].numel() == 8
    assert planned["cold_ids_cpu"].numel() == 8
    assert planned["warm_candidate_ids_cpu"].numel() >= 8


def test_plan_next_lm_head_cloud_reserves_hard_negative_budget_first():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=16)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.1,
        fixed_u_max=4,
        lm_head_u_max=8,
    )
    runtime.global_token_count_cpu.copy_(torch.arange(16, dtype=torch.long))

    step_meta = build_fixed_step_meta(
        slot_to_global=[0, 1, 2, 3],
        stage_slots=[0, 1, 2, 3],
        stage_ids=[0, 1, 2, 3],
        writeback_slots=[0, 1, 2, 3],
        writeback_ids=[0, 1, 2, 3],
        inputs_cpu_local=[[0, 1, 2, 3]],
        is_last_step=True,
    )
    planned = runtime.plan_next_lm_head_cloud(
        step_meta,
        warm_proportion=0.5,
        router_candidate_pool_size=0,
        router_topk=0,
        source_token_limit=0,
        hard_negative_ids_cpu=torch.tensor([7, 6], dtype=torch.long),
        hard_negative_budget=1,
    )

    assert planned["hard_negative_budget_target"] == 1
    assert planned["hard_negative_candidate_count"] == 2
    assert planned["hard_negative_ids_cpu"].tolist() == [7]
    assert 7 not in planned["warm_ids_cpu"].tolist()
    assert planned["cold_ids_cpu"][0].item() == 7
    assert planned["warm_ids_cpu"].numel() == 2
    assert planned["cold_ids_cpu"].numel() == 2


def test_fixed_u_clouds_expand_only_lm_head_capacity():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=4,
        lm_head_u_max=6,
    )
    step_meta = build_fixed_step_meta(
        slot_to_global=[0, 1, 2, -1],
        stage_slots=[0, 1, 2],
        stage_ids=[0, 1, 2],
        writeback_slots=[0, 1, 2],
        writeback_ids=[0, 1, 2],
        warm_ids=[4],
        cold_ids=[5],
        is_last_step=True,
    )
    step_ctx = runtime.prepare_step(step_meta)
    logits = model(torch.tensor([[0, 1, 2, 0]], dtype=torch.long), active_vocab=step_ctx.active_vocab)

    assert step_ctx.active_vocab is not None
    assert step_ctx.active_vocab["wte"].shape[0] == 4
    assert step_ctx.active_vocab["lm_head"].shape[0] == 6
    assert step_ctx.step_u_count == 3
    assert step_ctx.live_count == 5
    assert step_ctx.u_capacity == 6
    assert step_ctx.warm_slot_ids_cpu is not None
    assert step_ctx.warm_slot_ids_cpu[0].item() == 3
    assert torch.all(logits[..., 5:] < -1e8)
    assert torch.all(logits[..., 3:5] > -1e8)


def test_fixed_u_cloud_rows_use_separate_lm_head_learning_rates():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=1.0,
        unembedding_warm_lr=0.5,
        unembedding_cold_lr=0.25,
        fixed_u_max=3,
        lm_head_u_max=5,
        adam_betas=(0.0, 0.0),
    )
    original_rows = runtime.table_specs["lm_head"]["param"][[0, 1, 2, 4, 5]].clone()
    step_meta = build_fixed_step_meta(
        slot_to_global=[0, 1, 2],
        stage_slots=[0, 1, 2],
        stage_ids=[0, 1, 2],
        writeback_slots=[0, 1, 2],
        writeback_ids=[0, 1, 2],
        warm_ids=[4],
        cold_ids=[5],
        is_last_step=True,
    )
    step_ctx = runtime.prepare_step(step_meta)
    assert step_ctx.active_vocab is not None
    active_slots = step_ctx.active_slot_ids_cpu
    warm_slots = step_ctx.warm_slot_ids_cpu
    cold_slots = step_ctx.cold_slot_ids_cpu
    assert active_slots is not None and warm_slots is not None and cold_slots is not None
    step_ctx.active_vocab["wte"].grad = torch.zeros_like(step_ctx.active_vocab["wte"])
    step_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step_ctx.active_vocab["lm_head"])
    step_ctx.active_vocab["lm_head"].grad[active_slots] = 1.0
    step_ctx.active_vocab["lm_head"].grad[warm_slots] = 1.0
    step_ctx.active_vocab["lm_head"].grad[cold_slots] = 1.0
    for value_embed in step_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)
    runtime.step(step_ctx)

    updated_rows = runtime.table_specs["lm_head"]["param"][[0, 1, 2, 4, 5]]
    expected_deltas = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.25], dtype=updated_rows.dtype).unsqueeze(1)
    observed_deltas = original_rows - updated_rows
    assert torch.allclose(observed_deltas, expected_deltas.expand_as(observed_deltas), atol=1e-6)


def test_fixed_u_cloud_overlap_stays_resident_across_steps():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=3,
        lm_head_u_max=5,
    )

    step0 = build_fixed_step_meta(
        slot_to_global=[0, 1, 2],
        stage_slots=[0, 1, 2],
        stage_ids=[0, 1, 2],
        writeback_slots=[],
        writeback_ids=[],
        warm_ids=[4],
        cold_ids=[5],
        is_last_step=False,
    )
    step0_ctx = runtime.prepare_step(step0)
    warm_slots0 = step0_ctx.warm_slot_ids_cpu
    cold_slots0 = step0_ctx.cold_slot_ids_cpu
    assert warm_slots0 is not None and cold_slots0 is not None
    assert step0_ctx.stage_count == 5
    set_zero_sparse_grads(step0_ctx)
    runtime.step(step0_ctx)

    step1 = build_fixed_step_meta(
        slot_to_global=[0, 1, 2],
        stage_slots=[],
        stage_ids=[],
        writeback_slots=[],
        writeback_ids=[],
        warm_ids=[5],
        cold_ids=[4],
        is_last_step=False,
    )
    step1_ctx = runtime.prepare_step(step1)
    warm_slots1 = step1_ctx.warm_slot_ids_cpu
    cold_slots1 = step1_ctx.cold_slot_ids_cpu
    assert warm_slots1 is not None and cold_slots1 is not None

    assert step1_ctx.stage_count == 0
    assert step1_ctx.cloud_stage_ids_cpu is not None
    assert step1_ctx.cloud_stage_ids_cpu.numel() == 0
    assert step1_ctx.cloud_writeback_ids_cpu is not None
    assert step1_ctx.cloud_writeback_ids_cpu.numel() == 0
    assert warm_slots1[0].item() == cold_slots0[0].item()
    assert cold_slots1[0].item() == warm_slots0[0].item()


def test_fixed_u_cloud_writeback_is_deferred_until_cloud_leaves():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=1.0,
        unembedding_warm_lr=0.5,
        fixed_u_max=3,
        lm_head_u_max=4,
        adam_betas=(0.0, 0.0),
    )

    original_row4 = runtime.table_specs["lm_head"]["param"][4].clone()
    step0 = build_fixed_step_meta(
        slot_to_global=[0, 1, 2],
        stage_slots=[0, 1, 2],
        stage_ids=[0, 1, 2],
        writeback_slots=[],
        writeback_ids=[],
        warm_ids=[4],
        is_last_step=False,
    )
    step0_ctx = runtime.prepare_step(step0)
    warm_slots0 = step0_ctx.warm_slot_ids_cpu
    assert warm_slots0 is not None
    set_zero_sparse_grads(step0_ctx)
    step0_ctx.active_vocab["lm_head"].grad[warm_slots0] = 1.0
    runtime.step(step0_ctx)

    after_step_row4 = runtime.table_specs["lm_head"]["param"][4].clone()
    assert torch.allclose(after_step_row4, original_row4)

    step1 = build_fixed_step_meta(
        slot_to_global=[0, 1, 2],
        stage_slots=[],
        stage_ids=[],
        writeback_slots=[],
        writeback_ids=[],
        warm_ids=[5],
        is_last_step=False,
    )
    step1_ctx = runtime.prepare_step(step1)

    runtime._flush_pending_cpu_writeback()

    expected_row4 = original_row4 - 0.5
    assert torch.allclose(runtime.table_specs["lm_head"]["param"][4], expected_row4)
    assert step1_ctx.stage_count == 1
    assert step1_ctx.cloud_writeback_ids_cpu is not None
    assert step1_ctx.cloud_writeback_ids_cpu.tolist() == [4]
    assert step1_ctx.cloud_stage_ids_cpu is not None
    assert step1_ctx.cloud_stage_ids_cpu.tolist() == [5]


def test_plan_next_lm_head_cloud_retains_resident_cloud_ids_when_still_ranked():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=3,
        lm_head_u_max=5,
    )
    runtime.global_token_count_cpu.zero_()
    runtime.global_token_count_cpu[torch.tensor([4, 5, 6, 7])] = torch.tensor([95, 94, 100, 99], dtype=torch.long)

    first_step = build_fixed_step_meta(
        slot_to_global=[0, 1, 2],
        stage_slots=[0, 1, 2],
        stage_ids=[0, 1, 2],
        writeback_slots=[],
        writeback_ids=[],
        warm_ids=[4],
        cold_ids=[5],
        is_last_step=False,
    )
    first_ctx = runtime.prepare_step(first_step)
    set_zero_sparse_grads(first_ctx)
    runtime.step(first_ctx)

    next_step = build_fixed_step_meta(
        slot_to_global=[0, 1, 2],
        stage_slots=[],
        stage_ids=[],
        writeback_slots=[],
        writeback_ids=[],
        inputs_cpu_local=[[0, 1, 2, 0]],
        is_last_step=False,
    )
    planned = runtime.plan_next_lm_head_cloud(
        next_step,
        warm_proportion=0.0,
        router_candidate_pool_size=0,
        router_topk=0,
        source_token_limit=0,
    )

    assert planned["cold_ids_cpu"].tolist() == [4, 5]


def test_fixed_u_runtime_survives_inference_mode_materialization():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=6,
    )

    step_meta = build_fixed_step_meta(
        slot_to_global=[0, 1, 2, 3, -1, -1],
        stage_slots=[0, 1, 2, 3],
        stage_ids=[0, 1, 2, 3],
        writeback_slots=[0, 1, 2, 3],
        writeback_ids=[0, 1, 2, 3],
        is_last_step=False,
    )
    step_ctx = runtime.prepare_step(step_meta)
    active_slots = step_ctx.active_slot_ids_cpu
    assert active_slots is not None
    step_ctx.active_vocab["wte"].grad = torch.zeros_like(step_ctx.active_vocab["wte"])
    step_ctx.active_vocab["wte"].grad[active_slots] = 1
    step_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step_ctx.active_vocab["lm_head"])
    step_ctx.active_vocab["lm_head"].grad[active_slots] = 1
    for value_embed in step_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)
        value_embed.grad[active_slots] = 1
    runtime.step(step_ctx)

    with torch.inference_mode():
        with runtime.materialize_dense_params():
            _ = model(torch.tensor([[0, 1, 2, 3]], dtype=torch.long))

    next_ctx = runtime.prepare_step(step_meta)
    next_active_slots = next_ctx.active_slot_ids_cpu
    assert next_active_slots is not None
    next_ctx.active_vocab["wte"].grad = torch.zeros_like(next_ctx.active_vocab["wte"])
    next_ctx.active_vocab["wte"].grad[next_active_slots] = 1
    next_ctx.active_vocab["lm_head"].grad = torch.zeros_like(next_ctx.active_vocab["lm_head"])
    next_ctx.active_vocab["lm_head"].grad[next_active_slots] = 1
    for value_embed in next_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)
        value_embed.grad[next_active_slots] = 1
    runtime.step(next_ctx)


def test_sparse_logit_scale_changes_sparse_loss_for_same_targets():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
    )
    step_ctx = runtime.prepare_step(torch.arange(model.config.vocab_size, dtype=torch.long))
    idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    baseline_logits = model(idx, active_vocab=step_ctx.active_vocab)
    targets = baseline_logits.argmax(dim=-1)
    baseline_loss = model(idx, targets, active_vocab=step_ctx.active_vocab)
    scaled_loss = model(idx, targets, active_vocab=step_ctx.active_vocab, logit_scale=0.5)

    assert torch.isfinite(baseline_loss)
    assert torch.isfinite(scaled_loss)
    assert scaled_loss > baseline_loss


def test_sparse_cold_logit_bias_grows_with_absence_steps_and_batch_scale():
    torch.manual_seed(0)

    def build_runtime():
        return DynamicVocabRuntime(
            build_tiny_model(vocab_size=8),
            device="cpu",
            embedding_lr=0.01,
            value_embedding_lr=0.01,
            unembedding_lr=0.01,
            cold_bias_reference_tokens=8,
        )

    runtime_small = build_runtime()
    step0_small = runtime_small.prepare_step(torch.tensor([0], dtype=torch.long), cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    assert torch.allclose(step0_small.active_vocab["cold_logit_bias"], torch.zeros(1))
    set_zero_sparse_grads(step0_small)
    runtime_small.step(step0_small)

    step1_small = runtime_small.prepare_step(torch.tensor([1], dtype=torch.long), cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    set_zero_sparse_grads(step1_small)
    runtime_small.step(step1_small)

    revisit_small = runtime_small.prepare_step(torch.tensor([0], dtype=torch.long), cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    expected_small = torch.log1p(torch.tensor([1.0]))
    assert torch.allclose(revisit_small.active_vocab["cold_logit_bias"], expected_small, atol=1e-6)

    runtime_large = build_runtime()
    step0_large = runtime_large.prepare_step(torch.tensor([0], dtype=torch.long), cold_bias_scale=1.0, cold_bias_tokens_per_step=16)
    set_zero_sparse_grads(step0_large)
    runtime_large.step(step0_large)

    step1_large = runtime_large.prepare_step(torch.tensor([1], dtype=torch.long), cold_bias_scale=1.0, cold_bias_tokens_per_step=16)
    set_zero_sparse_grads(step1_large)
    runtime_large.step(step1_large)

    revisit_large = runtime_large.prepare_step(torch.tensor([0], dtype=torch.long), cold_bias_scale=1.0, cold_bias_tokens_per_step=16)
    expected_large = torch.log1p(torch.tensor([2.0]))
    assert torch.allclose(revisit_large.active_vocab["cold_logit_bias"], expected_large, atol=1e-6)
    assert revisit_large.active_vocab["cold_logit_bias"].item() > revisit_small.active_vocab["cold_logit_bias"].item()


def test_first_seen_rows_do_not_receive_cold_bias_even_late_in_training():
    runtime = DynamicVocabRuntime(
        build_tiny_model(vocab_size=8),
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        cold_bias_reference_tokens=8,
    )

    runtime.runtime_step = 400
    first_seen = runtime.prepare_step(torch.tensor([0, 3], dtype=torch.long), cold_bias_scale=512.0, cold_bias_tokens_per_step=8)
    assert torch.allclose(first_seen.active_vocab["cold_logit_bias"], torch.zeros(2))

    runtime.runtime_step = 400
    dense_bias = runtime.get_dense_cold_logit_bias(cold_bias_scale=512.0, cold_bias_tokens_per_step=8)
    assert dense_bias is not None
    assert torch.allclose(dense_bias, torch.zeros_like(dense_bias))


def test_fixed_u_cold_logit_bias_aligns_with_active_slots():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=6,
        cold_bias_reference_tokens=8,
    )

    step0 = build_fixed_step_meta(
        slot_to_global=[0, 2, -1, -1, -1, -1],
        stage_slots=[0, 1],
        stage_ids=[0, 2],
        writeback_slots=[0, 1],
        writeback_ids=[0, 2],
        is_last_step=True,
    )
    step0_ctx = runtime.prepare_step(step0, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    assert torch.allclose(step0_ctx.active_vocab["cold_logit_bias"][:2], torch.zeros(2))
    set_zero_sparse_grads(step0_ctx)
    runtime.step(step0_ctx)

    warm_other = build_fixed_step_meta(
        slot_to_global=[5, -1, -1, -1, -1, -1],
        stage_slots=[0],
        stage_ids=[5],
        writeback_slots=[0],
        writeback_ids=[5],
        is_last_step=True,
    )
    warm_other_ctx = runtime.prepare_step(warm_other, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    set_zero_sparse_grads(warm_other_ctx)
    runtime.step(warm_other_ctx)

    revisit = build_fixed_step_meta(
        slot_to_global=[-1, 0, -1, 2, -1, -1],
        stage_slots=[1, 3],
        stage_ids=[0, 2],
        writeback_slots=[1, 3],
        writeback_ids=[0, 2],
        is_last_step=True,
    )
    revisit_ctx = runtime.prepare_step(revisit, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    cold_bias = revisit_ctx.active_vocab["cold_logit_bias"]
    expected = torch.log1p(torch.tensor(1.0))
    assert torch.allclose(cold_bias[[1, 3]], expected.repeat(2), atol=1e-6)
    assert torch.allclose(cold_bias[[0, 2, 4, 5]], torch.zeros(4), atol=1e-6)


def test_fixed_u_grad_accum_cold_bias_is_window_stable_across_microsteps():
    runtime = DynamicVocabRuntime(
        build_tiny_model(vocab_size=8),
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=4,
        grad_accum_u_max=4,
        cold_bias_reference_tokens=8,
    )

    warm_both = build_fixed_step_meta(
        slot_to_global=[0, 2, -1, -1],
        stage_slots=[0, 1],
        stage_ids=[0, 2],
        writeback_slots=[0, 1],
        writeback_ids=[0, 2],
        is_last_step=True,
    )
    warm_both_ctx = runtime.prepare_step(warm_both, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    set_zero_sparse_grads(warm_both_ctx)
    runtime.step(warm_both_ctx)

    gap = build_fixed_step_meta(
        slot_to_global=[1, -1, -1, -1],
        stage_slots=[0],
        stage_ids=[1],
        writeback_slots=[0],
        writeback_ids=[1],
        is_last_step=True,
    )
    gap_ctx = runtime.prepare_step(gap, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    set_zero_sparse_grads(gap_ctx)
    runtime.step(gap_ctx)

    window_micro0 = build_fixed_step_meta(
        slot_to_global=[0, -1, -1, -1],
        stage_slots=[0],
        stage_ids=[0],
        writeback_slots=[],
        writeback_ids=[],
        grad_accum_ids=[0, 2],
        grad_accum_steps=2,
        grad_accum_micro_step=0,
        is_grad_accum_boundary=False,
        is_last_step=False,
    )
    window_ctx0 = runtime.prepare_step(window_micro0, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    expected = torch.log1p(torch.tensor(1.0))
    assert torch.allclose(window_ctx0.active_vocab["cold_logit_bias"][:2], torch.tensor([expected.item(), expected.item()]), atol=1e-6)
    set_zero_sparse_grads(window_ctx0)
    runtime.accumulate_gradients(window_ctx0)

    window_micro1 = build_fixed_step_meta(
        slot_to_global=[2, -1, -1, -1],
        stage_slots=[0],
        stage_ids=[2],
        writeback_slots=[0],
        writeback_ids=[2],
        grad_accum_ids=[0, 2],
        grad_accum_steps=2,
        grad_accum_micro_step=1,
        is_grad_accum_boundary=True,
        is_last_step=True,
    )
    window_ctx1 = runtime.prepare_step(window_micro1, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    assert torch.allclose(window_ctx1.active_vocab["cold_logit_bias"][:2], torch.tensor([expected.item(), expected.item()]), atol=1e-6)

def test_fixed_u_grad_accum_repeated_appearance_keeps_window_cold_bias_later_in_window():
    runtime = DynamicVocabRuntime(
        build_tiny_model(vocab_size=8),
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=4,
        grad_accum_u_max=4,
        cold_bias_reference_tokens=8,
    )

    warm = build_fixed_step_meta(
        slot_to_global=[0, -1, -1, -1],
        stage_slots=[0],
        stage_ids=[0],
        writeback_slots=[0],
        writeback_ids=[0],
        is_last_step=True,
    )
    warm_ctx = runtime.prepare_step(warm, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    set_zero_sparse_grads(warm_ctx)
    runtime.step(warm_ctx)

    gap = build_fixed_step_meta(
        slot_to_global=[3, -1, -1, -1],
        stage_slots=[0],
        stage_ids=[3],
        writeback_slots=[0],
        writeback_ids=[3],
        is_last_step=True,
    )
    gap_ctx = runtime.prepare_step(gap, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    set_zero_sparse_grads(gap_ctx)
    runtime.step(gap_ctx)

    window_micro0 = build_fixed_step_meta(
        slot_to_global=[0, -1, -1, -1],
        stage_slots=[0],
        stage_ids=[0],
        writeback_slots=[],
        writeback_ids=[],
        grad_accum_ids=[0],
        grad_accum_steps=2,
        grad_accum_micro_step=0,
        is_grad_accum_boundary=False,
        is_last_step=False,
    )
    window_ctx0 = runtime.prepare_step(window_micro0, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    expected = torch.log1p(torch.tensor(1.0))
    assert torch.allclose(window_ctx0.active_vocab["cold_logit_bias"][:1], torch.tensor([expected.item()]), atol=1e-6)
    set_zero_sparse_grads(window_ctx0)
    runtime.accumulate_gradients(window_ctx0)

    window_micro1 = build_fixed_step_meta(
        slot_to_global=[0, -1, -1, -1],
        stage_slots=[0],
        stage_ids=[0],
        writeback_slots=[0],
        writeback_ids=[0],
        grad_accum_ids=[0],
        grad_accum_steps=2,
        grad_accum_micro_step=1,
        is_grad_accum_boundary=True,
        is_last_step=True,
    )
    window_ctx1 = runtime.prepare_step(window_micro1, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    assert torch.allclose(window_ctx1.active_vocab["cold_logit_bias"][:1], torch.tensor([expected.item()]), atol=1e-6)


def test_fixed_u_grad_accum_apply_metrics_preserve_cold_bias_window_stats():
    runtime = DynamicVocabRuntime(
        build_tiny_model(vocab_size=8),
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=4,
        grad_accum_u_max=4,
        cold_bias_reference_tokens=8,
    )

    warm_both = build_fixed_step_meta(
        slot_to_global=[0, 2, -1, -1],
        stage_slots=[0, 1],
        stage_ids=[0, 2],
        writeback_slots=[0, 1],
        writeback_ids=[0, 2],
        is_last_step=True,
    )
    warm_both_ctx = runtime.prepare_step(warm_both, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    set_zero_sparse_grads(warm_both_ctx)
    runtime.step(warm_both_ctx)

    gap = build_fixed_step_meta(
        slot_to_global=[1, -1, -1, -1],
        stage_slots=[0],
        stage_ids=[1],
        writeback_slots=[0],
        writeback_ids=[1],
        is_last_step=True,
    )
    gap_ctx = runtime.prepare_step(gap, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    set_zero_sparse_grads(gap_ctx)
    runtime.step(gap_ctx)

    micro0 = build_fixed_step_meta(
        slot_to_global=[0, -1, -1, -1],
        stage_slots=[0],
        stage_ids=[0],
        writeback_slots=[],
        writeback_ids=[],
        grad_accum_ids=[0, 2],
        grad_accum_steps=2,
        grad_accum_micro_step=0,
        is_grad_accum_boundary=False,
        is_last_step=False,
    )
    micro0_ctx = runtime.prepare_step(micro0, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    set_zero_sparse_grads(micro0_ctx)
    runtime.accumulate_gradients(micro0_ctx)

    micro1 = build_fixed_step_meta(
        slot_to_global=[2, -1, -1, -1],
        stage_slots=[0],
        stage_ids=[2],
        writeback_slots=[0],
        writeback_ids=[2],
        grad_accum_ids=[0, 2],
        grad_accum_steps=2,
        grad_accum_micro_step=1,
        is_grad_accum_boundary=True,
        is_last_step=True,
    )
    micro1_ctx = runtime.prepare_step(micro1, cold_bias_scale=1.0, cold_bias_tokens_per_step=8)
    set_zero_sparse_grads(micro1_ctx)
    runtime.accumulate_gradients(micro1_ctx)

    lm_head_step_before_apply = runtime.state[runtime.table_specs["lm_head"]["param"]]["step"]
    metrics = runtime.apply_accumulated_gradients()
    lm_head_step_after_apply = runtime.state[runtime.table_specs["lm_head"]["param"]]["step"]

    assert lm_head_step_before_apply == 2
    assert lm_head_step_after_apply == 3
    expected = torch.log1p(torch.tensor(1.0))
    assert metrics.cold_bias_clamped_count == 0
    assert metrics.cold_bias_abs_max == expected.item()


def test_sparse_cold_logit_bias_clamps_for_extreme_absence():
    runtime = DynamicVocabRuntime(
        build_tiny_model(vocab_size=8),
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        cold_bias_reference_tokens=8,
    )

    cold_steps = torch.tensor([0.0, 2.0, 1_000.0], dtype=torch.float32)
    cold_bias = runtime._compute_cold_logit_bias_cpu(
        cold_steps,
        cold_bias_scale=512.0,
        cold_bias_tokens_per_step=8,
    )

    assert cold_bias[0].item() == 0.0
    assert cold_bias[1].item() == COLD_LOGIT_BIAS_CLAMP_MAX
    assert cold_bias[2].item() == COLD_LOGIT_BIAS_CLAMP_MAX

    runtime.runtime_step = 1003
    runtime.last_seen_step_cpu[:3] = torch.tensor([1002, 1000, 0], dtype=torch.long)
    step_ctx = runtime.prepare_step(
        torch.tensor([0, 1, 2], dtype=torch.long),
        cold_bias_scale=512.0,
        cold_bias_tokens_per_step=8,
    )
    assert step_ctx.cold_bias_clamped_count == 2
    assert step_ctx.cold_bias_abs_max == COLD_LOGIT_BIAS_CLAMP_MAX


def test_dense_cold_logit_bias_uses_same_clamp_bounds():
    runtime = DynamicVocabRuntime(
        build_tiny_model(vocab_size=8),
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        cold_bias_reference_tokens=8,
    )
    runtime.runtime_step = 1005
    runtime.last_seen_step_cpu.copy_(torch.tensor([1004, 1002, 0, 1004, 1004, 1004, 1004, 1004], dtype=torch.long))

    dense_bias = runtime.get_dense_cold_logit_bias(
        cold_bias_scale=512.0,
        cold_bias_tokens_per_step=8,
    )

    assert dense_bias is not None
    assert dense_bias[0].item() == 0.0
    assert dense_bias[1].item() == COLD_LOGIT_BIAS_CLAMP_MAX
    assert dense_bias[2].item() == COLD_LOGIT_BIAS_CLAMP_MAX


def test_cold_logit_bias_returns_zeros_for_non_positive_scale():
    runtime = DynamicVocabRuntime(
        build_tiny_model(vocab_size=8),
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        cold_bias_reference_tokens=8,
    )

    cold_steps = torch.tensor([0.0, 2.0, 1_000.0], dtype=torch.float32)
    cold_bias = runtime._compute_cold_logit_bias_cpu(
        cold_steps,
        cold_bias_scale=0.0,
        cold_bias_tokens_per_step=8,
    )

    assert torch.allclose(cold_bias, torch.zeros_like(cold_steps))


def test_fixed_u_cold_row_decay_updates_cpu_complement_after_staging():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=4,
    )

    base_lm_head = runtime.table_specs["lm_head"]["param"].clone()
    step_meta = build_fixed_step_meta(
        slot_to_global=[0, 2, -1, -1],
        stage_slots=[0, 1],
        stage_ids=[0, 2],
        writeback_slots=[0, 1],
        writeback_ids=[0, 2],
        is_last_step=True,
    )

    step_ctx = runtime.prepare_step(step_meta, cold_row_decay=0.25)

    assert torch.allclose(step_ctx.active_vocab["lm_head"][0], base_lm_head[0])
    assert torch.allclose(step_ctx.active_vocab["lm_head"][1], base_lm_head[2])

    cpu_lm_head = runtime.table_specs["lm_head"]["param"]
    assert torch.allclose(cpu_lm_head[0], base_lm_head[0])
    assert torch.allclose(cpu_lm_head[2], base_lm_head[2])
    assert torch.allclose(cpu_lm_head[1], base_lm_head[1] * 0.75)
    assert torch.allclose(cpu_lm_head[7], base_lm_head[7] * 0.75)


def test_fixed_u_cold_row_decay_accumulates_across_steps_for_absent_rows():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=2,
    )

    base_lm_head = runtime.table_specs["lm_head"]["param"].clone()
    step_meta = build_fixed_step_meta(
        slot_to_global=[0, -1],
        stage_slots=[0],
        stage_ids=[0],
        writeback_slots=[0],
        writeback_ids=[0],
        is_last_step=True,
    )

    step0_ctx = runtime.prepare_step(step_meta, cold_row_decay=0.5)
    set_zero_sparse_grads(step0_ctx)
    runtime.step(step0_ctx)

    step1_ctx = runtime.prepare_step(step_meta, cold_row_decay=0.5)

    cpu_lm_head = runtime.table_specs["lm_head"]["param"]
    assert torch.allclose(cpu_lm_head[0], base_lm_head[0])
    assert torch.allclose(cpu_lm_head[1], base_lm_head[1] * 0.25)
    assert torch.allclose(step1_ctx.active_vocab["lm_head"][0], base_lm_head[0])


def test_fixed_u_cold_row_decay_runs_once_per_grad_accum_union_stage():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=8)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=2,
        grad_accum_u_max=2,
    )

    base_lm_head = runtime.table_specs["lm_head"]["param"].clone()
    union_ids = [0, 2]
    micro0 = build_fixed_step_meta(
        slot_to_global=[0, -1],
        stage_slots=[0],
        stage_ids=[0],
        writeback_slots=[],
        writeback_ids=[],
        grad_accum_ids=union_ids,
        grad_accum_steps=2,
        grad_accum_micro_step=0,
        is_grad_accum_boundary=False,
        is_last_step=False,
    )
    micro0_ctx = runtime.prepare_step(micro0, cold_row_decay=0.25)
    assert torch.allclose(runtime.table_specs["lm_head"]["param"][1], base_lm_head[1] * 0.75)
    set_zero_sparse_grads(micro0_ctx)
    runtime.accumulate_gradients(micro0_ctx)

    micro1 = build_fixed_step_meta(
        slot_to_global=[2, -1],
        stage_slots=[0],
        stage_ids=[2],
        writeback_slots=[0],
        writeback_ids=[2],
        grad_accum_ids=union_ids,
        grad_accum_steps=2,
        grad_accum_micro_step=1,
        is_grad_accum_boundary=True,
        is_last_step=False,
    )
    micro1_ctx = runtime.prepare_step(micro1, cold_row_decay=0.25)
    assert torch.allclose(runtime.table_specs["lm_head"]["param"][1], base_lm_head[1] * 0.75)
    set_zero_sparse_grads(micro1_ctx)
    runtime.accumulate_gradients(micro1_ctx)
    runtime.apply_accumulated_gradients()


def test_dynamic_sparse_cold_row_decay_requires_fixed_u_mode():
    runtime = DynamicVocabRuntime(
        build_tiny_model(vocab_size=8),
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
    )

    with pytest.raises(ValueError, match="fixed-U"):
        runtime.prepare_step(torch.tensor([0, 1], dtype=torch.long), cold_row_decay=0.1)


def test_fixed_u_sparse_grad_accumulation_matches_single_union_update():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    reference_model = build_tiny_model(vocab_size=10)
    reference_model.load_state_dict(model.state_dict())

    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=3,
        grad_accum_u_max=4,
    )
    reference_runtime = DynamicVocabRuntime(
        reference_model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
    )
    union_ids = [1, 3, 7, 9]
    union_slot_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    union_step1_slot_ids = torch.tensor([3, 1, 2], dtype=torch.long)
    union_slot_by_token = {token_id: slot_id for slot_id, token_id in enumerate(union_ids)}

    step0 = build_fixed_step_meta(
        slot_to_global=[1, 3, 7],
        stage_slots=[0, 1, 2],
        stage_ids=[1, 3, 7],
        writeback_slots=[0],
        writeback_ids=[1],
        is_last_step=False,
        grad_accum_ids=union_ids,
        grad_accum_steps=2,
        grad_accum_micro_step=0,
        is_grad_accum_boundary=False,
    )
    step0_ctx = runtime.prepare_step(step0)
    slots0 = step0_ctx.active_slot_ids_cpu
    assert slots0 is not None
    assert step0_ctx.lm_head_active_ids_cpu is not None
    assert step0_ctx.lm_head_active_slot_ids_cpu is not None
    lm_head_slots0 = {
        int(token_id): int(slot_id)
        for token_id, slot_id in zip(step0_ctx.lm_head_active_ids_cpu.tolist(), step0_ctx.lm_head_active_slot_ids_cpu.tolist())
    }
    step0_ctx.active_vocab["wte"].grad = torch.zeros_like(step0_ctx.active_vocab["wte"])
    step0_ctx.active_vocab["wte"].grad[union_slot_by_token[1]] = 1
    step0_ctx.active_vocab["wte"].grad[union_slot_by_token[3]] = 2
    step0_ctx.active_vocab["wte"].grad[union_slot_by_token[7]] = 3
    step0_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step0_ctx.active_vocab["lm_head"])
    step0_ctx.active_vocab["lm_head"].grad[lm_head_slots0[1]] = 10
    step0_ctx.active_vocab["lm_head"].grad[lm_head_slots0[3]] = 20
    step0_ctx.active_vocab["lm_head"].grad[lm_head_slots0[7]] = 30
    for value_embed in step0_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.zeros_like(value_embed)
        value_embed.grad[union_slot_by_token[1]] = 100
        value_embed.grad[union_slot_by_token[3]] = 200
        value_embed.grad[union_slot_by_token[7]] = 300
    runtime.accumulate_gradients(step0_ctx)

    assert step0_ctx.grad_accum_queue_count == 0
    assert step0_ctx.grad_accum_resident_count == 3
    assert runtime.fixed_params["wte"].grad is not None
    assert torch.allclose(runtime.fixed_params["wte"].grad[union_slot_by_token[1]], torch.ones_like(runtime.fixed_params["wte"].grad[union_slot_by_token[1]]) * 1)
    assert torch.allclose(runtime.fixed_params["wte"].grad[union_slot_by_token[3]], torch.ones_like(runtime.fixed_params["wte"].grad[union_slot_by_token[3]]) * 2)
    assert torch.allclose(runtime.fixed_params["wte"].grad[union_slot_by_token[7]], torch.ones_like(runtime.fixed_params["wte"].grad[union_slot_by_token[7]]) * 3)
    assert torch.allclose(runtime.fixed_params["wte"].grad[union_slot_by_token[9]], torch.zeros_like(runtime.fixed_params["wte"].grad[union_slot_by_token[9]]))

    wte_param = runtime.table_specs["wte"]["param"]
    assert runtime.state[wte_param]["step"] == 0

    step1 = build_fixed_step_meta(
        slot_to_global=[9, 3, 7],
        stage_slots=[0],
        stage_ids=[9],
        writeback_slots=[1],
        writeback_ids=[3],
        is_last_step=False,
        grad_accum_ids=union_ids,
        grad_accum_steps=2,
        grad_accum_micro_step=1,
        is_grad_accum_boundary=True,
    )
    step1_ctx = runtime.prepare_step(step1)
    slots1 = step1_ctx.active_slot_ids_cpu
    assert slots1 is not None
    assert step1_ctx.lm_head_active_ids_cpu is not None
    assert step1_ctx.lm_head_active_slot_ids_cpu is not None
    lm_head_slots1 = {
        int(token_id): int(slot_id)
        for token_id, slot_id in zip(step1_ctx.lm_head_active_ids_cpu.tolist(), step1_ctx.lm_head_active_slot_ids_cpu.tolist())
    }
    assert step1_ctx.active_vocab["wte"].grad is not None
    step1_ctx.active_vocab["wte"].grad[union_slot_by_token[9]] += 6
    step1_ctx.active_vocab["wte"].grad[union_slot_by_token[3]] += 4
    step1_ctx.active_vocab["wte"].grad[union_slot_by_token[7]] += 5
    assert step1_ctx.active_vocab["lm_head"].grad is not None
    step1_ctx.active_vocab["lm_head"].grad[lm_head_slots1[9]] += 60
    step1_ctx.active_vocab["lm_head"].grad[lm_head_slots1[3]] += 40
    step1_ctx.active_vocab["lm_head"].grad[lm_head_slots1[7]] += 50
    for value_embed in step1_ctx.active_vocab["value_embeds"].values():
        assert value_embed.grad is not None
        value_embed.grad[union_slot_by_token[9]] += 600
        value_embed.grad[union_slot_by_token[3]] += 400
        value_embed.grad[union_slot_by_token[7]] += 500
    runtime.accumulate_gradients(step1_ctx)
    metrics = runtime.apply_accumulated_gradients()

    assert metrics.grad_accum_queue_count == 0
    assert metrics.grad_accum_resident_count == 4

    reference_ctx = reference_runtime.prepare_step(torch.tensor(union_ids, dtype=torch.long))
    reference_ctx.active_vocab["wte"].grad = torch.stack([
        torch.ones_like(reference_ctx.active_vocab["wte"][0]) * 1,
        torch.ones_like(reference_ctx.active_vocab["wte"][1]) * 6,
        torch.ones_like(reference_ctx.active_vocab["wte"][2]) * 8,
        torch.ones_like(reference_ctx.active_vocab["wte"][3]) * 6,
    ])
    reference_ctx.active_vocab["lm_head"].grad = torch.stack([
        torch.ones_like(reference_ctx.active_vocab["lm_head"][0]) * 10,
        torch.ones_like(reference_ctx.active_vocab["lm_head"][1]) * 60,
        torch.ones_like(reference_ctx.active_vocab["lm_head"][2]) * 80,
        torch.ones_like(reference_ctx.active_vocab["lm_head"][3]) * 60,
    ])
    for value_embed in reference_ctx.active_vocab["value_embeds"].values():
        value_embed.grad = torch.stack([
            torch.ones_like(value_embed[0]) * 100,
            torch.ones_like(value_embed[1]) * 600,
            torch.ones_like(value_embed[2]) * 800,
            torch.ones_like(value_embed[3]) * 600,
        ])
    reference_runtime.step(reference_ctx)

    union_ids_tensor = torch.tensor(union_ids, dtype=torch.long)
    for name in runtime.table_specs:
        runtime_param = runtime.table_specs[name]["param"]
        reference_param = reference_runtime.table_specs[name]["param"]
        assert torch.allclose(runtime_param[union_ids_tensor], reference_param[union_ids_tensor])
        assert runtime.state[runtime_param]["step"] == reference_runtime.state[reference_param]["step"]
    assert torch.equal(runtime.token_event_step_count_cpu[union_ids_tensor], torch.ones_like(union_ids_tensor))
    assert torch.equal(reference_runtime.token_event_step_count_cpu[union_ids_tensor], torch.ones_like(union_ids_tensor))

    runtime.flush_active_to_cpu()


def test_fixed_u_sparse_grad_accumulation_preserves_live_overlap_state():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=3,
        grad_accum_u_max=4,
    )

    union_ids = [1, 3, 7, 9]
    union_slot_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    union_step1_slot_ids = torch.tensor([3, 1, 2], dtype=torch.long)
    step0 = build_fixed_step_meta(
        slot_to_global=[1, 3, 7],
        stage_slots=[0, 1, 2],
        stage_ids=[1, 3, 7],
        writeback_slots=[0],
        writeback_ids=[1],
        is_last_step=False,
        grad_accum_ids=union_ids,
        grad_accum_steps=2,
        grad_accum_micro_step=0,
        is_grad_accum_boundary=False,
    )
    step0_ctx = runtime.prepare_step(step0)
    assert step0_ctx.active_slot_ids_cpu is not None
    for name, param in runtime.fixed_params.items():
        param.grad = torch.zeros_like(param)
        param.grad[union_slot_ids] = 1
    runtime.accumulate_gradients(step0_ctx)

    assert step0_ctx.grad_accum_queue_count == 0
    assert step0_ctx.grad_accum_resident_count == 3

    step1 = build_fixed_step_meta(
        slot_to_global=[9, 3, 7],
        stage_slots=[0],
        stage_ids=[9],
        writeback_slots=[1],
        writeback_ids=[3],
        is_last_step=False,
        grad_accum_ids=union_ids,
        grad_accum_steps=2,
        grad_accum_micro_step=1,
        is_grad_accum_boundary=True,
    )
    step1_ctx = runtime.prepare_step(step1)
    assert step1_ctx.active_slot_ids_cpu is not None
    for name, param in runtime.fixed_params.items():
        assert param.grad is not None
        param.grad[union_step1_slot_ids] += 2
    runtime.accumulate_gradients(step1_ctx)
    metrics = runtime.apply_accumulated_gradients()

    assert runtime._fixed_live_state
    assert runtime.fixed_slot_to_global_cpu is not None
    assert torch.equal(runtime.fixed_slot_to_global_cpu[:3], torch.tensor([9, 3, 7], dtype=torch.long))
    assert runtime.fixed_input_slot_to_global_cpu is not None
    assert torch.equal(runtime.fixed_input_slot_to_global_cpu[:4], torch.tensor(union_ids, dtype=torch.long))
    assert metrics.live_count == 4
    assert metrics.unique_count == 4
    assert metrics.u_capacity == 4


def test_fixed_u_grad_accum_lm_head_first_hot_lr_matches_single_union_step():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=10)
    reference_model = build_tiny_model(vocab_size=10)
    reference_model.load_state_dict(model.state_dict())

    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        first_hot_unembedding_lr=0.003,
        fixed_u_max=2,
        grad_accum_u_max=4,
    )
    reference_runtime = DynamicVocabRuntime(
        reference_model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        first_hot_unembedding_lr=0.003,
    )

    union_ids = [1, 3, 7, 9]
    union_slot_by_token = {token_id: slot_id for slot_id, token_id in enumerate(union_ids)}

    step0_ctx = runtime.prepare_step(
        build_fixed_step_meta(
            slot_to_global=[1, 3],
            stage_slots=[0, 1],
            stage_ids=[1, 3],
            writeback_slots=[0],
            writeback_ids=[1],
            is_last_step=False,
            grad_accum_ids=union_ids,
            grad_accum_steps=2,
            grad_accum_micro_step=0,
            is_grad_accum_boundary=False,
        )
    )
    step0_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step0_ctx.active_vocab["lm_head"])
    step0_ctx.active_vocab["lm_head"].grad[union_slot_by_token[1]] += 10
    step0_ctx.active_vocab["lm_head"].grad[union_slot_by_token[3]] += 20
    runtime.accumulate_gradients(step0_ctx)

    step1_ctx = runtime.prepare_step(
        build_fixed_step_meta(
            slot_to_global=[7, 9],
            stage_slots=[],
            stage_ids=[],
            writeback_slots=[],
            writeback_ids=[],
            is_last_step=True,
            grad_accum_ids=union_ids,
            grad_accum_steps=2,
            grad_accum_micro_step=1,
            is_grad_accum_boundary=True,
        )
    )
    if step1_ctx.active_vocab["lm_head"].grad is None:
        step1_ctx.active_vocab["lm_head"].grad = torch.zeros_like(step1_ctx.active_vocab["lm_head"])
    step1_ctx.active_vocab["lm_head"].grad[union_slot_by_token[7]] += 30
    step1_ctx.active_vocab["lm_head"].grad[union_slot_by_token[9]] += 40
    runtime.accumulate_gradients(step1_ctx)
    runtime.apply_accumulated_gradients()

    reference_ctx = reference_runtime.prepare_step(torch.tensor(union_ids, dtype=torch.long))
    reference_ctx.active_vocab["lm_head"].grad = torch.stack([
        torch.ones_like(reference_ctx.active_vocab["lm_head"][0]) * 10,
        torch.ones_like(reference_ctx.active_vocab["lm_head"][1]) * 20,
        torch.ones_like(reference_ctx.active_vocab["lm_head"][2]) * 30,
        torch.ones_like(reference_ctx.active_vocab["lm_head"][3]) * 40,
    ])
    reference_runtime.step(reference_ctx)

    union_ids_tensor = torch.tensor(union_ids, dtype=torch.long)
    runtime_param = runtime.table_specs["lm_head"]["param"]
    reference_param = reference_runtime.table_specs["lm_head"]["param"]
    assert torch.allclose(runtime_param[union_ids_tensor], reference_param[union_ids_tensor])
    assert torch.allclose(
        runtime.state[runtime_param]["exp_avg"][union_ids_tensor],
        reference_runtime.state[reference_param]["exp_avg"][union_ids_tensor],
    )
    assert torch.allclose(
        runtime.state[runtime_param]["exp_avg_sq"][union_ids_tensor],
        reference_runtime.state[reference_param]["exp_avg_sq"][union_ids_tensor],
    )


def test_fixed_u_sparse_grad_accumulation_preserves_reentrant_rows():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=32)
    reference_model = build_tiny_model(vocab_size=32)
    reference_model.load_state_dict(model.state_dict())

    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=6,
        grad_accum_u_max=12,
    )
    reference_runtime = DynamicVocabRuntime(
        reference_model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
    )

    microsteps = [
        {
            "slot_to_global": [1, 3, 4, 6, 9, 11],
            "stage_slots": [0, 1, 2, 3, 4, 5],
            "stage_ids": [1, 3, 4, 6, 9, 11],
            "writeback_slots": [1, 2],
            "writeback_ids": [3, 4],
            "active_ids": [1, 3, 4, 6, 9, 11],
        },
        {
            "slot_to_global": [1, 5, 10, 6, 9, 11],
            "stage_slots": [1, 2],
            "stage_ids": [5, 10],
            "writeback_slots": [2, 5],
            "writeback_ids": [10, 11],
            "active_ids": [1, 5, 6, 9, 10, 11],
        },
        {
            "slot_to_global": [1, 5, 0, 6, 9, 4],
            "stage_slots": [2, 5],
            "stage_ids": [0, 4],
            "writeback_slots": [0, 4],
            "writeback_ids": [1, 9],
            "active_ids": [0, 1, 4, 5, 6, 9],
        },
        {
            "slot_to_global": [3, 5, 0, 6, 10, 4],
            "stage_slots": [0, 4],
            "stage_ids": [3, 10],
            "writeback_slots": [],
            "writeback_ids": [],
            "active_ids": [0, 3, 4, 5, 6, 10],
        },
    ]
    union_ids = [0, 1, 3, 4, 5, 6, 9, 10, 11]
    union_slot_by_token = {token_id: slot_id for slot_id, token_id in enumerate(union_ids)}

    for micro_idx, micro in enumerate(microsteps):
        step_ctx = runtime.prepare_step(
            build_fixed_step_meta(
                slot_to_global=micro["slot_to_global"],
                stage_slots=micro["stage_slots"],
                stage_ids=micro["stage_ids"],
                writeback_slots=micro["writeback_slots"],
                writeback_ids=micro["writeback_ids"],
                inputs_union_cpu_local=[[union_ids.index(token_id) for token_id in micro["active_ids"]]],
                is_last_step=False,
                grad_accum_ids=union_ids,
                grad_accum_steps=len(microsteps),
                grad_accum_micro_step=micro_idx,
                is_grad_accum_boundary=micro_idx == len(microsteps) - 1,
            )
        )

        assert step_ctx.union_inputs is not None

        token_by_slot = {
            micro["slot_to_global"].index(token_id): token_id
            for token_id in micro["active_ids"]
        }
        assert step_ctx.lm_head_active_ids_cpu is not None
        assert step_ctx.lm_head_active_slot_ids_cpu is not None
        lm_head_slot_by_token = {
            int(token_id): int(slot_id)
            for token_id, slot_id in zip(step_ctx.lm_head_active_ids_cpu.tolist(), step_ctx.lm_head_active_slot_ids_cpu.tolist())
        }
        for name, param in runtime.fixed_params.items():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            scale = 10.0 if name == "lm_head" else 1.0
            if name.startswith("value_embeds."):
                scale = 100.0
            for slot_id in step_ctx.active_slot_ids_cpu.tolist():
                token_id = token_by_slot[slot_id]
                target_slot_id = lm_head_slot_by_token[token_id] if name == "lm_head" else union_slot_by_token[token_id]
                param.grad[target_slot_id] += scale * (token_id + 1) * (micro_idx + 1) / len(microsteps)
        runtime.accumulate_gradients(step_ctx)

    metrics = runtime.apply_accumulated_gradients()
    runtime.flush_active_to_cpu()

    reference_ctx = reference_runtime.prepare_step(torch.tensor(union_ids, dtype=torch.long))
    for name in ["wte", "lm_head"]:
        scale = 10.0 if name == "lm_head" else 1.0
        grad = torch.zeros_like(reference_ctx.active_vocab[name])
        for idx, token_id in enumerate(union_ids):
            total = 0.0
            for micro_idx, micro in enumerate(microsteps):
                if token_id in micro["active_ids"]:
                    total += scale * (token_id + 1) * (micro_idx + 1) / len(microsteps)
            grad[idx] = total
        reference_ctx.active_vocab[name].grad = grad
    for value_embed in reference_ctx.active_vocab["value_embeds"].values():
        grad = torch.zeros_like(value_embed)
        for idx, token_id in enumerate(union_ids):
            total = 0.0
            for micro_idx, micro in enumerate(microsteps):
                if token_id in micro["active_ids"]:
                    total += 100.0 * (token_id + 1) * (micro_idx + 1) / len(microsteps)
            grad[idx] = total
        value_embed.grad = grad
    reference_runtime.step(reference_ctx)

    union_ids_tensor = torch.tensor(union_ids, dtype=torch.long)
    for name in runtime.table_specs:
        runtime_param = runtime.table_specs[name]["param"]
        reference_param = reference_runtime.table_specs[name]["param"]
        assert torch.allclose(
            runtime_param[union_ids_tensor],
            reference_param[union_ids_tensor],
            atol=1e-3,
            rtol=1e-5,
        )
        assert torch.allclose(
            runtime.state[runtime_param]["exp_avg"][union_ids_tensor],
            reference_runtime.state[reference_param]["exp_avg"][union_ids_tensor],
            atol=4.0,
            rtol=1e-2,
        )
        assert torch.allclose(
            runtime.state[runtime_param]["exp_avg_sq"][union_ids_tensor],
            reference_runtime.state[reference_param]["exp_avg_sq"][union_ids_tensor],
            atol=2048.0,
            rtol=2e-2,
        )
    assert runtime._fixed_live_state
    assert runtime.fixed_slot_to_global_cpu is not None
    assert torch.equal(runtime.fixed_slot_to_global_cpu[:6], torch.tensor([3, 5, 0, 6, 10, 4], dtype=torch.long))
    assert runtime.fixed_input_slot_to_global_cpu is not None
    assert torch.equal(runtime.fixed_input_slot_to_global_cpu[:9], torch.tensor(union_ids, dtype=torch.long))
    assert metrics.grad_accum_queue_count == 0
    assert metrics.grad_accum_resident_count == 9
    assert metrics.live_count == 9
    assert metrics.unique_count == 9
    assert metrics.u_capacity == 12


def test_fixed_u_grad_accum_prepare_emits_union_inputs_for_input_tables():
    runtime = DynamicVocabRuntime(
        build_tiny_model(vocab_size=16),
        device="cpu",
        embedding_lr=0.01,
        value_embedding_lr=0.01,
        unembedding_lr=0.01,
        fixed_u_max=4,
        grad_accum_u_max=6,
    )

    step_ctx = runtime.prepare_step(
        build_fixed_step_meta(
            slot_to_global=[4, 7, 9, -1],
            stage_slots=[0, 1, 2],
            stage_ids=[4, 7, 9],
            writeback_slots=[],
            writeback_ids=[],
            inputs_union_cpu_local=[[2, 0, 1, 2]],
            grad_accum_ids=[7, 9, 4],
            grad_accum_steps=2,
            grad_accum_micro_step=0,
            is_grad_accum_boundary=False,
        )
    )

    assert step_ctx.union_inputs is not None
    assert torch.equal(step_ctx.union_inputs.cpu(), torch.tensor([[2, 0, 1, 2]], dtype=torch.long))


def test_fixed_u_grad_accum_prepare_exposes_exact_union_width_active_vocab():
    torch.manual_seed(0)
    model = build_tiny_model(vocab_size=16)
    runtime = DynamicVocabRuntime(
        model,
        device="cpu",
        embedding_lr=0.05,
        value_embedding_lr=0.04,
        unembedding_lr=0.03,
        fixed_u_max=6,
        grad_accum_u_max=12,
    )

    step_ctx = runtime.prepare_step(
        build_fixed_step_meta(
            slot_to_global=[7, 9, 4, -1, -1, -1],
            stage_slots=[0, 1, 2],
            stage_ids=[7, 9, 4],
            writeback_slots=[],
            writeback_ids=[],
            grad_accum_ids=[7, 9, 4],
            grad_accum_steps=2,
            grad_accum_micro_step=0,
            is_grad_accum_boundary=False,
            inputs_union_cpu_local=[[2, 0, 1, 2]],
        )
    )

    assert step_ctx.active_vocab is not None
    assert step_ctx.active_vocab["wte"].shape[0] == 3
    assert step_ctx.active_vocab["lm_head"].shape[0] == 3
    assert step_ctx.active_vocab["value_embeds"]["1"].shape[0] == 3
    assert "logit_mask" not in step_ctx.active_vocab
    assert runtime.fixed_input_slot_to_global_cpu is not None
    assert torch.equal(runtime.fixed_input_slot_to_global_cpu[:3], torch.tensor([7, 9, 4], dtype=torch.long))

import torch

from nanochat.prefetch import AsyncLoaderPrefetcher, clone_prefetched_item


def test_clone_prefetched_item_preserves_tensor_values_without_aliasing():
    source = torch.arange(8, dtype=torch.long).view(2, 4).pin_memory()
    cloned = clone_prefetched_item(source)

    source.zero_()

    assert torch.equal(cloned, torch.arange(8, dtype=torch.long).view(2, 4))
    assert cloned.data_ptr() != source.data_ptr()
    assert cloned.is_pinned()


def test_async_loader_prefetcher_copies_reused_loader_buffers():
    buffer = torch.zeros((2, 4), dtype=torch.long).pin_memory()

    def loader():
        for value in range(3):
            buffer.fill_(value)
            yield buffer

    prefetched = AsyncLoaderPrefetcher(loader(), max_prefetch=2)
    batches = list(prefetched)

    assert len(batches) == 3
    assert torch.equal(batches[0], torch.zeros((2, 4), dtype=torch.long))
    assert torch.equal(batches[1], torch.ones((2, 4), dtype=torch.long))
    assert torch.equal(batches[2], torch.full((2, 4), 2, dtype=torch.long))
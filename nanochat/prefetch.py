import queue
import threading

import torch


def _clone_tensor_for_prefetch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == "cpu":
        clone = torch.empty_like(tensor, device=tensor.device, pin_memory=tensor.is_pinned())
        clone.copy_(tensor)
        return clone
    return tensor.clone()


def clone_prefetched_item(item):
    if isinstance(item, torch.Tensor):
        return _clone_tensor_for_prefetch(item)
    if isinstance(item, dict):
        return {key: clone_prefetched_item(value) for key, value in item.items()}
    if isinstance(item, tuple):
        return tuple(clone_prefetched_item(value) for value in item)
    if isinstance(item, list):
        return [clone_prefetched_item(value) for value in item]
    return item


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
                self.queue.put(clone_prefetched_item(next(self.loader)))
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
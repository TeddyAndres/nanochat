import torch
import torch.distributed as dist


class SparseHybridOptimizer:
    def __init__(self, dense_optimizer, sparse_optimizer, sparse_params):
        self.dense_optimizer = dense_optimizer
        self.sparse_optimizer = sparse_optimizer
        self.sparse_params = list(sparse_params)
        self.param_groups = self.dense_optimizer.param_groups + self.sparse_optimizer.param_groups

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        self.dense_optimizer.zero_grad(set_to_none=set_to_none)
        self.sparse_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _sync_sparse_grads_ddp(self):
        if not (dist.is_available() and dist.is_initialized()):
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        for p in self.sparse_params:
            if p.grad is None:
                continue
            grad = p.grad.coalesce()
            values = grad.values().contiguous()
            if values.is_cuda:
                dist.all_reduce(values, op=dist.ReduceOp.AVG)
            else:
                if torch.cuda.is_available():
                    orig_device = values.device
                    values_cuda = values.to("cuda", non_blocking=True)
                    dist.all_reduce(values_cuda, op=dist.ReduceOp.AVG)
                    values = values_cuda.to(orig_device)
                else:
                    dist.all_reduce(values, op=dist.ReduceOp.AVG)
            p.grad = torch.sparse_coo_tensor(
                grad.indices(),
                values,
                size=grad.size(),
                dtype=values.dtype,
                device=values.device,
            ).coalesce()

    @torch.no_grad()
    def step(self, closure=None):
        self._sync_sparse_grads_ddp()
        if closure is None:
            loss = self.dense_optimizer.step()
        else:
            loss = self.dense_optimizer.step(closure=closure)
        self.sparse_optimizer.step()
        return loss

    def state_dict(self):
        return {
            "dense_optimizer": self.dense_optimizer.state_dict(),
            "sparse_optimizer": self.sparse_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.dense_optimizer.load_state_dict(state_dict["dense_optimizer"])
        self.sparse_optimizer.load_state_dict(state_dict["sparse_optimizer"])
import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Any, Type
from torch.optim import Optimizer

from cs336_basics.model import Embedding, Linear


class DDP_Sync(nn.Module):
    """Distributed Data Parallel wrapper with synchronous gradient synchronization."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._broadcast_module_state()

    def _broadcast_module_state(self) -> None:
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(dist.get_world_size())


class DDP(nn.Module):
    """Distributed Data Parallel wrapper with asynchronous gradient synchronization."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._pending_handles: list[dist.Work] = []
        self._broadcast_module_state()

    def _make_grad_hook(self):
        def hook(param: torch.Tensor) -> None:
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._pending_handles.append(handle)

        return hook

    def _broadcast_module_state(self) -> None:
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_grad_hook())

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for handle in self._pending_handles:
            handle.wait()
        self._pending_handles.clear()

        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.div_(dist.get_world_size())


class OptimizerStateSharder:
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.all_params = list(params)
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self._shard_param_groups()

    def _shard_param_groups(self):
        self.local_param_groups = [
            p for i, p in enumerate(self.all_params) if i % self.world_size == self.rank
        ]
        self.optimizer = self.optimizer_cls(self.local_param_groups, **self.kwargs)

    def step(self):
        self.optimizer.step()
        with torch.no_grad():
            for i, p in enumerate(self.all_params):
                owner = i % self.world_size
                dist.broadcast(p.data, src=owner)

    def zero_grad(self):
        self.optimizer.zero_grad()


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.fsdp_layers: list[nn.Module] = []
        self._pending_handles: list[dist.Work] = []
        self._fsdp_params: set[nn.Parameter] = set()

        for mod in self.module.modules():
            if isinstance(mod, (Linear, Embedding)):
                self._shard_layers(mod)
                self._register_hooks(mod)
                self.fsdp_layers.append(mod)
                self._fsdp_params.add(mod.weight)

        for p in self.module.parameters():
            if p.requires_grad and p not in self._fsdp_params:
                p.register_post_accumulate_grad_hook(self._make_replicated_grad_hook())

    def _shard_layers(self, mod: nn.Module):
        dim0 = mod.weight.data.shape[0]
        shard = dim0 // self.world_size
        shard_weight = mod.weight.data[self.rank * shard : (self.rank + 1) * shard]
        mod.weight = nn.Parameter(shard_weight)

    def _register_hooks(self, mod: nn.Module):
        mod.register_forward_pre_hook(self._forward_pre_hook())
        mod.register_forward_hook(self._forward_post_hook())

        if isinstance(mod, Linear):
            mod.register_full_backward_pre_hook(self._backward_pre_hook())

        mod.weight.register_post_accumulate_grad_hook(self._reduce_scatter_hook(mod))

    def _forward_pre_hook(self):
        def hook(module: nn.Module, _input: Any):
            shard = module.weight.data
            module._saved_shard = shard

            gather_shard = shard
            if self.compute_dtype is not None:
                gather_shard = shard.to(self.compute_dtype)

            gather_list = [torch.empty_like(gather_shard) for _ in range(self.world_size)]
            dist.all_gather(gather_list, gather_shard)
            module.weight.data = torch.cat(gather_list, dim=0)

        return hook

    def _forward_post_hook(self):
        def hook(module: nn.Module, _input: Any, _output: Any):
            module.weight.data = module._saved_shard
            del module._saved_shard

        return hook

    def _backward_pre_hook(self):
        def hook(module: nn.Module, _grad_output: Any):
            shard = module.weight.data
            module._saved_shard_bwd = shard

            gather_shard = shard
            if self.compute_dtype is not None:
                gather_shard = shard.to(self.compute_dtype)

            gather_list = [torch.empty_like(gather_shard) for _ in range(self.world_size)]
            dist.all_gather(gather_list, gather_shard)
            module.weight.data = torch.cat(gather_list, dim=0)

        return hook

    def _reduce_scatter_hook(self, module: nn.Module):
        def hook(param: torch.Tensor):
            if param.grad is None:
                return

            if isinstance(module, Linear) and hasattr(module, "_saved_shard_bwd"):
                module.weight.data = module._saved_shard_bwd
                del module._saved_shard_bwd

            full_grad = param.grad.to(torch.float32)
            shard_grad = torch.empty_like(param.data)
            dist.reduce_scatter_tensor(shard_grad, full_grad.contiguous(), op=dist.ReduceOp.SUM)
            shard_grad.div_(self.world_size)
            param.grad = shard_grad

        return hook

    def _make_replicated_grad_hook(self):
        def hook(param: torch.Tensor):
            if param.grad is None:
                return
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._pending_handles.append(handle)

        return hook

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for handle in self._pending_handles:
            handle.wait()
        self._pending_handles.clear()

        for p in self.module.parameters():
            if p.requires_grad and p not in self._fsdp_params and p.grad is not None:
                p.grad.div_(self.world_size)

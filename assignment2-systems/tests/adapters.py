from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from cs336_basics.model import Embedding, Linear
from ddp_and_fsdp import DDP, FSDP, OptimizerStateSharder


def get_flashattention_autograd_function_pytorch() -> type:
    raise NotImplementedError


def get_flashattention_autograd_function_triton() -> type:
    raise NotImplementedError


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return DDP(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    return ddp_model.finish_gradient_synchronization()


def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    return FSDP(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    return fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    full_params: dict[str, torch.Tensor] = {}
    world_size = dist.get_world_size()
    named_modules = dict(fsdp_model.module.named_modules())

    for name, param in fsdp_model.module.named_parameters():
        mod_name = name.rsplit(".", 1)[0]
        mod = named_modules[mod_name]
        if isinstance(mod, (Linear, Embedding)):
            gather_list = [torch.empty_like(param.data) for _ in range(world_size)]
            dist.all_gather(gather_list, param.data)
            full_params[name] = torch.cat(gather_list, dim=0)
        else:
            full_params[name] = param.data

    return full_params


def get_sharded_optimizer(
    params,
    optimizer_cls: type[torch.optim.Optimizer],
    **kwargs,
) -> torch.optim.Optimizer:
    return OptimizerStateSharder(params, optimizer_cls, **kwargs)

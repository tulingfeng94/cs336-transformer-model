import time
import torch
import math
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
import cs336_basics.model
from statistics import mean
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

def run_forward_only(model: BasicsTransformerLM, x: torch.Tensor):
    with torch.no_grad():
        model.forward(x)

def run_forward_and_backward(model: BasicsTransformerLM, optimizer: AdamW, x: torch.Tensor, y: torch.Tensor):
    optimizer.zero_grad()
    logits = model.forward(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()

def run_full(model: BasicsTransformerLM, optimizer: AdamW, x: torch.Tensor, y: torch.Tensor):
    optimizer.zero_grad()
    logits = model.forward(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step()

def annotated_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    with nvtx.range("computing attention scores"):
        attention_scores = Q @ K.transpose(-2, -1) / math.sqrt(K.size(-1))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, -float("inf"))
    with nvtx.range("computing softmax"):
        attention_weights = F.softmax(attention_scores, dim=-1)
    with nvtx.range("final matmul"):
        attention_output = attention_weights @ V
    return attention_output

def benchmarking(num_warmup: int = 5, num_trials: int = 10):
    # cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    model = BasicsTransformerLM(
        d_model=1024,
        d_ff=4096,
        num_layers=24,
        num_heads=16,
        context_length=512,
        vocab_size=10000,
        rope_theta=10000.0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = AdamW(model.parameters())
    x = torch.randint(0, 10000, (4, 512), device=device)
    y = torch.randint(0, 10000, (4, 512), device=device)

    for _ in range(num_warmup):
        run_forward_only(model, x)
        run_forward_and_backward(model, optimizer, x, y)
        run_full(model, optimizer, x, y)
    torch.cuda.synchronize()
    times_fwd: list[float] = []
    times_fwdbwd: list[float] = []
    times_full: list[float] = []

    for trial in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Forward only
        start_event.record()
        run_forward_only(model, x)
        end_event.record()
        torch.cuda.synchronize()
        times_fwd.append(start_event.elapsed_time(end_event))

        # Forward and backward
        start_event.record()
        run_forward_and_backward(model, optimizer, x, y)
        end_event.record()
        torch.cuda.synchronize()
        times_fwdbwd.append(start_event.elapsed_time(end_event))

        # Full
        start_event.record()
        run_full(model, optimizer, x, y)
        end_event.record()
        torch.cuda.synchronize()
        times_full.append(start_event.elapsed_time(end_event))

    print(f"Forward only: {mean(times_fwd)}ms")
    print(f"Forward and backward: {mean(times_fwdbwd)}ms")
    print(f"Full: {mean(times_full)}ms")

def profiling(num_warmup: int = 5):
    model = BasicsTransformerLM(
        d_model=1024,
        d_ff=4096,
        num_layers=24,
        num_heads=16,
        context_length=512,
        vocab_size=10000,
        rope_theta=10000.0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = AdamW(model.parameters())
    x = torch.randint(0, 10000, (4, 512), device=device)
    y = torch.randint(0, 10000, (4, 512), device=device)

    for _ in range(num_warmup):
        run_full(model, optimizer, x, y)

    with nvtx.range("profiling", color="purple"):
        optimizer.zero_grad()
        with nvtx.range("forward", color="green"):
            logits = model.forward(x)
        
        with nvtx.range("backward", color="red"):
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
        
        with nvtx.range("optimize", color="blue"):
            optimizer.step()
        
    torch.cuda.synchronize()

if __name__ == "__main__":
    # benchmarking()
    profiling()
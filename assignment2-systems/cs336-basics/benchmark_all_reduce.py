"""Benchmark the runtime of ``dist.all_reduce`` in a single-node, multi-process setup.

Sweeps over:
  * data size:  1MB, 10MB, 100MB, 1GB  (float32 tensors)
  * world size: 2, 4, 6 processes

Works with either the ``nccl`` backend (one process per GPU) or the ``gloo``
backend (CPU only), so it can run on a GPU node or fall back to CPU for a quick
smoke test.

Example:
    # On a node with >= 6 GPUs:
    python benchmark_all_reduce.py --backend nccl

    # CPU smoke test (small sizes / fewer procs recommended):
    python benchmark_all_reduce.py --backend gloo --world-sizes 2 4 --sizes 1MB 10MB
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

BYTES_PER_FLOAT32 = 4
_UNITS = {"KB": 2**10, "MB": 2**20, "GB": 2**30}


def parse_size(text: str) -> int:
    """Parse a size string like ``100MB`` or ``1GB`` into a number of bytes."""
    text = text.strip().upper()
    for suffix, factor in _UNITS.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * factor)
    return int(text)


@dataclass
class Result:
    backend: str
    world_size: int
    size_bytes: int
    size_label: str
    avg_ms: float
    bus_gbps: float


def _setup(rank: int, world_size: int, backend: str, port: str) -> str:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    if backend == "nccl":
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def _time_all_reduce(data: torch.Tensor, device: str, warmup: int, trials: int) -> float:
    is_cuda = device.startswith("cuda")

    for _ in range(warmup):
        dist.all_reduce(data, async_op=False)
    if is_cuda:
        torch.cuda.synchronize()
    dist.barrier()

    if is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(trials):
            dist.all_reduce(data, async_op=False)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(trials):
            dist.all_reduce(data, async_op=False)
        total_ms = (time.perf_counter() - t0) * 1e3

    return total_ms / trials


def _worker(
    rank: int,
    world_size: int,
    backend: str,
    port: str,
    sizes: list[tuple[str, int]],
    warmup: int,
    trials: int,
    result_path: str,
) -> None:
    device = _setup(rank, world_size, backend, port)
    rows: list[Result] = []

    for label, size_bytes in sizes:
        n = size_bytes // BYTES_PER_FLOAT32
        data = torch.randn(n, dtype=torch.float32, device=device)

        avg_ms = _time_all_reduce(data, device, warmup, trials)

        t = torch.tensor([avg_ms], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        avg_ms = float(t.item())

        factor = 2 * (world_size - 1) / world_size
        bus_gbps = (factor * size_bytes) / (avg_ms / 1e3) / 1e9

        rows.append(Result(backend, world_size, size_bytes, label, avg_ms, bus_gbps))
        del data
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if rank == 0:
        write_header = not os.path.exists(result_path)
        with open(result_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["backend", "world_size", "size_label", "size_bytes", "avg_ms", "bus_gbps"])
            for r in rows:
                writer.writerow(
                    [r.backend, r.world_size, r.size_label, r.size_bytes, f"{r.avg_ms:.4f}", f"{r.bus_gbps:.3f}"]
                )
        for r in rows:
            print(
                f"[{r.backend}] world_size={r.world_size:<2} size={r.size_label:>6} "
                f"avg={r.avg_ms:8.3f} ms  bus_bw={r.bus_gbps:7.2f} GB/s"
            )

    dist.barrier()
    dist.destroy_process_group()


def maybe_plot(csv_path: str, out_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not installed; skipping plots (CSV written). "
            "Install with `uv add matplotlib` to enable."
        )
        return

    import collections

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    by_ws: dict[int, list[tuple[int, float]]] = collections.defaultdict(list)
    for r in rows:
        by_ws[int(r["world_size"])].append((int(r["size_bytes"]), float(r["avg_ms"])))

    plt.figure(figsize=(7, 5))
    for ws in sorted(by_ws):
        pts = sorted(by_ws[ws])
        xs = [p[0] / 2**20 for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"{ws} procs")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Tensor size (MB)")
    plt.ylabel("all_reduce time (ms)")
    plt.title(f"all_reduce runtime ({rows[0]['backend']})")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    out = os.path.join(out_dir, "all_reduce_time.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--sizes", type=str, nargs="+", default=["1MB", "10MB", "100MB", "1GB"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--port", type=str, default="29500")
    parser.add_argument("--output-dir", type=str, default="all_reduce_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"all_reduce_{args.backend}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    sizes = [(s, parse_size(s)) for s in args.sizes]

    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise SystemExit("nccl backend requested but CUDA is not available.")
        max_ws = max(args.world_sizes)
        ngpu = torch.cuda.device_count()
        if max_ws > ngpu:
            raise SystemExit(f"Requested world_size={max_ws} but only {ngpu} GPUs are visible.")

    for world_size in args.world_sizes:
        print(f"\n=== backend={args.backend} world_size={world_size} ===")
        mp.spawn(
            _worker,
            args=(world_size, args.backend, args.port, sizes, args.warmup, args.trials, csv_path),
            nprocs=world_size,
            join=True,
        )

    print(f"\nResults written to {csv_path}")
    maybe_plot(csv_path, args.output_dir)


if __name__ == "__main__":
    main()

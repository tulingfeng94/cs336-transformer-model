import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import csv
import time
from dataclasses import dataclass, astuple

BYTES_PER_FLOAT32 = 4
_UNITS = {"GB": 2**30, "MB": 2**20, "KB": 2**10}


def human_size(nbytes: int) -> str:
    for unit, factor in _UNITS.items():
        if nbytes >= factor:
            return f"{nbytes // factor}{unit}"
    return f"{nbytes}B"


@dataclass
class Result:
    backend: str
    world_size: int
    size_bytes: int
    size_label: str
    avg_ms: float
    bus_gbps: float


def setup(rank: int, world_size: int, backend: str) -> str:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if backend == "nccl":
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def time_all_reduce(data: torch.Tensor, device: str, warmup: int, trials: int) -> float:
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


def worker(
    rank: int,
    world_size: int,
    backend: str,
    warmup: int,
    trials: int,
    size_list: list[int],
    csv_path: str,
) -> None:
    device = setup(rank, world_size, backend)
    rows: list[Result] = []

    for size in size_list:
        n = int(size) // BYTES_PER_FLOAT32
        data = torch.randn(n, dtype=torch.float32, device=device)
        avg_ms = time_all_reduce(data, device, warmup, trials)

        t = torch.tensor([avg_ms], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        avg_ms = float(t.item())

        factor = 2 * (world_size - 1) / world_size
        bus_gbps = (factor * size) / (avg_ms / 1e3) / 1e9

        rows.append(Result(backend, world_size, size, human_size(size), avg_ms, bus_gbps))
        del data
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if rank == 0:
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["backend", "world_size", "size_bytes", "size_label", "avg_ms", "bus_gbps"])
            for row in rows:
                writer.writerow(astuple(row))
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


if __name__ == "__main__":
    backend = "nccl"  # switch to "gloo" for a CPU smoke test
    size_list = [1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024, 1024 * 1024 * 1024]
    world_sizes_list = [2, 4, 6]
    trials = 20
    warmup = 5

    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "data_parallel.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    for world_size in world_sizes_list:
        print(f"\n=== backend={backend} world_size={world_size} ===")
        mp.spawn(
            fn=worker,
            args=(world_size, backend, warmup, trials, size_list, csv_path),
            nprocs=world_size,
            join=True,
        )

    print(f"\nResults written to {csv_path}")
    maybe_plot(csv_path, out_dir)

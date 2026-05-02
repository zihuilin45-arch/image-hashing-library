"""
parallel_hash.py — L3 process-level parallelism across images.

Two parallelism entry points:
  - parallel_hash(paths, workers): per-image Pool.map (one task per image).
    Works well when per-image compute >> per-task IPC overhead.
  - parallel_hash_batched(paths, workers, batch_size): each Pool task
    processes a batch of paths, amortising IPC cost. Essential when
    per-image compute is small (e.g., L1 iter 3's 2.5 ms), where the
    per-task version's IPC overhead dominates.

Correctness guarantee (tested in test_parallel.py):
  both variants == sequential_hash(paths) for all worker counts.
"""
from __future__ import annotations
from multiprocessing import Pool
from findlib.hasher import FINDNumpyHasher

# Module-level globals. Inside each worker process, these hold that
# worker's private hasher instance after `_init_worker` runs.
_WORKER_HASHER = None


def _init_worker():
    """Called once per worker process by Pool, before any tasks."""
    global _WORKER_HASHER
    _WORKER_HASHER = FINDNumpyHasher()


def _hash_one(path):
    """Per-image worker task: hashes one path, returns (path, hash_str)."""
    return path, str(_WORKER_HASHER.fromFile(path))


def _hash_batch(paths_batch):
    """
    Batched worker task: hashes a list of paths in one Pool invocation.

    This amortises fixed per-task IPC overhead (pickling paths in,
    unpickling hash strings out, Pool dispatch loop) across many images.
    Each paths_batch is typically 10-100 paths depending on batch_size.
    """
    return [(p, str(_WORKER_HASHER.fromFile(p))) for p in paths_batch]


def parallel_hash(paths, workers=4, chunksize=None):
    """
    Per-image parallel hashing (one Pool task per image).

    Returns list of (path, hash_str) tuples, preserving input order.
    """
    if chunksize is None:
        chunksize = max(1, len(paths) // (workers * 4))

    with Pool(processes=workers, initializer=_init_worker) as pool:
        results = pool.map(_hash_one, paths, chunksize=chunksize)
    return results


def parallel_hash_batched(paths, workers=4, batch_size=None):
    """
    Batched parallel hashing: each Pool task processes `batch_size` paths.

    Batch_size default: len(paths) // workers, giving one batch per worker.
    This minimises IPC at the cost of worst-case load imbalance (a slow
    worker blocks 1/workers of total throughput).

    Returns list of (path, hash_str) tuples, preserving input order.
    """
    if batch_size is None:
        batch_size = max(1, len(paths) // workers)

    # Split paths into batches of size batch_size
    batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]

    with Pool(processes=workers, initializer=_init_worker) as pool:
        batch_results = pool.map(_hash_batch, batches)

    # Flatten list-of-batches back into a flat list of (path, hash) tuples
    return [item for batch in batch_results for item in batch]


def sequential_hash(paths):
    """Reference single-process implementation for correctness testing."""
    hasher = FINDNumpyHasher()
    return [(p, str(hasher.fromFile(p))) for p in paths]

def parallel_hash_imap(paths, workers=8, chunksize=None):
    """
    Streaming dispatch via Pool.imap_unordered.
    
    Returns dict {path: hash_str}. Order of completion is non-deterministic;
    we return dict to make this explicit (vs. parallel_hash which returns
    list[(path, hash)]).
    
    Note on dispatch choice: under cool-down benchmark protocol on the LoC 
    dataset (14,687 same-size 250x250 images, compute-bound FIND hashing), 
    parallel_hash (pool.map) outperforms parallel_hash_imap by ~7% at 8 
    workers (1397 vs 1298 img/s). Pool.map's chunk-allocation amortises 
    IPC overhead better for compute-bound homogeneous workloads. 
    parallel_hash is recommended as default. parallel_hash_imap is retained 
    for ablation study and for workloads with high per-image variance 
    (mixed image sizes, I/O-heavy preprocessing) where dynamic streaming 
    dispatch can outperform static chunking.
    """
    if chunksize is None:
        chunksize = max(1, len(paths) // (workers * 4))
    
    results = {}
    with Pool(processes=workers, initializer=_init_worker) as pool:
        for path, hash_str in pool.imap_unordered(_hash_one, paths, chunksize=chunksize):
            results[path] = hash_str
    return results
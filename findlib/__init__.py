"""
findlib — image hashing library based on the FIND algorithm.

Provides:
- FINDNumpyHasher: optimised image hasher (NumPy/scipy vectorised)
- parallel_hash: multiprocessing wrapper for batch hashing
- sequential_hash: single-process wrapper for small batches
"""

from findlib.hasher import FINDNumpyHasher
from findlib.parallel import parallel_hash, sequential_hash

__version__ = "0.1.0"
__all__ = [
    "FINDNumpyHasher",
    "parallel_hash",
    "sequential_hash",
]
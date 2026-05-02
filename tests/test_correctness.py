"""
test_correctness.py — algorithm correctness and execution consistency tests.

Two validation levels:
  1. Algorithm correctness (L1): FINDNumpyHasher == FINDHasher (bit-exact)
  2. Execution consistency (L3): parallel_hash == sequential_hash

Test fixtures (tests/fixtures/sample_01..05.jpg) are 5 images randomly sampled
from the LoC meme_images working corpus (subset_part2.txt, n=14,687) using
random.seed(42) for reproducibility. The same dataset and sampling protocol
are used throughout Parts 1, 2, and 3.

Run:
    python -m pytest tests/test_correctness.py -v
"""

import unittest
from pathlib import Path

from findlib.find_original import FINDHasher
from findlib.hasher import FINDNumpyHasher
from findlib.parallel import parallel_hash, sequential_hash


# Fixture path resolution: tests/fixtures/ relative to this test file
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def get_fixture_paths() -> list:
    """Return sorted list of sample_*.jpg paths in tests/fixtures/."""
    paths = sorted(FIXTURES_DIR.glob("sample_*.jpg"))
    if not paths:
        raise FileNotFoundError(
            f"No sample_*.jpg fixtures found in {FIXTURES_DIR}. "
            f"Run the fixture sampling script (see README) to generate them."
        )
    return [str(p) for p in paths]


class TestL1AlgorithmCorrectness(unittest.TestCase):
    """L1: FINDNumpyHasher must produce bit-exact same output as FINDHasher."""

    @classmethod
    def setUpClass(cls):
        cls.original = FINDHasher()
        cls.optimised = FINDNumpyHasher()
        cls.paths = get_fixture_paths()

    def test_l1_bit_exact_vs_original(self):
        """Optimised NumPy hash == original Python-loop hash on all fixtures."""
        for p in self.paths:
            h_orig = str(self.original.fromFile(p))
            h_opt = str(self.optimised.fromFile(p))
            self.assertEqual(
                h_orig, h_opt,
                f"L1 bit-exact mismatch on {p}:\n"
                f"  original={h_orig}\n  optimised={h_opt}"
            )


class TestL3ExecutionConsistency(unittest.TestCase):
    """L3: parallel execution must produce same output as sequential."""

    @classmethod
    def setUpClass(cls):
        cls.paths = get_fixture_paths()

    def test_l3_parallel_matches_sequential(self):
        """parallel_hash output == sequential_hash output on all fixtures."""
        seq = dict(sequential_hash(self.paths))
        par = dict(parallel_hash(self.paths, workers=2))
        for p in self.paths:
            self.assertEqual(
                seq[p], par[p],
                f"L3 parallel vs sequential mismatch on {p}:\n"
                f"  seq={seq[p]}\n  par={par[p]}"
            )


if __name__ == "__main__":
    unittest.main()
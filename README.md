# findlib

A Python library for image hashing based on the FIND algorithm, with a NumPy/scipy-vectorised hasher, multiprocessing support for batch hashing, and a FastAPI server exposing a /compare endpoint over HTTP.

## Installation

### From source (Python >= 3.13)

    git clone <this-repo>
    cd findlib
    pip install -r requirements.txt

### Via Docker

    docker build -t findlib .
    docker run -p 8945:8945 findlib

## Usage

### Python API

    from findlib import FINDNumpyHasher, parallel_hash, sequential_hash

    hasher = FINDNumpyHasher()
    hash1 = hasher.fromFile("image.jpg")

    hashes = sequential_hash(["a.jpg", "b.jpg", "c.jpg"])
    hashes = parallel_hash(["a.jpg", "b.jpg"], workers=8)

### REST API

The FastAPI server exposes one endpoint, POST /compare, accepting two image files via multipart form upload and returning their FIND hashes plus Hamming distance.

    curl -X POST "http://127.0.0.1:8945/compare" \
      -F "image1=@a.jpg" \
      -F "image2=@b.jpg"

Response:

    {
      "image1_hash": "20f7881c9e863231...",
      "image2_hash": "9a4e135b6c9e6133...",
      "distance": 134
    }

Interactive API docs are available at http://127.0.0.1:8945/docs (Swagger UI).

## Architecture

**Note on optimisation labelling.** findlib references "L1" and "L3" optimisations rather than "L1" and "L2". This reflects the project's original three-tier optimisation framework: L1 (intra-image vectorisation), L2 (single-image multithreading), and L3 (cross-image multiprocessing). The intermediate L2 candidate was rejected at the candidate-evaluation stage after profiling identified Python's GIL as a hard blocker for single-image CPU-bound workloads (see report Part 1 §1.2). The L1/L3 labels are preserved across docstrings, this README, and the report for narrative consistency.

findlib provides two layers of optimisation over the original FIND research code:

- L1 (intra-image vectorisation): replaces Python loops with NumPy/scipy C-level operations (scipy.ndimage.uniform_filter, np.dot, np.partition). Achieves ~57x per-image speedup with bit-exact output preservation.
- L3 (cross-image multiprocessing): wraps L1 in a multiprocessing.Pool for batch hashing. Achieves ~3.4x over L1 sequential at 8 workers on M1 4P+4E hardware. The cross-over to net-positive throughput is at N approximately 300-500 images; for smaller batches, sequential L1 is faster.

The /compare endpoint uses sequential L1 (per-request batch size N=2 is far below the multiprocessing cross-over). Future /batch_compare endpoints handling many images per request would benefit from L3.

## Repository Structure

    findlib_release/
    findlib/                    Main package
        __init__.py             Public API exports
        api.py                  FastAPI server
        hasher.py               L1 NumPy/scipy hasher (FINDNumpyHasher)
        parallel.py             L3 multiprocessing wrappers
        matrix.py               Matrix utility helpers
        find_original.py        Original FIND reference (for tests)
    tests/
        test_correctness.py     L1 bit-exact + L3 concurrency consistency
        test_api.py             FastAPI endpoint integration tests
        fixtures/               20 reference images for bit-exact unit tests
    Dockerfile                  Single-stage Python 3.13-slim container
    pyproject.toml              PEP 621 project metadata (name, deps, classifiers)
    requirements.txt            Pinned runtime dependencies (mirror of pyproject.toml)
    README.md
    LICENSE                     MIT license

## Testing

    python -m pytest tests/ -v

Six tests cover:

- Algorithm bit-exactness (FINDNumpyHasher vs FINDHasher)
- Concurrency consistency (parallel_hash vs sequential_hash)
- API root endpoint metadata
- API /compare for identical, distinct, and hex-format inputs

### Test Fixtures

tests/fixtures/ contains 20 reference JPEG images (subset_golden.txt) — the same fixture used in Part 1 §1.2.5 bit-exact verification from the LoC meme_images dataset. The same dataset and sampling protocol are used throughout the project's evaluation.

## Known Issues and Future Improvements

- Original boxFilter stride and window-axis bugs (silent on square inputs): the upstream FIND code uses inconsistent row-major stride conventions inside boxFilter (k * rows + l vs k * numCols + l elsewhere) and computes windowSizeAlongRows from numCols. Both bugs produce incorrect output on non-square images but are silent on square inputs (99.998% of the test corpus). The L1 substitution to scipy.ndimage.uniform_filter resolves both as a side effect.
- np.partition semantic fragility: findlib.hasher uses np.partition(arr, 127)[127] rather than np.median because FIND Torben median returns the lower of two middle values for even-length arrays, while np.median returns their mean - silently flipping 1-2 hash bits at threshold-boundary cases. Future contributors must preserve this lower-median semantics.
- Pillow img.thumbnail resample non-lock (in find_original.py only, not L1): cross-Pillow-version reproducibility depends on Pillow evolving default resample filter.
- Container security hardening: the production Dockerfile runs as root. A future improvement is to add a non-root app user (useradd app && USER app) following standard hardening practice.

## License

MIT (see LICENSE).

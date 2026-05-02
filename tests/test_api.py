"""
test_api.py — integration tests for the /compare FastAPI endpoint.

Uses FastAPI TestClient (in-process; no live server needed) and the same
fixture images as test_correctness.py (tests/fixtures/sample_*.jpg) for
narrative consistency across the test suite.

Pattern reference: inpractice_lab_w06_solution.ipynb (Week 6).

Run:
    python -m pytest tests/test_api.py -v
"""

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from findlib.api import app


# Same fixture path as test_correctness.py — reuse 20 LoC sample images
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def get_fixture_paths() -> list:
    """Return sorted list of sample_*.jpg paths in tests/fixtures/."""
    paths = sorted(FIXTURES_DIR.glob("sample_*.jpg"))
    if not paths:
        raise FileNotFoundError(
            f"No sample_*.jpg fixtures found in {FIXTURES_DIR}. "
            f"See README.md for fixture regeneration instructions."
        )
    return paths


class TestRootEndpoint(unittest.TestCase):
    """Tests the / health-check endpoint."""

    def setUp(self):
        self.client = TestClient(app)

    def test_root_returns_service_info(self):
        """GET / should return service metadata."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["service"], "findlib")
        self.assertIn("/compare", data["endpoints"])


class TestCompareEndpoint(unittest.TestCase):
    """Tests the /compare endpoint with multipart file uploads using LoC fixtures."""

    def setUp(self):
        self.client = TestClient(app)
        self.fixtures = get_fixture_paths()

    def test_compare_identical_images_distance_zero(self):
        """Same LoC image hashed twice should produce distance 0 (deterministic)."""
        img_path = self.fixtures[0]
        with open(img_path, "rb") as f1, open(img_path, "rb") as f2:
            response = self.client.post(
                "/compare",
                files={
                    "image1": (img_path.name, f1, "image/jpeg"),
                    "image2": (img_path.name, f2, "image/jpeg"),
                },
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("image1_hash", data)
        self.assertIn("image2_hash", data)
        self.assertIn("distance", data)
        self.assertEqual(data["image1_hash"], data["image2_hash"])
        self.assertEqual(data["distance"], 0)

    def test_compare_different_images_returns_valid_distance(self):
        """Two distinct LoC images should produce a valid Hamming distance in [0, 256]."""
        img1, img2 = self.fixtures[0], self.fixtures[1]
        with open(img1, "rb") as f1, open(img2, "rb") as f2:
            response = self.client.post(
                "/compare",
                files={
                    "image1": (img1.name, f1, "image/jpeg"),
                    "image2": (img2.name, f2, "image/jpeg"),
                },
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data["distance"], int)
        self.assertGreaterEqual(data["distance"], 0)
        self.assertLessEqual(data["distance"], 256)
        # Distinct LoC images from different families should produce distinct hashes
        self.assertNotEqual(data["image1_hash"], data["image2_hash"])

    def test_compare_hash_is_64_char_hex(self):
        """FIND outputs 256-bit hashes encoded as 64-character hex strings."""
        img_path = self.fixtures[0]
        with open(img_path, "rb") as f1, open(img_path, "rb") as f2:
            response = self.client.post(
                "/compare",
                files={
                    "image1": (img_path.name, f1, "image/jpeg"),
                    "image2": (img_path.name, f2, "image/jpeg"),
                },
            )
        data = response.json()
        for key in ("image1_hash", "image2_hash"):
            self.assertEqual(len(data[key]), 64,
                             f"{key} should be 64 hex chars, got {len(data[key])}")
            int(data[key], 16)  # raises ValueError if not valid hex


if __name__ == "__main__":
    unittest.main()

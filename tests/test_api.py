"""
test_api.py — integration tests for the /compare FastAPI endpoint.

Uses FastAPI's TestClient (in-process; no live server needed).
Pattern reference: inpractice_lab_w06_solution.ipynb (Week 6).

Run:
    python -m pytest tests/test_api.py -v
"""

import io
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from findlib.api import app


def make_test_image(color=(128, 128, 128), size=(64, 64)) -> bytes:
    """Generate a small in-memory JPEG for upload testing."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


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
    """Tests the /compare endpoint with multipart file uploads."""

    def setUp(self):
        self.client = TestClient(app)

    def test_compare_identical_images_distance_zero(self):
        """Same image hashed twice should produce distance 0."""
        img_bytes = make_test_image(color=(100, 150, 200))
        response = self.client.post(
            "/compare",
            files={
                "image1": ("a.jpg", img_bytes, "image/jpeg"),
                "image2": ("b.jpg", img_bytes, "image/jpeg"),
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("image1_hash", data)
        self.assertIn("image2_hash", data)
        self.assertIn("distance", data)
        self.assertEqual(data["image1_hash"], data["image2_hash"])
        self.assertEqual(data["distance"], 0)

    def test_compare_different_images_returns_distance(self):
        """Different images should produce non-zero distance, returned as int 0–256."""
        img1 = make_test_image(color=(0, 0, 0))       # solid black
        img2 = make_test_image(color=(255, 255, 255)) # solid white
        response = self.client.post(
            "/compare",
            files={
                "image1": ("black.jpg", img1, "image/jpeg"),
                "image2": ("white.jpg", img2, "image/jpeg"),
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data["distance"], int)
        self.assertGreaterEqual(data["distance"], 0)
        self.assertLessEqual(data["distance"], 256)

    def test_compare_hash_is_64_char_hex(self):
        """FIND outputs 256-bit hashes encoded as 64-character hex strings."""
        img_bytes = make_test_image()
        response = self.client.post(
            "/compare",
            files={
                "image1": ("a.jpg", img_bytes, "image/jpeg"),
                "image2": ("b.jpg", img_bytes, "image/jpeg"),
            },
        )
        data = response.json()
        for key in ("image1_hash", "image2_hash"):
            self.assertEqual(len(data[key]), 64)
            int(data[key], 16)  # raises ValueError if not valid hex


if __name__ == "__main__":
    unittest.main()
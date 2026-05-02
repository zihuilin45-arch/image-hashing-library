"""
findlib.api — FastAPI server exposing the /compare endpoint.

Usage:
    uvicorn findlib.api:app --host 0.0.0.0 --port 8945

Endpoint:
    POST /compare
        Form fields: image1 (file), image2 (file)
        Returns: {"image1_hash": str, "image2_hash": str, "distance": int}
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from findlib import FINDNumpyHasher

app = FastAPI(
    title="findlib",
    description="Image hashing library based on the FIND algorithm.",
    version="0.1.0",
)

# Module-level hasher: instantiated once at startup, reused across requests.
# FINDNumpyHasher holds DCT matrix and other constants — no per-request state.
hasher = FINDNumpyHasher()


@app.get("/")
def root():
    """Health check / API discovery."""
    return {
        "service": "findlib",
        "version": "0.1.0",
        "endpoints": ["/compare"],
    }


@app.post("/compare")
async def compare(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    """
    Hash two uploaded images using FIND and return their Hamming distance.

    Args:
        image1: first image (JPEG/PNG/etc — anything PIL can decode)
        image2: second image

    Returns:
        JSON with image1_hash, image2_hash (hex strings), and distance (int).
    """
    try:
        # Read uploaded bytes into PIL Image objects
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()

        img1 = Image.open(io.BytesIO(img1_bytes))
        img2 = Image.open(io.BytesIO(img2_bytes))

        # Hash each image
        hash1 = hasher.fromImage(img1)
        hash2 = hasher.fromImage(img2)

        # Hamming distance via imagehash's __sub__ operator
        distance = hash1 - hash2

        return {
            "image1_hash": str(hash1),
            "image2_hash": str(hash2),
            "distance": int(distance),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process images: {str(e)}",
        )
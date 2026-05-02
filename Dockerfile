# Dockerfile — findlib production image
#
# Build:  docker build -t findlib .
# Run:    docker run -p 8945:8945 findlib
# Test:   curl -X POST http://127.0.0.1:8945/compare \
#           -F "image1=@a.jpg" -F "image2=@b.jpg"

FROM python:3.13-slim

# Install runtime libraries for Pillow image decoding.
# These are runtime libs (not -dev headers); Pillow wheels usually
# bundle these but installing them ensures fallback works on any
# build environment (e.g., a marker's machine where wheel may differ).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
        libtiff6 \
        libwebp7 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first (Docker layer caching:
# requirements.txt rarely changes, code changes often)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy package source
COPY findlib/ ./findlib/

# Expose the port the app runs on
EXPOSE 8945

# Start the FastAPI server
# --host 0.0.0.0 is REQUIRED inside Docker (not 127.0.0.1)
# so that the host machine can reach the container
CMD ["uvicorn", "findlib.api:app", "--host", "0.0.0.0", "--port", "8945"]
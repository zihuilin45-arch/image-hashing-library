"""
FINd_numpy.py — L1 language-level optimisation of FINDHasher.

Strategy: Override pipeline methods of FINDHasher with NumPy equivalents.
Bit-exact against golden_hashes.txt on square inputs (99.998% of LoC).

Verified gotchas (see part1a_locked.md):
  G1 Torben       np.partition(flat, 127)[127], NOT np.median
  G2 stride bug   scipy handles correctly; NumPy bit-exact on square only
  G3 DCT scale    D @ A @ D.T with un-normalised D
  G4 bit reverse  hash_matrix[::-1, ::-1] after thresholding
  G5 window axis  preserve FIND's reversed computation (silent on square)
  G6 asym window  effective size = 2 * ((win+2)//2), NOT win

Sub-steps (L1.1 - L1.6, all within this class):
  L1.1 fillLuma    NumPy asarray @ coeffs
  L1.2 boxFilter   scipy.ndimage.uniform_filter
  L1.3 DCT         D @ A @ D.T
  L1.4 decimate    fancy indexing
  L1.5 Torben      np.partition
  L1.6 bit pack    NumPy operations
"""
from __future__ import annotations
import numpy as np
from PIL import Image
from findlib.find_original import FINDHasher
from imagehash import ImageHash


class FINDNumpyHasher(FINDHasher):
    """
    NumPy-based hasher. Inherits constants and DCT matrix from FINDHasher.
    Overrides pipeline methods one at a time; unoverridden methods delegate
    to the parent class.
    """

    # ------------------------------------------------------------------
    # L1.5 — Torben median → np.partition (gotcha G1)
    # ------------------------------------------------------------------
    def dctOutput2hash(self, dct_output):
        """
        Replaces FINd's dctOutput2hash (FINd.py line 147-160):
          - Torben median over 256 values  (matrix.py, returns x[127])
          - Thresholding: bit = 1 if value > median
          - Bit reversal: hash[15-i, 15-j] = bit

        NumPy equivalent:
          - np.partition(flat, 127)[127]           (gotcha G1)
          - (dct > threshold)
          - hash_matrix[::-1, ::-1]                (gotcha G4)
        """
        dct = np.array(dct_output, dtype=np.float64)
        threshold = np.partition(dct.ravel(), 127)[127]   # G1
        bits = (dct > threshold).astype(np.uint8)
        hash_matrix = bits[::-1, ::-1]                    # G4
        return ImageHash(hash_matrix.astype(bool))
    # ------------------------------------------------------------------
    # L1.1 — Preprocessing: RGB → luminance (replaces fillFloatLuma)
    # ------------------------------------------------------------------
    def fillFloatLumaFromBufferImage(self, img, luma):
        """
        Replaces FINd's per-pixel Python loop (FINd.py line 66-77):
          for i in range(numRows):
              for j in range(numCols):
                  r, g, b = rgb_image.getpixel((j, i))
                  luma[i*numCols + j] = R_coeff*r + G_coeff*g + B_coeff*b

        NumPy version:
          - np.asarray loads entire RGB buffer in one PIL → C call
          - Matrix-vector product applies Rec.601 luma coefficients
          - Eliminates 62,500 getpixel() calls per 250x250 image (~180 ms
            of Python/C boundary cost per Day 1 line_profiler data)

        The caller passes a Python list `luma` of length numRows*numCols
        that must be written in place (FINd's imperative interface).
        """
        numCols, numRows = img.size
        rgb = np.asarray(img.convert("RGB"), dtype=np.float64)  # (H, W, 3)
        coeffs = np.array(
            [self.LUMA_FROM_R_COEFF, self.LUMA_FROM_G_COEFF, self.LUMA_FROM_B_COEFF],
            dtype=np.float64,
        )
        luma_2d = rgb @ coeffs                                  # (H, W)
        # Write back into caller's Python list via slice assignment.
        # .tolist() is a single C call, vastly cheaper than per-element
        # Python-level assignment (~62,500 iterations eliminated).
        luma[:] = luma_2d.ravel().tolist()
        return luma_2d
        
    # ------------------------------------------------------------------
    # L1.3 — DCT: replace triple-nested loops with D @ A @ D.T (gotcha G3)
    # ------------------------------------------------------------------
    def dct64To16(self, A, T, B):
        """
        Replaces FINd's two sequential triple-nested loops (FINd.py line 110-130):
          T = D @ A    -> 16 x 64 x 64 = 65536 Python mul-adds
          B = T @ Dt   -> 16 x 64 x 16 = 16384 Python mul-adds

        NumPy version: two matrix multiplications dispatched to BLAS.

        Gotcha G3: self.DCTMatrix (from parent FINDHasher.__init__) is
        an UN-NORMALISED DCT matrix — no sqrt(2/64) factor applied (FINd.py
        line 24 is dead code). We preserve this convention. Do NOT use
        scipy.fft.dct here (it would apply its own normalisation and
        break bit-exact parity).
        """
        # Cache ndarray view of parent's DCT matrix on first call.
        # self._D_cache lives on the instance so it's reused across calls.
        if not hasattr(self, "_D_cache"):
            self._D_cache = np.array(self.DCT_matrix, dtype=np.float64)  # (16, 64)
        D = self._D_cache

        A_arr = np.array(A, dtype=np.float64)     # (64, 64)
        T_arr = D @ A_arr                          # (16, 64)
        B_arr = T_arr @ D.T                        # (16, 16)

        # Write back via per-row slice assignment (.tolist() is a single
        # C call per row; 32 C calls replace 1,280 Python iterations).
        for i in range(16):
            T[i][:] = T_arr[i].tolist()
        for i in range(16):
            B[i][:] = B_arr[i].tolist()

    # ------------------------------------------------------------------
    # L1.4 — Decimation: 64x64 downsample via NumPy fancy indexing
    # ------------------------------------------------------------------
    def decimateFloat(self, in_, inRows, inCols, out):
        """
        Replaces FINd's nested loop (FINd.py line 100-108):
          for i in range(64):
              ini = int(((i + 0.5) * inRows) / 64)
              for j in range(64):
                  inj = int(((j + 0.5) * inCols) / 64)
                  out[i][j] = in_[ini * inCols + inj]

        NumPy version: vectorised row/col index arrays + fancy indexing.
        Input `in_` is a row-major 1D Python list of length inRows*inCols.
        Output `out` is a Python 64x64 list-of-lists, written in place.

        Correctness note: the `int(...)` in FINd truncates toward zero
        for non-negative inputs. np.astype(int) on positive floats
        matches this behaviour (truncation = floor for positives).
        """
        # Compute the 64 row and column indices into the source buffer
        i_range = np.arange(64)
        ini = ((i_range + 0.5) * inRows / 64).astype(int)   # (64,)
        inj = ((i_range + 0.5) * inCols / 64).astype(int)   # (64,)

        # Reshape 1D input into 2D for direct 2D indexing
        in_2d = np.asarray(in_, dtype=np.float64).reshape(inRows, inCols)

        # Fancy indexing: in_2d[row_idx[:, None], col_idx[None, :]] → (64, 64)
        # Broadcasting: (64, 1) with (1, 64) → (64, 64)
        out_arr = in_2d[ini[:, None], inj[None, :]]

        # Write back via per-row slice assignment (64 C calls replace
        # 4,096 Python iterations).
        for i in range(64):
            out[i][:] = out_arr[i].tolist()
    # ------------------------------------------------------------------
    # L1.2 — Box filter: scipy.ndimage.uniform_filter
    # ------------------------------------------------------------------
    # This is the biggest single bottleneck (81.9% of pipeline time per
    # §1.2 pipeline attribution). It also concentrates the most gotchas:
    #   G2 stride bug     — scipy fixes automatically (bit-exact on square only)
    #   G5 window axis    — preserve caller's reversed args
    #   G6 asymmetric win — effective size = 2 * ((win+2)//2), NOT win
    #   boundary mode     — FINd truncates + renormalises; scipy does not by
    #                       default, so we emulate via constant-pad + count mask
    @classmethod
    def boxFilter(cls, input, output, rows, cols, rowWin, colWin):
        """
        Replaces FINd's O(h*w*wr*wc) nested-loop box filter.

        FINd's smoothing behaviour (FINd.py line 167-181):
          - window covers range(i-halfWin, i+halfWin) = 2*halfWin elements
            where halfWin = (win+2)//2  (so win=4 → window=6, gotcha G6)
          - at boundaries, window is truncated and divided by actual count
            (not a standard zero-pad or reflect mode)
          - uses input[k*rows+l] instead of k*cols+l (stride bug G2,
            silent on square inputs: rows == cols for 99.998% of LoC)

        NumPy/scipy version:
          - Convert input to 2D ndarray (reshape is zero-copy view)
          - Use uniform_filter with size matching G6's effective window
          - Emulate FINd's truncate-and-renormalise by computing both the
            zero-padded sum and a matching count mask, then dividing
        """
        from scipy.ndimage import uniform_filter

        # Effective window sizes (gotcha G6)
        eff_rows = 2 * ((rowWin + 2) // 2)
        eff_cols = 2 * ((colWin + 2) // 2)

        # Reshape input 1D list to 2D
        in_2d = np.asarray(input, dtype=np.float64).reshape(rows, cols)

        # Sum within the window (mode='constant', cval=0) gives zero-padded mean.
        # uniform_filter returns the MEAN, not the sum, so multiply by window area
        # to get the sum; then we'll divide by the actual (truncated) count.
        mean_padded = uniform_filter(
            in_2d, size=(eff_rows, eff_cols),
            mode='constant', cval=0.0
        )
        sum_padded = mean_padded * (eff_rows * eff_cols)

        # Count of real (non-padded) pixels in each window.
        # Computed by running the same filter on a ones-array: the resulting
        # mean * window_size = actual pixel count for that position.
        ones = np.ones_like(in_2d)
        count_mean = uniform_filter(
            ones, size=(eff_rows, eff_cols),
            mode='constant', cval=0.0
        )
        counts = count_mean * (eff_rows * eff_cols)

        # FINd's output: sum / actual_count
        out_arr = sum_padded / counts

        # Write back via single C-level slice assignment
        # (eliminates ~62,500 Python-level iterations).
        output[:] = out_arr.ravel().tolist()

        # ==================================================================
    # L1 Iteration 2: override findHash256FromFloatLuma to let ndarrays
    # flow directly between pipeline stages (no list ↔ ndarray round-trips).
    # ==================================================================
    def findHash256FromFloatLuma(
        self,
        fullBuffer1,
        fullBuffer2,
        numRows,
        numCols,
        buffer64x64,
        buffer16x64,
        buffer16x16,
    ):
        """
        L1 iteration-2 pipeline: ndarrays flow directly between stages,
        bypassing the list-buffer contract of the parent class.

        Per cProfile (iter 1): the parent's list-based pipeline forced
        ~13 np.asarray calls + ~98 .tolist() calls per image, costing
        ~3.8 ms. This override eliminates all of them by keeping the
        working buffer as a 2D ndarray throughout boxFilter → decimate
        → DCT → quantise.

        The list parameters (fullBuffer1, buffer64x64 etc.) are still
        accepted for signature compatibility but not materially used.

        Preserves gotchas G5 (window axes reversed) and G6 (asymmetric
        effective window) by calling the parent's
        computeBoxFilterWindowSize unchanged.
        """
        # Same window-size computation as parent (preserves G5 behaviour)
        windowSizeAlongRows = self.computeBoxFilterWindowSize(numCols)
        windowSizeAlongCols = self.computeBoxFilterWindowSize(numRows)

        # Accept either ndarray (fast path from iter 3 fromImage) or list (compatibility path).
        # When called from our override fromImage, fullBuffer1 is already a (numRows, numCols)
        # ndarray — no conversion needed. When called from the parent's fromImage (e.g., in
        # tests that use FINDHasher's pipeline), fullBuffer1 is a flat Python list.
        if isinstance(fullBuffer1, np.ndarray):
            luma_2d = fullBuffer1
        else:
            luma_2d = np.asarray(fullBuffer1, dtype=np.float64).reshape(numRows, numCols)

        # ---- L1.2 box filter (ndarray in, ndarray out) ----
        from scipy.ndimage import uniform_filter
        eff_rows = 2 * ((windowSizeAlongRows + 2) // 2)  # G6
        eff_cols = 2 * ((windowSizeAlongCols + 2) // 2)

        # Sum within the window (mode='constant' for zero-pad)
        mean_padded = uniform_filter(
            luma_2d, size=(eff_rows, eff_cols),
            mode='constant', cval=0.0,
        )
        sum_padded = mean_padded * (eff_rows * eff_cols)

        # Count of real (non-padded) pixels — renormalises boundary (FINd behaviour)
        # Cache this per (rows, cols, eff_rows, eff_cols) signature
        if not hasattr(self, "_count_cache"):
            self._count_cache = {}
        cache_key = (numRows, numCols, eff_rows, eff_cols)
        if cache_key not in self._count_cache:
            ones = np.ones((numRows, numCols))
            count_mean = uniform_filter(
                ones, size=(eff_rows, eff_cols),
                mode='constant', cval=0.0,
            )
            self._count_cache[cache_key] = count_mean * (eff_rows * eff_cols)
        counts = self._count_cache[cache_key]

        smoothed_2d = sum_padded / counts  # ← box-filtered luma, still ndarray

        # ---- L1.4 decimate to 64×64 (fancy indexing) ----
        i_range = np.arange(64)
        ini = ((i_range + 0.5) * numRows / 64).astype(int)
        inj = ((i_range + 0.5) * numCols / 64).astype(int)
        buf64 = smoothed_2d[ini[:, None], inj[None, :]]  # (64, 64) ndarray

        # ---- L1.3 DCT 64→16 (D @ A @ D.T) ----
        if not hasattr(self, "_D_cache"):
            self._D_cache = np.array(self.DCT_matrix, dtype=np.float64)
        D = self._D_cache
        buf16 = D @ buf64 @ D.T  # (16, 16) ndarray

        # ---- L1.5 Torben-equivalent median + threshold + axis reversal ----
        threshold = np.partition(buf16.ravel(), 127)[127]  # G1
        bits = (buf16 > threshold).astype(np.uint8)
        hash_matrix = bits[::-1, ::-1]  # G4

        return ImageHash(hash_matrix.astype(bool))
    
    # ==================================================================
    # L1 Iteration 3: override fromImage to bypass father's Python-list
    # buffer allocation and img.copy(). Makes luma flow as ndarray from
    # fillLuma directly into findHash256FromFloatLuma, with no list
    # allocations and no defensive Image copy.
    # ==================================================================
    def fromImage(self, img):
        """
        L1 iteration-3 image-level pipeline override.

        Parent's fromImage (FINd.py line 47-64) does:
          1. img = img.copy()                         -> 0.48 ms (defensive Pillow copy)
          2. img.thumbnail((512, 512))                -> no-op for <=512px inputs
          3. allocate buffer1 (flat list, numRows*numCols floats) -> ~0.2 ms
          4. allocate buffer2 (same, unused by iter 2)            -> ~0.2 ms (WASTE)
          5. allocate buffer64x64, buffer16x64, buffer16x16 (unused) -> ~0.1 ms (WASTE)
          6. fillFloatLumaFromBufferImage(img, buffer1)
          7. findHash256FromFloatLuma(buffer1, buffer2, ...)

        Iteration 3 eliminates:
          - step 1 (copy): safe because fromFile passes a fresh Image from
            Image.open() that isn't referenced externally after return.
          - steps 3-5 except buffer1: our findHash256FromFloatLuma accepts
            an ndarray directly; intermediate buffers are allocated as
            ndarrays inside each stage (not Python lists) on demand.

        Safety note: if a user calls `fromImage(cached_img)` directly with
        an Image they still hold a reference to, thumbnail() mutates that
        Image in place. This contract must be documented in the library API.
        """
        img.thumbnail((512, 512))   # modifies in place; safe inside fromFile
        numCols, numRows = img.size

        # Step 6: compute luma as ndarray (uses our override L1.1; ignores luma_list arg)
        luma_list = [0.0] * (numRows * numCols)  # kept for compat signature
        luma_2d = self.fillFloatLumaFromBufferImage(img, luma_list)

        # Step 7: pass ndarray directly (our override accepts it)
        # All trailing buffers are legacy placeholders; findHash256FromFloatLuma doesn't use them.
        return self.findHash256FromFloatLuma(
            luma_2d, None, numRows, numCols, None, None, None
        )
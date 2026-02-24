"""
FP4 GEMM for DGX Spark GB10 (sm_121)

Python wrapper around CUTLASS-based FP4 GEMM shared library.
Uses hardware FP4 tensor cores (mma.sync.aligned.block_scale)
for up to 280 TFLOPS on Blackwell GeForce / GB10.

Usage:
    from fp4_gemm import fp4_matmul

    # A @ B^T → C  (like F.linear)
    C = fp4_matmul(A, B)  # A: [M, K], B: [N, K] → C: [M, N]

    # With alpha/beta: D = alpha * A_fp4 @ B_fp4^T + beta * C
    D = fp4_matmul(A, B, C=C, alpha=1.0, beta=1.0)

Note: M, N must be multiples of 128. K must be a multiple of 128.
      Both inputs are BF16 — quantization to FP4 happens inside the kernel.
"""

import os
import ctypes
import torch

# Load the shared library
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfp4gemm.so")
_lib = ctypes.CDLL(_lib_path)

# C API signatures
_lib.fp4_gemm_sf_vec_size.restype = ctypes.c_int
_lib.fp4_gemm_sf_vec_size.argtypes = []

_lib.fp4_gemm_init.restype = ctypes.c_int
_lib.fp4_gemm_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.fp4_gemm_run.restype = ctypes.c_int
_lib.fp4_gemm_run.argtypes = [
    ctypes.c_void_p,  # A_bf16
    ctypes.c_void_p,  # B_bf16
    ctypes.c_void_p,  # C_bf16
    ctypes.c_void_p,  # D_bf16
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int,     # K
    ctypes.c_float,   # alpha
    ctypes.c_float,   # beta
]

_lib.fp4_gemm_cleanup.restype = None
_lib.fp4_gemm_cleanup.argtypes = []

# Pre-quantized weight cache API
_lib.fp4_quantize_weights.restype = ctypes.c_void_p
_lib.fp4_quantize_weights.argtypes = [
    ctypes.c_void_p,  # weight_bf16
    ctypes.c_int,     # N
    ctypes.c_int,     # K
]

_lib.fp4_weight_cache_free.restype = None
_lib.fp4_weight_cache_free.argtypes = [ctypes.c_void_p]

_lib.fp4_weight_cache_N.restype = ctypes.c_int
_lib.fp4_weight_cache_N.argtypes = [ctypes.c_void_p]

_lib.fp4_weight_cache_K.restype = ctypes.c_int
_lib.fp4_weight_cache_K.argtypes = [ctypes.c_void_p]

_lib.fp4_gemm_run_cached.restype = ctypes.c_int
_lib.fp4_gemm_run_cached.argtypes = [
    ctypes.c_void_p,  # A_bf16
    ctypes.c_void_p,  # cache_handle
    ctypes.c_void_p,  # C_bf16
    ctypes.c_void_p,  # D_bf16
    ctypes.c_int,     # M
    ctypes.c_float,   # alpha
    ctypes.c_float,   # beta
]

# Query scale factor vector size
SF_VEC_SIZE = _lib.fp4_gemm_sf_vec_size()


def _pad_to_multiple(x: int, m: int) -> int:
    """Round up x to the nearest multiple of m."""
    return ((x + m - 1) // m) * m


def fp4_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor:
    """
    FP4 matrix multiply using hardware tensor cores on sm_121.

    Computes: D = alpha * quantize_fp4(A) @ quantize_fp4(B)^T + beta * C

    This is equivalent to F.linear(A, B) when alpha=1, beta=0.

    Args:
        A: Input activation tensor [M, K] or [batch, M, K], BF16 on CUDA
        B: Weight tensor [N, K], BF16 on CUDA
        C: Optional bias/residual tensor [M, N], BF16 on CUDA
        alpha: Scalar multiplier for GEMM result
        beta: Scalar multiplier for C

    Returns:
        D: Output tensor [M, N] or [batch, M, N], BF16 on CUDA

    Notes:
        - M, N must be multiples of 128. K must be a multiple of 128.
        - If dimensions aren't multiples of 128, inputs are automatically padded.
        - Quantization to FP4 E2M1 with UE4M3 block scales happens on GPU.
        - Peak throughput: ~143 TFLOPS (including quantization overhead).
        - 5-9x faster than BF16 cuBLAS at GPT-OSS dimensions.
    """
    # Handle batched input
    batched = A.dim() == 3
    if batched:
        batch_size = A.shape[0]
        A = A.reshape(-1, A.shape[-1])

    assert A.dim() == 2, f"A must be 2D [M, K] or 3D [batch, M, K], got shape {A.shape}"
    assert B.dim() == 2, f"B must be 2D [N, K], got shape {B.shape}"
    assert A.shape[1] == B.shape[1], f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[1]}"
    assert A.is_cuda and B.is_cuda, "Both tensors must be on CUDA"

    M_orig, K_orig = A.shape
    N_orig = B.shape[0]

    # Convert to BF16 if needed
    if A.dtype != torch.bfloat16:
        A = A.to(torch.bfloat16)
    if B.dtype != torch.bfloat16:
        B = B.to(torch.bfloat16)

    # Pad to multiples of 128
    M = _pad_to_multiple(M_orig, 128)
    N = _pad_to_multiple(N_orig, 128)
    K = _pad_to_multiple(K_orig, 128)

    padded = (M != M_orig or N != N_orig or K != K_orig)

    if padded:
        A_padded = torch.zeros(M, K, dtype=torch.bfloat16, device=A.device)
        A_padded[:M_orig, :K_orig] = A
        A = A_padded

        B_padded = torch.zeros(N, K, dtype=torch.bfloat16, device=B.device)
        B_padded[:N_orig, :K_orig] = B
        B = B_padded

    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()

    # Allocate output
    D = torch.zeros(M, N, dtype=torch.bfloat16, device=A.device)

    # Handle C tensor
    c_ptr = ctypes.c_void_p(0)
    if C is not None:
        if C.dtype != torch.bfloat16:
            C = C.to(torch.bfloat16)
        if padded:
            C_padded = torch.zeros(M, N, dtype=torch.bfloat16, device=C.device)
            C_padded[:M_orig, :N_orig] = C
            C = C_padded
        C = C.contiguous()
        c_ptr = ctypes.c_void_p(C.data_ptr())

    # Run GEMM
    rc = _lib.fp4_gemm_run(
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        c_ptr,
        ctypes.c_void_p(D.data_ptr()),
        M, N, K,
        alpha, beta,
    )

    if rc != 0:
        raise RuntimeError(f"fp4_gemm_run failed with error code {rc}")

    # Unpad output
    if padded:
        D = D[:M_orig, :N_orig].contiguous()

    # Restore batch dimension
    if batched:
        D = D.reshape(batch_size, -1, N_orig)

    return D


def fp4_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Drop-in replacement for F.linear using FP4 tensor cores.

    Args:
        x: Input tensor [..., K], BF16 on CUDA
        weight: Weight tensor [N, K], BF16 on CUDA
        bias: Optional bias [N], BF16 on CUDA

    Returns:
        Output tensor [..., N]
    """
    # Flatten leading dims
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    out = fp4_matmul(x_2d, weight)

    if bias is not None:
        out = out + bias.unsqueeze(0)

    # Restore shape
    out = out.reshape(*orig_shape[:-1], weight.shape[0])
    return out


def cleanup():
    """Release CUDA memory held by the FP4 GEMM library."""
    _lib.fp4_gemm_cleanup()


# ============================================================================
# Pre-quantized weight cache API
# ============================================================================

class FP4WeightCache:
    """
    Cached FP4 weights for fast inference.

    Quantizes BF16 weights to NVFP4 format once, then reuses the
    pre-quantized data for every GEMM call. Only activations get
    quantized on the fly.

    Usage:
        cache = FP4WeightCache(weight)  # quantize once
        output = cache.forward(x)       # fast inference
        cache.free()                    # release GPU memory

    Or as a context manager:
        with FP4WeightCache(weight) as cache:
            output = cache.forward(x)
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None):
        """
        Quantize a weight matrix to FP4 and cache on device.

        Args:
            weight: [N, K] BF16 weight matrix on CUDA
            bias: Optional [N] bias vector
        """
        assert weight.dim() == 2, f"Weight must be 2D [N, K], got {weight.shape}"
        assert weight.is_cuda, "Weight must be on CUDA"

        self.N_orig = weight.shape[0]
        self.K_orig = weight.shape[1]
        self.bias = bias

        # Pad to multiples of 128
        self.N = _pad_to_multiple(self.N_orig, 128)
        self.K = _pad_to_multiple(self.K_orig, 128)
        self.padded = (self.N != self.N_orig or self.K != self.K_orig)

        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)

        if self.padded:
            w_padded = torch.zeros(self.N, self.K, dtype=torch.bfloat16, device=weight.device)
            w_padded[:self.N_orig, :self.K_orig] = weight
            weight = w_padded

        weight = weight.contiguous()

        # Quantize to FP4 on GPU and store handle
        self._handle = _lib.fp4_quantize_weights(
            ctypes.c_void_p(weight.data_ptr()),
            self.N, self.K)

        if not self._handle:
            raise RuntimeError("fp4_quantize_weights failed")

        # Memory tracking
        self.fp4_bytes = self.N * self.K // 2
        # Scale factors: 1 byte per 16 elements
        self.sf_bytes = self.N * (self.K // 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run FP4 GEMM: output = x @ cached_weight.T + bias

        Args:
            x: Input activation [..., K], BF16 on CUDA

        Returns:
            Output tensor [..., N]
        """
        if self._handle is None:
            raise RuntimeError("FP4WeightCache has been freed")

        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        M_orig = x_2d.shape[0]
        K_in = x_2d.shape[1]

        if K_in != self.K_orig:
            raise ValueError(f"Input K={K_in} doesn't match weight K={self.K_orig}")

        if x_2d.dtype != torch.bfloat16:
            x_2d = x_2d.to(torch.bfloat16)

        # Pad M to multiple of 128
        M = _pad_to_multiple(M_orig, 128)
        m_padded = M != M_orig

        # Pad K if needed
        if self.padded and K_in < self.K:
            x_pad = torch.zeros(M, self.K, dtype=torch.bfloat16, device=x_2d.device)
            x_pad[:M_orig, :K_in] = x_2d
            x_2d = x_pad
        elif m_padded:
            x_pad = torch.zeros(M, self.K, dtype=torch.bfloat16, device=x_2d.device)
            x_pad[:M_orig, :] = x_2d
            x_2d = x_pad

        x_2d = x_2d.contiguous()

        # Allocate output
        D = torch.zeros(M, self.N, dtype=torch.bfloat16, device=x_2d.device)

        # Run cached GEMM (only quantizes A)
        rc = _lib.fp4_gemm_run_cached(
            ctypes.c_void_p(x_2d.data_ptr()),
            ctypes.c_void_p(self._handle),
            ctypes.c_void_p(0),
            ctypes.c_void_p(D.data_ptr()),
            M, 1.0, 0.0)

        if rc != 0:
            raise RuntimeError(f"fp4_gemm_run_cached failed with error code {rc}")

        # Unpad
        if m_padded or self.padded:
            D = D[:M_orig, :self.N_orig].contiguous()

        # Add bias
        if self.bias is not None:
            D = D + self.bias.unsqueeze(0)

        # Restore shape
        D = D.reshape(*orig_shape[:-1], self.N_orig)
        return D

    def free(self):
        """Release GPU memory for cached weights."""
        if self._handle:
            _lib.fp4_weight_cache_free(ctypes.c_void_p(self._handle))
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()

    def __del__(self):
        self.free()

    def __repr__(self):
        status = "active" if self._handle else "freed"
        mb = (self.fp4_bytes + self.sf_bytes) / (1024 * 1024)
        return f"FP4WeightCache([{self.N_orig}, {self.K_orig}], {mb:.1f} MB, {status})"


def fp4_quantize(weight: torch.Tensor, bias: torch.Tensor = None) -> FP4WeightCache:
    """
    Quantize a weight matrix to FP4 and cache on device.

    Args:
        weight: [N, K] BF16 weight matrix on CUDA
        bias: Optional [N] bias vector

    Returns:
        FP4WeightCache handle for fast inference
    """
    return FP4WeightCache(weight, bias)


def fp4_cached_matmul(
    A: torch.Tensor,
    cache: FP4WeightCache,
) -> torch.Tensor:
    """
    FP4 matrix multiply with pre-quantized weights.

    Computes: D = quantize_fp4(A) @ cached_B_fp4.T

    Only A is quantized on the fly. B uses pre-quantized cache.
    Expected ~200-220 TFLOPS (vs 143 TFLOPS when both are quantized dynamically).

    Args:
        A: Input tensor [M, K] or [batch, M, K], BF16 on CUDA
        cache: Pre-quantized weight cache from fp4_quantize()

    Returns:
        D: Output tensor [M, N] or [batch, M, N], BF16 on CUDA
    """
    return cache.forward(A)


def fp4_cached_linear(
    x: torch.Tensor,
    cache: FP4WeightCache,
) -> torch.Tensor:
    """
    Drop-in replacement for F.linear using cached FP4 weights.

    Args:
        x: Input tensor [..., K], BF16 on CUDA
        cache: Pre-quantized weight cache (includes bias if provided)

    Returns:
        Output tensor [..., N]
    """
    return cache.forward(x)


if __name__ == "__main__":
    import time
    import sys

    print(f"FP4 GEMM Library loaded from {_lib_path}")
    print(f"Scale factor vector size: {SF_VEC_SIZE}")
    print()

    # Test at GPT-OSS dimensions
    sizes = [
        (128, 2880, 2880),    # Small batch
        (256, 2880, 2880),    # Medium batch
        (512, 2880, 2880),    # Large batch
        (2048, 2880, 2880),   # Very large
        (2048, 7680, 2880),   # Wide output (MLP up-proj)
        (2048, 2880, 7680),   # Wide input (MLP down-proj)
        (4096, 2880, 2880),   # Maximum
    ]

    iters = 20
    print("=" * 80)
    print("BENCHMARK: FP4 Dynamic vs FP4 Cached vs BF16 F.linear vs Float32 torch.mm")
    print("=" * 80)
    print()

    for M, N, K in sizes:
        print(f"--- M={M}, N={N}, K={K} ---")

        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

        # 1) FP4 Dynamic (both A and B quantized every call)
        D = fp4_matmul(A, B)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            D = fp4_matmul(A, B)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fp4_dyn_ms = (t1 - t0) / iters * 1000
        fp4_dyn_tflops = 2 * M * N * K / (fp4_dyn_ms / 1000) / 1e12

        # 2) FP4 Cached (B pre-quantized, only A quantized each call)
        cache = fp4_quantize(B)
        D_cached = cache.forward(A)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            D_cached = cache.forward(A)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fp4_cached_ms = (t1 - t0) / iters * 1000
        fp4_cached_tflops = 2 * M * N * K / (fp4_cached_ms / 1000) / 1e12

        # 3) BF16 F.linear (what PyTorch training uses)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            D_bf16 = torch.nn.functional.linear(A, B)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        bf16_ms = (t1 - t0) / iters * 1000

        # 4) Float32 torch.mm
        B_T = B.t().contiguous()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            D_f32 = torch.mm(A.float(), B_T.float()).bfloat16()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        f32_ms = (t1 - t0) / iters * 1000

        # Correctness (dynamic vs cached)
        D_ref = torch.nn.functional.linear(A.float(), B.float()).bfloat16()
        err_dyn = (D.float() - D_ref.float()).abs().mean() / D_ref.float().abs().mean()
        err_cached = (D_cached.float() - D_ref.float()).abs().mean() / D_ref.float().abs().mean()

        print(f"  FP4 Dynamic:  {fp4_dyn_ms:7.3f} ms  ({fp4_dyn_tflops:6.1f} TFLOPS)")
        print(f"  FP4 Cached:   {fp4_cached_ms:7.3f} ms  ({fp4_cached_tflops:6.1f} TFLOPS)  <- pre-quantized weights")
        print(f"  BF16 F.linear:{bf16_ms:7.3f} ms")
        print(f"  Float32 mm:   {f32_ms:7.3f} ms")
        print(f"  Cached speedup vs Dynamic: {fp4_dyn_ms/fp4_cached_ms:.2f}x")
        print(f"  Cached speedup vs Float32: {f32_ms/fp4_cached_ms:.2f}x")
        print(f"  Cached vs BF16: {'faster' if fp4_cached_ms < bf16_ms else 'slower'} ({fp4_cached_ms/bf16_ms:.2f}x)")
        print(f"  Rel error: dynamic={err_dyn:.6f}, cached={err_cached:.6f}")
        print()

        cache.free()

    cleanup()

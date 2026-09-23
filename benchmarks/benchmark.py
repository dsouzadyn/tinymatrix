"""Benchmarking script for TinyMatrix core operations."""

import time
from tinymatrix import Matrix


def benchmark_op(name: str, fn, runs: int = 5) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append(end - start)
    avg_ms = (sum(times) / len(times)) * 1000.0
    print(f"  {name:<30} {avg_ms:>8.3f} ms")
    return avg_ms


def run_benchmarks(sizes=(10, 50, 100)):
    print("=" * 45)
    print("TinyMatrix Benchmark Suite")
    print("=" * 45)

    for n in sizes:
        print(f"\n--- Matrix Size {n}x{n} ---")
        A = Matrix.random(n, n)
        B = Matrix.random(n, n)

        benchmark_op("Standard matmul (A @ B)", lambda: A @ B)
        benchmark_op(
            "Blocked matmul (tile=16)", lambda: A.matmul_blocked(B, block_size=16)
        )
        if n >= 32:
            benchmark_op("Strassen matmul", lambda: A.matmul_strassen(B, threshold=16))

        benchmark_op("Transpose (A.T)", lambda: A.T)
        benchmark_op("Determinant (A.det())", lambda: A.det())
        benchmark_op("Inverse (A.inv())", lambda: A.inv())
        benchmark_op("LU Decomposition", lambda: A.lu())
        benchmark_op("QR Decomposition", lambda: A.qr())
        benchmark_op(
            "Cholesky ((A.T @ A).cholesky())",
            lambda: (A.T @ A + Matrix.identity(n) * 0.1).cholesky(),
        )
        benchmark_op("SVD (A.svd())", lambda: A.svd())


if __name__ == "__main__":
    run_benchmarks()

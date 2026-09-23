"""Performance tracing and timeline profiling CLI for TinyMatrix.

Usage:
    uv run python benchmarks/trace.py
    uv run python benchmarks/trace.py --size 30,60 --save-trace trace.json
    uv run python benchmarks/trace.py --profile svd
"""

import argparse
import sys
from typing import List

from tinymatrix import (
    Matrix,
    PerformanceTracer,
    enable_tracing,
    profile_code,
)


def run_traced_suite(sizes: List[int], tracer: PerformanceTracer) -> None:
    print(f"Tracing operations across matrix dimensions: {sizes}...")

    for n in sizes:
        A = Matrix.random(n, n)
        B = Matrix.random(n, n)
        SPD = A.T @ A + Matrix.identity(n) * 0.1
        b_vec = [1.0] * n

        # Arithmetic & Matmul
        _ = A @ B
        _ = A.matmul_blocked(B, block_size=16)
        if n >= 16:
            _ = A.matmul_strassen(B, threshold=8)

        # Linear Algebra Core
        _ = A.det()
        _ = SPD.inv()
        _ = A.rank()
        _ = A.norm("fro")

        # Decompositions
        _ = A.lu()
        _ = A.qr()
        _ = SPD.cholesky()
        _ = SPD.eig()
        _ = A.svd()

        # Solvers
        _ = SPD.solve(b_vec)
        _ = A.lstsq(b_vec)


def main():
    parser = argparse.ArgumentParser(
        description="TinyMatrix Performance Tracing & Timeline Profiler"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="20,40",
        help="Comma-separated matrix sizes to trace (default: 20,40)",
    )
    parser.add_argument(
        "--save-trace",
        type=str,
        default=None,
        help="Path to save Chrome Tracing / Perfetto JSON (e.g. trace.json)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile a specific operation with cProfile (e.g. matmul, svd, lu, inv)",
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.size.split(",") if s.strip()]

    if args.profile:
        n = sizes[-1]
        print(
            f"\nProfiling operation '{args.profile}' on {n}x{n} matrix with cProfile..."
        )
        A = Matrix.random(n, n)
        SPD = A.T @ A + Matrix.identity(n) * 0.1

        ops = {
            "matmul": lambda: A @ A,
            "blocked": lambda: A.matmul_blocked(A, block_size=16),
            "strassen": lambda: A.matmul_strassen(A, threshold=16),
            "lu": lambda: A.lu(),
            "qr": lambda: A.qr(),
            "cholesky": lambda: SPD.cholesky(),
            "svd": lambda: A.svd(),
            "eig": lambda: SPD.eig(),
            "inv": lambda: SPD.inv(),
            "det": lambda: A.det(),
        }

        if args.profile not in ops:
            print(f"Unknown operation '{args.profile}'. Available: {list(ops.keys())}")
            sys.exit(1)

        _, stats = profile_code(ops[args.profile])
        stats.strip_dirs().sort_stats("cumulative").print_stats(20)
        return

    # Run PerformanceTracer
    tracer = enable_tracing(track_memory=True)
    tracer.clear()

    try:
        run_traced_suite(sizes, tracer)
    finally:
        tracer.stop()

    print("\n" + "=" * 78)
    print("TinyMatrix Performance Trace Summary")
    print("=" * 78)
    print(tracer.format_summary_table())

    if args.save_trace:
        tracer.export_chrome_trace(args.save_trace)
        print(f"\n[OK] Chrome Trace JSON exported to: {args.save_trace}")
        print(
            "     View interactively at: https://ui.perfetto.dev/ or chrome://tracing"
        )


if __name__ == "__main__":
    main()

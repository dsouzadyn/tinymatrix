# TinyMatrix Performance & Optimization Guide

TinyMatrix is engineered to be a dependency-free, pure-Python linear algebra and matrix manipulation helper.

## Computational Complexity

| Operation | Algorithm | Time Complexity | Space Complexity |
|---|---|---|---|
| Addition / Subtraction | Element-wise with broadcasting | $O(M \times N)$ | $O(M \times N)$ |
| Matrix Multiplication (`@`) | Standard triple loop / blocked tiling | $O(M \times K \times N)$ | $O(M \times N)$ |
| Strassen Multiplication | Recursive $7$-product divide-and-conquer | $O(N^{\log_2 7}) \approx O(N^{2.807})$ | $O(N^2)$ |
| Transpose (`.T`) | Matrix transposition | $O(M \times N)$ | $O(M \times N)$ |
| Determinant (`.det()`) | Gaussian elimination with partial pivoting | $O(N^3)$ | $O(N^2)$ |
| Inverse (`.inv()`) | Gauss-Jordan elimination on $[A \mid I]$ | $O(N^3)$ | $O(N^2)$ |
| Rank (`.rank()`) | Row echelon form with machine tolerance | $O(M \times N \times \min(M, N))$ | $O(M \times N)$ |
| LU Decomposition | Pivoted LU ($PA = LU$) | $O(M \times N \times \min(M, N))$ | $O(M \times N)$ |
| QR Decomposition | Modified Gram-Schmidt | $O(M \times N^2)$ | $O(M \times N)$ |
| Cholesky Decomposition | Cholesky-Banachiewicz algorithm | $O(N^3 / 3)$ | $O(N^2)$ |
| SVD | Spectral decomposition via Jacobi rotations | $O(M N^2 + N^3)$ | $O(M \times N)$ |
| Linear Solve (`solve`) | Forward and back substitution via LU | $O(N^3)$ | $O(N^2)$ |

## Optimization Highlights

1. **Blocked Matrix Multiplication**: Cache-friendly loop tiling (`.matmul_blocked(B, block_size=32)`) minimizes CPU cache thrashing on medium-to-large matrices.
2. **Strassen's Algorithm**: For large square matrices, `.matmul_strassen(B)` offers sub-cubic asymptotic scaling.
3. **Lazy Evaluation Tree**: Build expression graphs via `.lazy()` to defer execution and eliminate intermediate allocations in chained arithmetic expressions (`(A.lazy() + B.lazy()) @ C.lazy()`).
4. **NumPy Interoperability**: Seamless conversion to and from NumPy ndarrays using `.to_numpy()` and `Matrix.from_numpy()` when NumPy is available in your environment.

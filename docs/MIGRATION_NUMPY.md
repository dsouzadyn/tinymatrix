# TinyMatrix vs NumPy: Comparison & Migration Guide

TinyMatrix is designed for environments where installing NumPy is impractical (embedded systems, constrained serverless functions, educational contexts, zero-dependency requirements) or where you want a clean, single-class Matrix helper.

## Feature Comparison

| Feature | TinyMatrix | NumPy (`numpy.ndarray`) |
|---|:---:|:---:|
| **Zero Dependencies** | Yes (Pure Python) | No (compiled C/Fortran binaries) |
| **Install Size** | < 50 KB | > 30 MB |
| **Startup Overhead** | ~1 ms | ~50-100 ms |
| **Matrix Multiplication** | `@`, standard, blocked, Strassen | `@` (BLAS/LAPACK accelerated) |
| **Dtypes Supported** | `float`, `int`, `decimal`, `complex` | Extensive native C types |
| **Arbitrary Precision** | Yes (via `dtype="decimal"`) | Requires external (`mpmath`/`gmpy2`) |
| **Decompositions** | LU, QR, Cholesky, SVD, Eig | Full LAPACK suite |
| **Solvers** | `solve(A, b)`, `lstsq(A, b)` | `np.linalg.solve`, `np.linalg.lstsq` |
| **Sparse Support** | CSR & CSC formats | `scipy.sparse` (requires SciPy) |
| **Broadcasting** | 2D vector broadcasting | N-dimensional broadcasting |

## Syntax Cheat Sheet

| NumPy Syntax | TinyMatrix Equivalent | Notes |
|---|---|---|
| `np.array([[1, 2], [3, 4]])` | `Matrix(matrix=[[1, 2], [3, 4]])` | Standard construction |
| `np.zeros((m, n))` | `Matrix.zeroes(m, n)` | Zero matrix |
| `np.ones((m, n))` | `Matrix.ones(m, n)` | Ones matrix |
| `np.eye(n)` | `Matrix.identity(n)` | Identity matrix |
| `A.T` | `A.T` | Transpose |
| `A @ B` | `A @ B` | Matrix multiplication |
| `A * B` | `A * B` | Element-wise (Hadamard) product |
| `A * 2` | `A * 2` | Scalar multiplication |
| `A / 2` | `A / 2` | Scalar division |
| `np.linalg.det(A)` | `A.det()` or `det(A)` | Determinant |
| `np.linalg.inv(A)` | `A.inv()` or `inv(A)` | Matrix inverse |
| `np.linalg.matrix_rank(A)` | `A.rank()` or `rank(A)` | Matrix rank |
| `np.linalg.norm(A)` | `A.norm()` or `norm(A)` | Frobenius norm |
| `np.linalg.cond(A)` | `A.cond()` or `cond(A)` | Condition number |
| `np.linalg.pinv(A)` | `A.pinv()` or `pinv(A)` | Moore-Penrose pseudo-inverse |
| `np.linalg.solve(A, b)` | `A.solve(b)` or `solve(A, b)` | Linear system solver |
| `np.linalg.lstsq(A, b)` | `A.lstsq(b)` or `lstsq(A, b)` | Least-squares solver |
| `np.vstack([A, B])` | `vstack([A, B])` | Vertical stacking |
| `np.hstack([A, B])` | `hstack([A, B])` | Horizontal stacking |
| `np.block([[A, B], [C, D]])` | `block([[A, B], [C, D]])` | 2D block matrix concatenation |

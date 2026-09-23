<div align="center">

<img src="docs/logo.svg" alt="TinyMatrix Logo" width="520"/>

<p><strong>A minimal, zero-dependency Python library for matrix manipulation, linear algebra, decompositions, solvers, and sparse computation.</strong></p>

[![CI](https://github.com/dsouzadyn/tinymatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/dsouzadyn/tinymatrix/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/dsouzadyn/tinymatrix)

</div>

---

TinyMatrix is engineered for learning, serverless functions, embedded systems, and lightweight scripts where a pure-Python, readable matrix library is preferred over a multi-megabyte binary dependency like NumPy.

## 🚀 Key Highlights

- **Zero Dependencies**: Pure Python (>= 3.12) with standard library only.
- **Rich Data Types**: Native support for `float`, `int`, `decimal` (exact precision), and `complex`.
- **Full Linear Algebra Core**: Determinants, matrix inverses, matrix rank, norms, Kronecker products, condition numbers, and Moore-Penrose pseudo-inverses.
- **Decompositions & Solvers**: LU, QR (Modified Gram-Schmidt), Cholesky, SVD, Eigenvalues/Eigenvectors, linear equation solver ($Ax = b$), and least-squares ($Ax \approx b$).
- **Sparse Representations**: Compressed Sparse Row (`CSRMatrix`) and Column (`CSCMatrix`) formats with sparse-dense matrix multiplication.
- **High-Performance Optimizations**: Blocked (cache-friendly) multiplication, Strassen's algorithm, and `LazyMatrix` deferred expression evaluation.
- **Pythonic Ergonomics**: 2D slicing, fancy indexing, boolean masks, axis reductions, broadcasting, and immutability mode (`frozen=True`).

---

## 📦 Installation

Install TinyMatrix directly with `pip` or `uv`:

```bash
pip install tinymatrix
```

Or install in development mode from source:

```bash
git clone https://github.com/dsouzadyn/tinymatrix.git
cd tinymatrix
uv sync
```

---

## 💡 Quickstart & Feature Guide

### 1. Matrix Creation & Data Types

TinyMatrix supports explicit construction, common factories, random distributions, and multiple numeric datatypes:

```python
from decimal import Decimal
from tinymatrix import Matrix

# From nested lists
A = Matrix(matrix=[[1, 2], [3, 4]])

# Built-in constructors
Z = Matrix.zeroes(2, 3)          # 2x3 zeros (float)
O = Matrix.ones(3, 3)            # 3x3 ones
I = Matrix.identity(3)           # 3x3 identity matrix
R = Matrix.random(3, 3)          # Uniform in [0, 1)
N = Matrix.normal(3, 3)          # Standard normal distribution

# Explicit dtypes: float, int, decimal, complex
A_int = Matrix(matrix=[[1, 2], [3, 4]], dtype="int")
A_dec = Matrix(matrix=[[Decimal("1.1"), Decimal("2.2")]], dtype="decimal")
A_cplx = Matrix(matrix=[[1 + 2j, 3 + 4j]], dtype="complex")
```

### 2. Arithmetic, Broadcasting & Power

Standard operators work seamlessly with scalar operands, other matrices, and automatic vector broadcasting:

```python
A = Matrix(matrix=[[1, 2], [3, 4]])
B = Matrix(matrix=[[5, 6], [7, 8]])

# Element-wise operations
C = A + B                         # Matrix addition
D = A - B                         # Matrix subtraction
H = A * B                         # Hadamard (element-wise) product
div = A / 2                       # Scalar division

# Automatic 2D vector broadcasting
row_vec = Matrix(matrix=[[10, 20]])  # 1x2 row vector
B_sum = A + row_vec                  # Broadcasts across all rows

# Matrix multiplication & matrix power
prod = A @ Matrix(matrix=[[1], [2]]) # Matrix multiplication
P = A ** 3                           # Matrix power via repeated squaring
```

### 3. Indexing, Slicing & Boolean Masking

Flexible indexing with full support for integer slices, negative indices, row selection, and boolean conditions:

```python
A = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Standard 2D access & slicing
val = A[0, 1]                     # 2.0
row = A[0]                        # 1x3 Matrix
sub = A[0:2, 1:3]                 # 2x2 submatrix

# Fancy indexing (row selection)
selected = A[[0, 2]]              # Rows 0 and 2

# Boolean masks & comparisons
mask = A > 5                      # Matrix of 1s and 0s
gt_vals = A[A > 5]                # [6.0, 7.0, 8.0, 9.0]

# Immutability mode (frozen matrix)
A_frozen = A.copy().freeze()
# A_frozen[0, 0] = 99             # Raises TypeError!
```

### 4. Helpers, Reductions & Math Functions

Access components, perform axis reductions, and apply element-wise functions:

```python
A = Matrix(matrix=[[1, 2], [3, 4]])

# Structural helpers
r0 = A.row(0)                     # 1x2 Matrix
c1 = A.col(1)                     # 2x1 Matrix
flat = A.flatten()                # [1.0, 2.0, 3.0, 4.0]
diag = A.diagonal()               # [1.0, 4.0]
tr = A.trace()                    # 5.0

# Reductions (axis=None for scalar, axis=0 for cols, axis=1 for rows)
total = A.sum()                   # 10.0
col_means = A.mean(axis=0)        # [[2.0, 3.0]] (1x2 Matrix)
row_maxs = A.max(axis=1)          # [[2.0], [4.0]] (2x1 Matrix)
std_dev = A.std()                 # Standard deviation

# Reshaping & Squeezing
R = A.reshape(1, 4)               # Reshape to 1x4
sq = Matrix(matrix=[[42]]).squeeze() # 42.0

# Element-wise math
sqrts = A.sqrt()                  # Element-wise sqrt
exps = A.exp()                    # Element-wise exp
scaled = A.apply(lambda x: x * 10) # Custom callback
```

### 5. Linear Algebra Core & Decompositions

Comprehensive linear algebra algorithms with both OOP and functional APIs:

```python
from tinymatrix import det, inv, rank, norm, kron, cond, pinv

A = Matrix(matrix=[[4, 7], [2, 6]])

# Linear Algebra Core
d = A.det()                       # Determinant (10.0)
inv_A = A.inv()                   # Inverse
rk = A.rank()                     # Matrix rank (2)
fro_norm = A.norm("fro")          # Frobenius norm
L1_norm = A.norm(1)               # 1-norm (max column sum)
kappa = A.cond()                  # Condition number
A_pinv = A.pinv()                 # Moore-Penrose pseudo-inverse

# Decompositions
P, L, U = A.lu()                  # Pivoted LU decomposition (P @ A == L @ U)
Q, R = A.qr()                     # QR decomposition (A == Q @ R)
evals, evecs = A.eig()            # Eigenvalues & Eigenvectors
U_svd, s_vals, Vt = A.svd()       # Singular Value Decomposition (SVD)

# Cholesky decomposition (symmetric positive-definite)
SPD = A.T @ A
L_chol = SPD.cholesky()           # L @ L.T == SPD
```

### 6. Linear Equation Solvers

Solve systems of linear equations and least-squares problems:

```python
from tinymatrix import solve, lstsq

# Exact square system: A @ x = b
A = Matrix(matrix=[[2, 1], [1, 3]])
b = [5, 5]
x = solve(A, b)                   # [[2.0], [1.0]] (x=2, y=1)

# Overdetermined least-squares: min ||A @ x - b||_2
A_rect = Matrix(matrix=[[1, 1], [1, 2], [1, 3]])
b_rect = [2, 3, 4]
x_lsq, residual = lstsq(A_rect, b_rect)
```

### 7. Sparse Matrices & Stacking

Work with memory-efficient sparse formats and combine matrices:

```python
from tinymatrix import CSRMatrix, CSCMatrix, vstack, hstack, block

# Sparse representations
dense = Matrix(matrix=[[1, 0, 0], [0, 0, 5], [2, 0, 0]])
csr = CSRMatrix.from_dense(dense) # Compressed Sparse Row
csc = csr.to_csc()                # Convert to Compressed Sparse Column
sparse_prod = csr @ dense         # Fast sparse-dense matmul
dense_again = csr.to_dense()

# Stacking & Block matrix assembly
A = Matrix(matrix=[[1, 2], [3, 4]])
B = Matrix(matrix=[[5, 6], [7, 8]])

V = vstack([A, B])                # Vertical stack (4x2)
H = hstack([A, B])                # Horizontal stack (2x4)
BLK = block([[A, B], [B, A]])     # 2D Block concatenation (4x4)
```

### 8. Performance & Lazy Evaluation

Optimize throughput on larger matrices with tiling, Strassen's algorithm, and expression graph evaluation:

```python
A = Matrix.random(64, 64)
B = Matrix.random(64, 64)

# Cache-friendly blocked matrix multiplication
C_blocked = A.matmul_blocked(B, block_size=32)

# Strassen sub-cubic multiplication for square matrices
C_strassen = A.matmul_strassen(B, threshold=16)

# Lazy evaluation tree (defers execution until evaluate())
lazy_expr = (A.lazy() + B.lazy()) @ A.lazy()
result = lazy_expr.evaluate()
```

---

## 📖 Practical Examples

Executable standalone scripts are provided in the [`examples/`](examples/) directory:

- [`examples/linear_regression.py`](examples/linear_regression.py): Ordinary least-squares line fitting.
- [`examples/pagerank.py`](examples/pagerank.py): Markov transition matrix and power iteration PageRank algorithm.
- [`examples/kalman_filter.py`](examples/kalman_filter.py): 1D tracking Kalman filter with state updates and covariance propagation.
- [`benchmarks/benchmark.py`](benchmarks/benchmark.py): Performance profiling script measuring speed across matrix sizes.

---

## 📚 Documentation & Guides

- **Sphinx Documentation**: Complete HTML API docs built with the **Furo** theme in `docs/` (`uv run sphinx-build -b html docs docs/_build/html`).
- **[NumPy Migration Guide](docs/MIGRATION_NUMPY.md)**: Detailed syntax translation table and comparison guide for NumPy users.
- **[Performance & Complexity Guide](docs/PERFORMANCE.md)**: Asymptotic time and space complexity table for all operations.

---

## 🛠️ Development

TinyMatrix uses `uv` for fast dependency management and `pre-commit` for code quality:

```bash
# Install dependencies & pre-commit hooks
uv sync
uv run pre-commit install

# Run test suite with full coverage
uv run pytest --cov=tinymatrix --cov-report=term-missing

# Run linter and formatter
uv run ruff check .
uv run ruff format --check .

# Build Sphinx documentation
uv run sphinx-build -b html docs docs/_build/html
```

---

## 📄 License

Licensed under the [MIT License](LICENSE).


# TinyMatrix

TinyMatrix is a minimal, dependency-free Python library that provides a small Matrix type with basic linear-algebra-like operations: construction, element-wise arithmetic, matrix multiplication, transpose, indexing/slicing, and a few static constructors (identity, zeroes, ones).

Designed for learning and small scripts where a tiny and readable matrix helper is preferred over a large dependency.

## Install

Install editable from the project root for development:

```bash
python -m pip install -e .
```

Or install normally (if packaged):

```bash
python -m pip install .
```

## Quick usage

```python
from tinymatrix import Matrix

# Create a matrix
A = Matrix(matrix=[[1, 2], [3, 4]])

# Arithmetic
B = Matrix(matrix=[[5, 6], [7, 8]])
C = A + B  # element-wise addition

# Scalar arithmetic & division
D = A * 2
div = A / 2

# Element-wise multiplication (Hadamard) & broadcasting
H = A * B
broadcast_sum = A + Matrix(matrix=[[10, 20]])  # (2x2) + (1x2)

# Matrix multiplication & power
E = A @ Matrix(matrix=[[1], [2]])
P = A ** 3

# Transpose & helpers
T = A.T
r0 = A.row(0)
c1 = A.col(1)
diag = A.diagonal()
tr = A.trace()
# Linear algebra core
d = A.det()  # or det(A)
inv_A = A.inv()  # or inv(A)
rk = A.rank()  # or rank(A)
f_norm = A.norm("fro")  # or norm(A)
K = A.kron(B)  # or kron(A, B)

# Reductions & math
total = A.sum()
col_means = A.mean(axis=0)
row_maxs = A.max(axis=1)
scaled = A.apply(lambda x: x ** 2)
# Indexing / slicing
val = A[0, 1]
sub = A[0:1, :]

# Constructors
I = Matrix.identity(3)
Z = Matrix.zeroes(2, 3)
O = Matrix.ones(2, 2)
R = Matrix.random(2, 3, low=0.0, high=1.0)
N = Matrix.normal(3, 3, mean=0.0, std=1.0)

Refer to `tests/test_matrix.py` for additional examples and expected behavior.

## Development

Install development dependencies and setup pre-commit hooks:

```bash
uv sync
uv run pre-commit install
```

Run tests:

```bash
uv run pytest
```

Run linter and formatter:

```bash
uv run ruff check .
uv run ruff format --check .
```
## License

Licensed under the MIT License. See the LICENSE file for full text.

## 🗺️ **Roadmap / TODO**

### Phase 1: Foundation & Quality
* [x] Add full type hints for all public APIs
* [x] Add optional `dtype` support (float/int/Decimal/complex)
* [x] Add stricter input and type validation
* [x] Improve slice/index edge-case tests
* [x] Add GitHub Actions for lint, test, and build
* [x] Add code coverage tooling + badge

### Phase 2: Core Matrix Operations
* [x] Add `.row(i)`, `.col(j)`, `.flatten()` helpers
* [x] Add `.apply(func)` for element-wise operations
* [x] Add broadcasting support (1×N and M×1 vectors)
* [x] Add random matrix constructors (uniform, normal)
* [x] Improve printing for large matrices (ellipsis, alignment)
* [x] Add `.reshape()`, `.squeeze()`, `.expand_dims()`
* [x] Add element-wise functions (abs, sqrt, exp, log, sin, cos, etc.)
* [x] Add reduction operations (sum, mean, std, min, max) with axis support
* [x] Add `.diagonal()`, `.trace()` operations

### Phase 3: Linear Algebra Core
* [x] Implement determinant calculation
* [x] Implement matrix inverse
* [x] Implement matrix rank
* [x] Implement matrix norm (Frobenius, spectral, etc.)
* [x] Add matrix power (`A ** n`)
* [x] Add Kronecker product
* [x] Add Hadamard (element-wise) product operator
### Phase 4: Decompositions & Solvers
* [x] Add LU decomposition
* [x] Add QR decomposition
* [x] Add Cholesky decomposition
* [x] Add SVD (Singular Value Decomposition)
* [x] Add eigenvalue/eigenvector computation
* [x] Add linear system solver `Ax = b`
* [x] Add least-squares solver

### Phase 5: Performance & Optimization
* [ ] Optimize matrix multiplication (Strassen or blocked algorithms)
* [ ] Add optional NumPy fallback mode for performance comparisons
* [ ] Add lazy evaluation for chained operations (optional)
* [ ] Profile and optimize hotspots
* [ ] Add benchmarking utilities and performance docs

### Phase 6: Advanced Features
* [ ] Add sparse matrix support (CSR/CSC formats)
* [ ] Add immutability mode (`frozen=True`)
* [ ] Add matrix views (no-copy slicing where possible)
* [ ] Add `.copy()` and copy-on-write semantics
* [ ] Add stacking operations (vstack, hstack, block)
* [ ] Add advanced indexing (fancy indexing, boolean masks)
* [ ] Add matrix condition number calculation
* [ ] Add pseudo-inverse (Moore-Penrose)

### Phase 7: Documentation & Polish
* [ ] Expand README with comprehensive examples
* [ ] Add Sphinx docs site
* [ ] Add example notebooks (Jupyter)
* [ ] Add comparison guide: TinyMatrix vs NumPy
* [ ] Add migration guide for NumPy users
* [ ] Complete PyPI metadata
* [ ] Add mathematical notation in docstrings
* [ ] Create logo and branding

### Stretch Goals
* [ ] Add matrix calculus operations (gradient, Jacobian, Hessian)
* [ ] Add graph/network matrices (adjacency, Laplacian)
* [ ] Add special matrix generators (Toeplitz, Hankel, Vandermonde)
* [ ] Add matrix equation solvers (Sylvester, Lyapunov)
* [ ] Add tensor product operations
* [ ] Consider Cython/C extensions for critical paths (while keeping pure-Python as default)

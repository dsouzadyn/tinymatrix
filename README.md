
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

# Scalar multiply
D = A * 2

# Matrix multiplication
E = A @ Matrix(matrix=[[1], [2]])

# Transpose
T = A.T

# Indexing / slicing
val = A[0, 1]
sub = A[0:1, :]

# Static constructors
I = Matrix.identity(3)
Z = Matrix.zeroes(2, 3)
O = Matrix.ones(2, 2)
```

Refer to `tests/test_matrix.py` for additional examples and expected behavior.

## Running tests

From the project root run:

```bash
pytest -q
```

## License

Licensed under the MIT License. See the LICENSE file for full text.

## 🗺️ **Roadmap / TODO**

### Phase 1: Foundation & Quality
* [x] Add full type hints for all public APIs
* [x] Add optional `dtype` support (float/int/Decimal/complex)
* [ ] Add stricter input and type validation
* [ ] Improve slice/index edge-case tests
* [ ] Add GitHub Actions for lint, test, and build
* [ ] Add code coverage tooling + badge

### Phase 2: Core Matrix Operations
* [ ] Add `.row(i)`, `.col(j)`, `.flatten()` helpers
* [ ] Add `.apply(func)` for element-wise operations
* [ ] Add broadcasting support (1×N and M×1 vectors)
* [ ] Add random matrix constructors (uniform, normal)
* [ ] Improve printing for large matrices (ellipsis, alignment)
* [ ] Add `.reshape()`, `.squeeze()`, `.expand_dims()`
* [ ] Add element-wise functions (abs, sqrt, exp, log, sin, cos, etc.)
* [ ] Add reduction operations (sum, mean, std, min, max) with axis support
* [ ] Add `.diagonal()`, `.trace()` operations

### Phase 3: Linear Algebra Core
* [ ] Implement determinant calculation
* [ ] Implement matrix inverse
* [ ] Implement matrix rank
* [ ] Implement matrix norm (Frobenius, spectral, etc.)
* [ ] Add matrix power (`A ** n`)
* [ ] Add Kronecker product
* [ ] Add Hadamard (element-wise) product operator

### Phase 4: Decompositions & Solvers
* [ ] Add LU decomposition
* [ ] Add QR decomposition
* [ ] Add Cholesky decomposition
* [ ] Add SVD (Singular Value Decomposition)
* [ ] Add eigenvalue/eigenvector computation
* [ ] Add linear system solver `Ax = b`
* [ ] Add least-squares solver

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
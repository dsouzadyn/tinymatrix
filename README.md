
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

Simple personal/academic use. Check the project metadata for license details.

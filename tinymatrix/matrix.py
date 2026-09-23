import cmath
import math
import random
from typing import (
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

from .exceptions import ShapeError, SingularMatrixError
from .types import DTYPES, TinyMatrixData, TinyMatrixIndexPair, TinyMatrixNumeric


class Matrix:
    def __init__(
        self,
        m: Optional[int] = None,
        n: Optional[int] = None,
        matrix: Optional[TinyMatrixData] = None,
        dtype: str = "float",
    ) -> None:
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.dtype = dtype
        self.cast = DTYPES[dtype]

        if matrix is not None:
            if not isinstance(matrix, list):
                raise TypeError("Matrix data must be a list of lists")
            if len(matrix) == 0:
                self.m = 0
                self.n = 0
                self.M = []
            else:
                for row in matrix:
                    if not isinstance(row, (list, tuple)):
                        raise TypeError("Each row must be a list or tuple")
                first_len = len(matrix[0])
                if not all(len(row) == first_len for row in matrix):
                    raise ShapeError("All rows must have equal length")

                self.M = [[self.cast(x) for x in row] for row in matrix]
                self.m = len(matrix)
                self.n = first_len
        else:
            if m is None or n is None:
                raise ValueError("Provide either matrix or (m, n)")
            if type(m) is not int or type(n) is not int:
                raise TypeError("Dimensions m and n must be integers")
            if m < 0 or n < 0:
                raise ValueError("Dimensions m and n must be non-negative integers")
            self.m = m
            self.n = n
            self.M = [[self.cast(0) for _ in range(n)] for _ in range(m)]

    def _is_valid_scalar(self, value: object) -> bool:
        if isinstance(value, (Matrix, list, tuple, dict, set)):
            return False
        try:
            self.cast(value)
            return True
        except Exception:
            return False

    # =========================================================================
    # Static & Class Constructors
    # =========================================================================

    @staticmethod
    def zeroes(m: int, n: int, dtype: str = "float") -> "Matrix":
        return Matrix(m, n, dtype=dtype)

    @staticmethod
    def ones(m: int, n: int, dtype: str = "float") -> "Matrix":
        mat = Matrix(m, n, dtype=dtype)
        one = mat.cast(1)
        for row in range(m):
            for col in range(n):
                mat.M[row][col] = one
        return mat

    @staticmethod
    def identity(n: int, dtype: str = "float") -> "Matrix":
        if type(n) is not int:
            raise TypeError("Dimension n must be an integer")
        if n < 0:
            raise ValueError("Dimension n must be non-negative")
        mat = Matrix(n, n, dtype=dtype)
        one = mat.cast(1)
        for row in range(n):
            mat.M[row][row] = one
        return mat

    @classmethod
    def random(
        cls,
        m: int,
        n: int,
        low: float = 0.0,
        high: float = 1.0,
        dtype: str = "float",
    ) -> "Matrix":
        """Construct matrix with random elements uniformly drawn from [low, high)."""
        mat = cls(m, n, dtype=dtype)
        for r in range(m):
            for c in range(n):
                val = random.uniform(low, high)
                mat.M[r][c] = mat.cast(val)
        return mat

    @classmethod
    def uniform(
        cls,
        m: int,
        n: int,
        low: float = 0.0,
        high: float = 1.0,
        dtype: str = "float",
    ) -> "Matrix":
        """Alias for Matrix.random()."""
        return cls.random(m, n, low=low, high=high, dtype=dtype)

    @classmethod
    def normal(
        cls,
        m: int,
        n: int,
        mean: float = 0.0,
        std: float = 1.0,
        dtype: str = "float",
    ) -> "Matrix":
        """Construct matrix with normally distributed random elements."""
        mat = cls(m, n, dtype=dtype)
        for r in range(m):
            for c in range(n):
                val = random.gauss(mean, std)
                mat.M[r][c] = mat.cast(val)
        return mat

    # =========================================================================
    # Properties & Basic Methods
    # =========================================================================

    def shape(self) -> Tuple[int, int]:
        return self.m, self.n

    def is_square(self) -> bool:
        return self.m == self.n

    def tolist(self) -> List[List[TinyMatrixNumeric]]:
        return [row[:] for row in self.M]

    def copy(self) -> "Matrix":
        return Matrix(matrix=[row[:] for row in self.M], dtype=self.dtype)

    def astype(self, dtype: str) -> "Matrix":
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")

        cast = DTYPES[dtype]
        data = [[cast(x) for x in row] for row in self.M]
        return Matrix(matrix=data, dtype=dtype)

    def __len__(self) -> int:
        return self.m

    def __iter__(self) -> Iterator["Matrix"]:
        for r in range(self.m):
            yield self[r]

    # =========================================================================
    # Indexing & Slicing
    # =========================================================================

    @overload
    def __getitem__(self, idx: int) -> "Matrix": ...

    @overload
    def __getitem__(self, idx: Tuple[int, int]) -> TinyMatrixNumeric: ...

    @overload
    def __getitem__(self, idx: Tuple[int, slice]) -> "Matrix": ...

    @overload
    def __getitem__(self, idx: Tuple[slice, int]) -> "Matrix": ...

    @overload
    def __getitem__(self, idx: Tuple[slice, slice]) -> "Matrix": ...

    def __getitem__(
        self, idx: Union[int, TinyMatrixIndexPair]
    ) -> Union[TinyMatrixNumeric, "Matrix"]:
        if isinstance(idx, tuple):
            row, col = idx

            if isinstance(row, slice) or isinstance(col, slice):
                rows = range(*row.indices(self.m)) if isinstance(row, slice) else [row]
                cols = range(*col.indices(self.n)) if isinstance(col, slice) else [col]

                data = [[self.M[r][c] for c in cols] for r in rows]
                return Matrix(matrix=data, dtype=self.dtype)

            return self.M[row][col]
        elif isinstance(idx, int):
            return Matrix(matrix=[self.M[idx][:]], dtype=self.dtype)
        else:
            raise TypeError("Invalid index type")

    def __setitem__(
        self,
        idx: Union[int, TinyMatrixIndexPair],
        value: Union[
            TinyMatrixNumeric,
            "Matrix",
            List[TinyMatrixNumeric],
            Tuple[TinyMatrixNumeric, ...],
        ],
    ) -> None:
        if isinstance(idx, tuple):
            row, col = idx

            if isinstance(row, slice) or isinstance(col, slice):
                if not isinstance(value, Matrix):
                    raise TypeError("Slice assignment requires a Matrix")

                rows = range(*row.indices(self.m)) if isinstance(row, slice) else [row]
                cols = range(*col.indices(self.n)) if isinstance(col, slice) else [col]

                if value.m != len(rows) or value.n != len(cols):
                    raise ShapeError("Slice shape mismatch")

                for i, ri in enumerate(rows):
                    for j, cj in enumerate(cols):
                        self.M[ri][cj] = self.cast(value.M[i][j])
                return

            if not self._is_valid_scalar(value):
                raise TypeError("Single element assignment requires a number")

            self.M[row][col] = self.cast(value)
            return

        elif isinstance(idx, int):
            if not isinstance(value, (list, tuple)):
                raise TypeError("Row assignment requires a list or tuple")
            if len(value) != self.n:
                raise ShapeError("Row length mismatch")

            self.M[idx] = [self.cast(v) for v in value]
            return
        else:
            raise TypeError("Invalid index type")

    # =========================================================================
    # Helpers: Row, Col, Flatten, Diagonal, Trace
    # =========================================================================

    def row(self, i: int) -> "Matrix":
        """Return row i as a 1×N Matrix."""
        if type(i) is not int:
            raise TypeError("Row index must be an integer")
        if i < -self.m or i >= self.m:
            raise IndexError("Row index out of range")
        return Matrix(matrix=[self.M[i][:]], dtype=self.dtype)

    def col(self, j: int) -> "Matrix":
        """Return column j as an M×1 Matrix."""
        if type(j) is not int:
            raise TypeError("Column index must be an integer")
        if j < -self.n or j >= self.n:
            raise IndexError("Column index out of range")
        return Matrix(matrix=[[r[j]] for r in self.M], dtype=self.dtype)

    def flatten(self) -> List[TinyMatrixNumeric]:
        """Return elements as a 1D list in row-major order."""
        return [x for row in self.M for x in row]

    def diagonal(self) -> List[TinyMatrixNumeric]:
        """Return the main diagonal elements as a list."""
        k = min(self.m, self.n)
        return [self.M[i][i] for i in range(k)]

    def trace(self) -> TinyMatrixNumeric:
        """Return the sum of the main diagonal for a square matrix."""
        if self.m != self.n:
            raise ShapeError("Trace is only defined for square matrices")
        if self.m == 0:
            return self.cast(0)
        return self.cast(sum(self.diagonal()))

    # =========================================================================
    # Linear Algebra Core: Determinant, Inverse, Rank, Norm, Kronecker
    # =========================================================================

    def det(self) -> TinyMatrixNumeric:
        """Compute the determinant of a square matrix."""
        if self.m != self.n:
            raise ShapeError("Determinant is only defined for square matrices")

        n = self.m
        if n == 0:
            return self.cast(1)
        if n == 1:
            return self.M[0][0]
        if n == 2:
            return self.cast(self.M[0][0] * self.M[1][1] - self.M[0][1] * self.M[1][0])

        # Gaussian elimination with partial pivoting
        A = [row[:] for row in self.M]
        sign = 1
        tol = 1e-14

        for k in range(n):
            pivot_row = k
            max_val = abs(A[k][k])
            for i in range(k + 1, n):
                val = abs(A[i][k])
                if val > max_val:
                    max_val = val
                    pivot_row = i

            if max_val < tol:
                return self.cast(0)

            if pivot_row != k:
                A[k], A[pivot_row] = A[pivot_row], A[k]
                sign = -sign

            pivot = A[k][k]
            for i in range(k + 1, n):
                factor = A[i][k] / pivot
                for j in range(k + 1, n):
                    A[i][j] -= factor * A[k][j]

        prod = sign
        for k in range(n):
            prod *= A[k][k]

        if self.dtype == "int":
            return self.cast(round(float(prod)))
        return self.cast(prod)

    def inv(self) -> "Matrix":
        """Compute the multiplicative inverse of a square matrix using Gauss-Jordan elimination."""
        if self.m != self.n:
            raise ShapeError("Matrix inverse is only defined for square matrices")

        n = self.m
        if n == 0:
            return Matrix(0, 0, dtype=self.dtype)

        aug = []
        for i in range(n):
            identity_row = [self.cast(1) if i == j else self.cast(0) for j in range(n)]
            aug.append([row_val for row_val in self.M[i]] + identity_row)

        tol = 1e-14
        for k in range(n):
            pivot_row = k
            max_val = abs(aug[k][k])
            for i in range(k + 1, n):
                val = abs(aug[i][k])
                if val > max_val:
                    max_val = val
                    pivot_row = i

            if max_val < tol:
                raise SingularMatrixError("Matrix is singular and cannot be inverted")

            if pivot_row != k:
                aug[k], aug[pivot_row] = aug[pivot_row], aug[k]

            pivot = aug[k][k]
            for j in range(k, 2 * n):
                aug[k][j] /= pivot

            for i in range(n):
                if i != k:
                    factor = aug[i][k]
                    if abs(factor) > 0:
                        for j in range(k, 2 * n):
                            aug[i][j] -= factor * aug[k][j]

        inv_data = [[self.cast(aug[i][j]) for j in range(n, 2 * n)] for i in range(n)]
        return Matrix(matrix=inv_data, dtype=self.dtype)

    def rank(self, tol: Optional[float] = None) -> int:
        """Compute matrix rank using Gaussian elimination with partial pivoting."""
        if self.m == 0 or self.n == 0:
            return 0

        A = [
            [float(abs(x)) if self.dtype == "complex" else float(x) for x in row]
            for row in self.M
        ]

        if tol is None:
            max_val = max(abs(x) for row in A for x in row) if A else 0.0
            tol = max(self.m, self.n) * 1e-15 * max_val if max_val > 0 else 1e-14

        lead = 0
        r = 0
        while r < self.m and lead < self.n:
            pivot_row = r
            max_val = abs(A[r][lead])
            for i in range(r + 1, self.m):
                val = abs(A[i][lead])
                if val > max_val:
                    max_val = val
                    pivot_row = i

            if max_val <= tol:
                lead += 1
                continue

            if pivot_row != r:
                A[r], A[pivot_row] = A[pivot_row], A[r]

            pivot = A[r][lead]
            for i in range(r + 1, self.m):
                factor = A[i][lead] / pivot
                for j in range(lead, self.n):
                    A[i][j] -= factor * A[r][j]

            lead += 1
            r += 1

        rank_count = 0
        for i in range(self.m):
            if any(abs(x) > tol for x in A[i]):
                rank_count += 1
        return rank_count

    def norm(self, ord: Union[int, float, str] = "fro") -> float:
        """Compute the matrix or vector norm."""
        if self.m == 0 or self.n == 0:
            return 0.0

        if ord == "fro":
            return math.sqrt(sum(abs(x) ** 2 for row in self.M for x in row))
        elif ord == 1:
            return float(
                max(
                    sum(abs(self.M[r][c]) for r in range(self.m)) for c in range(self.n)
                )
            )
        elif ord in (float("inf"), "inf"):
            return float(max(sum(abs(x) for x in row) for row in self.M))
        elif ord == -1:
            return float(
                min(
                    sum(abs(self.M[r][c]) for r in range(self.m)) for c in range(self.n)
                )
            )
        elif ord in (float("-inf"), "-inf"):
            return float(min(sum(abs(x) for x in row) for row in self.M))
        elif ord == 2:
            if self.m == 1 or self.n == 1:
                return math.sqrt(sum(abs(x) ** 2 for row in self.M for x in row))

            AtA = self.T @ self
            v = [1.0] * AtA.n
            for _ in range(50):
                w = [
                    sum(float(AtA.M[i][j]) * v[j] for j in range(AtA.n))
                    for i in range(AtA.n)
                ]
                norm_w = math.sqrt(sum(x**2 for x in w))
                if norm_w < 1e-15:
                    return 0.0
                v = [x / norm_w for x in w]
            w = [
                sum(float(AtA.M[i][j]) * v[j] for j in range(AtA.n))
                for i in range(AtA.n)
            ]
            lambda_max = max(0.0, sum(v[i] * w[i] for i in range(AtA.n)))
            return math.sqrt(lambda_max)
        else:
            raise ValueError(f"Invalid norm order: {ord}")

    def kron(self, other: "Matrix") -> "Matrix":
        """Compute the Kronecker product of two matrices."""
        if not isinstance(other, Matrix):
            raise TypeError("Kronecker product requires another Matrix")

        out_m = self.m * other.m
        out_n = self.n * other.n
        result = Matrix(out_m, out_n, dtype=self.dtype)

        for r1 in range(self.m):
            a_val = self.M[r1]
            for c1 in range(self.n):
                val = a_val[c1]
                for r2 in range(other.m):
                    b_row = other.M[r2]
                    for c2 in range(other.n):
                        result.M[r1 * other.m + r2][c1 * other.n + c2] = self.cast(
                            val * b_row[c2]
                        )

        return result

    # =========================================================================
    # Transformations: Reshape, Squeeze, Expand Dims
    # =========================================================================

    def reshape(self, m: int, n: int) -> "Matrix":
        """Reshape matrix into shape (m, n)."""
        if type(m) is not int or type(n) is not int:
            raise TypeError("Dimensions m and n must be integers")
        if m < 0 or n < 0:
            raise ValueError("Dimensions m and n must be non-negative integers")
        if m * n != self.m * self.n:
            raise ShapeError(
                f"Cannot reshape matrix of size {self.m * self.n} into ({m}, {n})"
            )
        flat = self.flatten()
        data = [flat[i * n : (i + 1) * n] for i in range(m)]
        return Matrix(matrix=data, dtype=self.dtype)

    def squeeze(
        self, axis: Optional[int] = None
    ) -> Union["Matrix", List[TinyMatrixNumeric], TinyMatrixNumeric]:
        """Remove single-dimensional entries from the shape."""
        if axis is not None:
            if axis not in (0, 1):
                raise ValueError(f"Invalid axis {axis}: must be 0 or 1")
            if axis == 0:
                if self.m != 1:
                    raise ValueError("Cannot squeeze axis 0 with dimension size != 1")
                return self.flatten()
            else:
                if self.n != 1:
                    raise ValueError("Cannot squeeze axis 1 with dimension size != 1")
                return self.flatten()

        if self.m == 1 and self.n == 1:
            return self.M[0][0]
        elif self.m == 1 or self.n == 1:
            return self.flatten()
        return self.copy()

    @staticmethod
    def expand_dims(
        data: Union["Matrix", Sequence[TinyMatrixNumeric]], axis: int
    ) -> "Matrix":
        """Expand the shape of a vector or 1D list into a 2D Matrix."""
        if axis not in (0, 1):
            raise ValueError(f"Invalid axis {axis}: must be 0 or 1")
        if isinstance(data, Matrix):
            if axis == 0:
                return data.reshape(1, data.m * data.n)
            else:
                return data.reshape(data.m * data.n, 1)
        elif isinstance(data, (list, tuple)):
            if axis == 0:
                return Matrix(matrix=[list(data)])
            else:
                return Matrix(matrix=[[x] for x in data])
        else:
            raise TypeError("Expected Matrix or sequence")

    # =========================================================================
    # Apply & Element-Wise Math Functions
    # =========================================================================

    def apply(self, func: Callable[[TinyMatrixNumeric], TinyMatrixNumeric]) -> "Matrix":
        """Apply a function element-wise and return a new Matrix."""
        data = [[self.cast(func(x)) for x in row] for row in self.M]
        return Matrix(matrix=data, dtype=self.dtype)

    def abs(self) -> "Matrix":
        """Element-wise absolute value."""
        return self.apply(abs)

    def __abs__(self) -> "Matrix":
        return self.abs()

    def sqrt(self) -> "Matrix":
        """Element-wise square root."""
        if self.dtype == "complex":
            return self.apply(cmath.sqrt)
        return self.apply(math.sqrt)

    def exp(self) -> "Matrix":
        """Element-wise exponential."""
        if self.dtype == "complex":
            return self.apply(cmath.exp)
        return self.apply(math.exp)

    def log(self) -> "Matrix":
        """Element-wise natural logarithm."""
        if self.dtype == "complex":
            return self.apply(cmath.log)
        return self.apply(math.log)

    def sin(self) -> "Matrix":
        """Element-wise sine."""
        if self.dtype == "complex":
            return self.apply(cmath.sin)
        return self.apply(math.sin)

    def cos(self) -> "Matrix":
        """Element-wise cosine."""
        if self.dtype == "complex":
            return self.apply(cmath.cos)
        return self.apply(math.cos)

    # =========================================================================
    # Reductions with Axis
    # =========================================================================

    def sum(self, axis: Optional[int] = None) -> Union[TinyMatrixNumeric, "Matrix"]:
        """Sum of matrix elements along an axis or overall."""
        if axis is None:
            if self.m == 0 or self.n == 0:
                return self.cast(0)
            return self.cast(sum(self.flatten()))
        elif axis == 0:
            if self.m == 0:
                return Matrix(1, self.n, dtype=self.dtype)
            sums = [
                self.cast(sum(self.M[r][c] for r in range(self.m)))
                for c in range(self.n)
            ]
            return Matrix(matrix=[sums], dtype=self.dtype)
        elif axis == 1:
            if self.n == 0:
                return Matrix(self.m, 1, dtype=self.dtype)
            sums = [[self.cast(sum(row))] for row in self.M]
            return Matrix(matrix=sums, dtype=self.dtype)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be None, 0, or 1")

    def mean(self, axis: Optional[int] = None) -> Union[TinyMatrixNumeric, "Matrix"]:
        """Arithmetic mean of matrix elements along an axis or overall."""
        if self.m == 0 or self.n == 0:
            raise ValueError("Cannot compute mean of empty matrix")
        if axis is None:
            total = sum(self.flatten())
            count = self.m * self.n
            return self.cast(total / count)
        elif axis == 0:
            sums = [
                sum(self.M[r][c] for r in range(self.m)) / self.m for c in range(self.n)
            ]
            return Matrix(matrix=[[self.cast(x) for x in sums]], dtype=self.dtype)
        elif axis == 1:
            sums = [[self.cast(sum(row) / self.n)] for row in self.M]
            return Matrix(matrix=sums, dtype=self.dtype)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be None, 0, or 1")

    def min(self, axis: Optional[int] = None) -> Union[TinyMatrixNumeric, "Matrix"]:
        """Minimum value along an axis or overall."""
        if self.m == 0 or self.n == 0:
            raise ValueError("Cannot compute min of empty matrix")
        if self.dtype == "complex":
            raise TypeError("min is not supported for complex numbers")
        if axis is None:
            return self.cast(min(self.flatten()))
        elif axis == 0:
            mins = [min(self.M[r][c] for r in range(self.m)) for c in range(self.n)]
            return Matrix(matrix=[[self.cast(x) for x in mins]], dtype=self.dtype)
        elif axis == 1:
            mins = [[self.cast(min(row))] for row in self.M]
            return Matrix(matrix=mins, dtype=self.dtype)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be None, 0, or 1")

    def max(self, axis: Optional[int] = None) -> Union[TinyMatrixNumeric, "Matrix"]:
        """Maximum value along an axis or overall."""
        if self.m == 0 or self.n == 0:
            raise ValueError("Cannot compute max of empty matrix")
        if self.dtype == "complex":
            raise TypeError("max is not supported for complex numbers")
        if axis is None:
            return self.cast(max(self.flatten()))
        elif axis == 0:
            maxs = [max(self.M[r][c] for r in range(self.m)) for c in range(self.n)]
            return Matrix(matrix=[[self.cast(x) for x in maxs]], dtype=self.dtype)
        elif axis == 1:
            maxs = [[self.cast(max(row))] for row in self.M]
            return Matrix(matrix=maxs, dtype=self.dtype)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be None, 0, or 1")

    def std(
        self, axis: Optional[int] = None, ddof: int = 0
    ) -> Union[TinyMatrixNumeric, "Matrix"]:
        """Standard deviation along an axis or overall."""
        if self.m == 0 or self.n == 0:
            raise ValueError("Cannot compute std of empty matrix")
        if self.dtype == "complex":
            raise TypeError("std is not supported for complex numbers")

        def _calc_std(vals: List[TinyMatrixNumeric]) -> float:
            n = len(vals)
            if n <= ddof:
                raise ValueError("Degrees of freedom must be less than sample size")
            mean_val = float(sum(vals) / n)
            variance = sum((float(x) - mean_val) ** 2 for x in vals) / (n - ddof)
            return math.sqrt(variance)

        if axis is None:
            return self.cast(_calc_std(self.flatten()))
        elif axis == 0:
            stds = [
                _calc_std([self.M[r][c] for r in range(self.m)]) for c in range(self.n)
            ]
            return Matrix(matrix=[[self.cast(x) for x in stds]], dtype=self.dtype)
        elif axis == 1:
            stds = [[self.cast(_calc_std(row))] for row in self.M]
            return Matrix(matrix=stds, dtype=self.dtype)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be None, 0, or 1")

    # =========================================================================
    # String Representation & Formatting
    # =========================================================================

    def __repr__(self) -> str:
        if self.m == 0 or self.n == 0:
            return "\n".join("[]" for _ in range(self.m))

        # Check if small matrix (<= 10x10) to format cleanly with existing test contract
        if self.m <= 10 and self.n <= 10:
            return "\n".join(
                "[" + " ".join(str(x) for x in row) + "]" for row in self.M
            )

        # Truncated representation for large matrices
        row_indices = (
            list(range(3)) + [-1] + list(range(self.m - 3, self.m))
            if self.m > 10
            else list(range(self.m))
        )
        lines = []
        for r in row_indices:
            if r == -1:
                lines.append("...")
                continue
            row = self.M[r]
            if self.n > 10:
                first = " ".join(str(row[c]) for c in range(3))
                last = " ".join(str(row[c]) for c in range(self.n - 3, self.n))
                lines.append(f"[{first} ... {last}]")
            else:
                lines.append("[" + " ".join(str(x) for x in row) + "]")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    # =========================================================================
    # Arithmetic & Broadcasting
    # =========================================================================

    @staticmethod
    def _broadcast_shapes(
        s1: Tuple[int, int], s2: Tuple[int, int], op_name: str
    ) -> Tuple[int, int]:
        m1, n1 = s1
        m2, n2 = s2
        if (m1 == 0 or n1 == 0 or m2 == 0 or n2 == 0) and (m1, n1) != (m2, n2):
            raise ShapeError(f"Matrix size mismatch for {op_name}")
        if m1 != m2 and m1 != 1 and m2 != 1:
            raise ShapeError(f"Matrix size mismatch for {op_name}")
        if n1 != n2 and n1 != 1 and n2 != 1:
            raise ShapeError(f"Matrix size mismatch for {op_name}")
        return max(m1, m2), max(n1, n2)

    def _elementwise_op(
        self,
        other: Union["Matrix", TinyMatrixNumeric],
        op: Callable[[TinyMatrixNumeric, TinyMatrixNumeric], TinyMatrixNumeric],
        op_name: str,
    ) -> "Matrix":
        if isinstance(other, Matrix):
            out_m, out_n = self._broadcast_shapes(self.shape(), other.shape(), op_name)
            result = Matrix(out_m, out_n, dtype=self.dtype)
            m1, n1 = self.shape()
            m2, n2 = other.shape()
            for r in range(out_m):
                r1 = r if m1 > 1 else 0
                r2 = r if m2 > 1 else 0
                for c in range(out_n):
                    c1 = c if n1 > 1 else 0
                    c2 = c if n2 > 1 else 0
                    a = self.cast(self.M[r1][c1])
                    b = self.cast(other.M[r2][c2])
                    result.M[r][c] = self.cast(op(a, b))
            return result
        elif self._is_valid_scalar(other):
            scalar_val = self.cast(other)
            result = Matrix(self.m, self.n, dtype=self.dtype)
            for r in range(self.m):
                for c in range(self.n):
                    a = self.cast(self.M[r][c])
                    result.M[r][c] = self.cast(op(a, scalar_val))
            return result
        else:
            raise TypeError(
                f"Unsupported operand type for {op_name}: {type(other).__name__}"
            )

    def __add__(self, other: Union["Matrix", TinyMatrixNumeric]) -> "Matrix":
        if not isinstance(other, Matrix) and not self._is_valid_scalar(other):
            raise TypeError("Matrix can only be added to a matrix or scalar")
        return self._elementwise_op(other, lambda a, b: a + b, "addition")

    def __radd__(self, other: TinyMatrixNumeric) -> "Matrix":
        return self.__add__(other)

    def __iadd__(self, other: Union["Matrix", TinyMatrixNumeric]) -> "Matrix":
        res = self.__add__(other)
        self.M = res.M
        self.m = res.m
        self.n = res.n
        return self

    def __sub__(self, other: Union["Matrix", TinyMatrixNumeric]) -> "Matrix":
        if not isinstance(other, Matrix) and not self._is_valid_scalar(other):
            raise TypeError("Matrix can only be subtracted from a matrix or scalar")
        return self._elementwise_op(other, lambda a, b: a - b, "subtraction")

    def __rsub__(self, other: TinyMatrixNumeric) -> "Matrix":
        if not self._is_valid_scalar(other):
            raise TypeError("Matrix can only be subtracted from a matrix or scalar")
        other_mat = Matrix.ones(self.m, self.n, dtype=self.dtype) * other
        return other_mat - self

    def __isub__(self, other: Union["Matrix", TinyMatrixNumeric]) -> "Matrix":
        res = self.__sub__(other)
        self.M = res.M
        self.m = res.m
        self.n = res.n
        return self

    def __mul__(self, other: Union["Matrix", TinyMatrixNumeric]) -> "Matrix":
        if not isinstance(other, Matrix) and not self._is_valid_scalar(other):
            raise TypeError("Matrix can only be multiplied by a scalar")
        return self._elementwise_op(other, lambda a, b: a * b, "multiplication")

    def __rmul__(self, scalar: TinyMatrixNumeric) -> "Matrix":
        return self.__mul__(scalar)

    def __imul__(self, other: Union["Matrix", TinyMatrixNumeric]) -> "Matrix":
        res = self.__mul__(other)
        self.M = res.M
        self.m = res.m
        self.n = res.n
        return self

    def __truediv__(self, other: Union["Matrix", TinyMatrixNumeric]) -> "Matrix":
        def _div(a, b):
            if b == 0:
                raise ZeroDivisionError("division by zero")
            return a / b

        if not isinstance(other, Matrix) and not self._is_valid_scalar(other):
            raise TypeError("Matrix can only be divided by a scalar or matrix")
        return self._elementwise_op(other, _div, "division")

    def __rtruediv__(self, scalar: TinyMatrixNumeric) -> "Matrix":
        if not self._is_valid_scalar(scalar):
            raise TypeError("Matrix can only be divided by a scalar")
        scalar_val = self.cast(scalar)
        result = Matrix(self.m, self.n, dtype=self.dtype)
        for r in range(self.m):
            for c in range(self.n):
                val = self.M[r][c]
                if val == 0:
                    raise ZeroDivisionError("division by zero")
                result.M[r][c] = self.cast(scalar_val / val)
        return result

    def __itruediv__(self, other: Union["Matrix", TinyMatrixNumeric]) -> "Matrix":
        res = self.__truediv__(other)
        self.M = res.M
        self.m = res.m
        self.n = res.n
        return self

    def __neg__(self) -> "Matrix":
        return self * -1

    def __pow__(self, power: int) -> "Matrix":
        if type(power) is not int or power < 0:
            raise ValueError("Matrix power requires a non-negative integer")
        if not self.is_square():
            raise ShapeError("Matrix power requires a square matrix")

        if power == 0:
            return Matrix.identity(self.m, dtype=self.dtype)
        elif power == 1:
            return self.copy()

        result = Matrix.identity(self.m, dtype=self.dtype)
        base = self.copy()
        while power > 0:
            if power % 2 == 1:
                result = result @ base
            base = base @ base
            power //= 2
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return False
        if self.shape() != other.shape():
            return False

        for row in range(self.m):
            for col in range(self.n):
                if self.M[row][col] != other.M[row][col]:
                    return False

        return True

    def __matmul__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            raise TypeError(
                f"Unsupported operand type for @: 'Matrix' and '{type(other).__name__}'"
            )
        if self.n != other.m:
            raise ShapeError(
                f"Cannot multiply: ({self.m}*{self.n}) @ ({other.m}*{other.n})"
            )

        result = Matrix(self.m, other.n, dtype=self.dtype)
        for row in range(self.m):
            for k in range(self.n):
                aik = self.cast(self.M[row][k])
                for col in range(other.n):
                    result.M[row][col] = self.cast(
                        result.M[row][col] + aik * other.M[k][col]
                    )

        return result

    @property
    def T(self) -> "Matrix":
        result = Matrix(self.n, self.m, dtype=self.dtype)
        for row in range(self.m):
            for col in range(self.n):
                result.M[col][row] = self.M[row][col]

        return result

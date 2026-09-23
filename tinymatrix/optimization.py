"""Performance optimizations, blocked/Strassen matmul, and lazy evaluation for TinyMatrix."""

from typing import Any, Callable, Tuple, Union

from .exceptions import ShapeError
from .matrix import Matrix
from .types import TinyMatrixNumeric


def matmul_blocked(A: Matrix, B: Matrix, block_size: int = 32) -> Matrix:
    """Compute matrix multiplication using cache-friendly loop tiling/blocking."""
    if A.n != B.m:
        raise ShapeError(f"Cannot multiply: ({A.m}*{A.n}) @ ({B.m}*{B.n})")

    M, K, N = A.m, A.n, B.n
    result = Matrix.zeroes(M, N, dtype=A.dtype)

    for i0 in range(0, M, block_size):
        i_max = min(i0 + block_size, M)
        for k0 in range(0, K, block_size):
            k_max = min(k0 + block_size, K)
            for j0 in range(0, N, block_size):
                j_max = min(j0 + block_size, N)

                for i in range(i0, i_max):
                    for k in range(k0, k_max):
                        aik = A.M[i][k]
                        for j in range(j0, j_max):
                            result.M[i][j] = result.cast(
                                result.M[i][j] + aik * B.M[k][j]
                            )

    return result


def _pad_to_pow2(A: Matrix, target_size: int) -> Matrix:
    padded = Matrix.zeroes(target_size, target_size, dtype=A.dtype)
    for r in range(A.m):
        for c in range(A.n):
            padded.M[r][c] = A.M[r][c]
    return padded


def matmul_strassen(A: Matrix, B: Matrix, threshold: int = 32) -> Matrix:
    """Compute matrix multiplication using Strassen's algorithm for square matrices."""
    if A.n != B.m:
        raise ShapeError(f"Cannot multiply: ({A.m}*{A.n}) @ ({B.m}*{B.n})")

    M, K, N = A.m, A.n, B.n
    if M <= threshold or K <= threshold or N <= threshold:
        return A @ B

    # Find next power of 2
    max_dim = max(M, K, N)
    n2 = 1
    while n2 < max_dim:
        n2 *= 2

    # Pad matrices to power of 2
    A_pad = _pad_to_pow2(A, n2)
    B_pad = _pad_to_pow2(B, n2)

    def _strassen_rec(X: Matrix, Y: Matrix) -> Matrix:
        n = X.m
        if n <= threshold:
            return X @ Y

        mid = n // 2
        A11 = X[0:mid, 0:mid]
        A12 = X[0:mid, mid:n]
        A21 = X[mid:n, 0:mid]
        A22 = X[mid:n, mid:n]

        B11 = Y[0:mid, 0:mid]
        B12 = Y[0:mid, mid:n]
        B21 = Y[mid:n, 0:mid]
        B22 = Y[mid:n, mid:n]

        # Strassen's 7 products
        M1 = _strassen_rec(A11 + A22, B11 + B22)
        M2 = _strassen_rec(A21 + A22, B11)
        M3 = _strassen_rec(A11, B12 - B22)
        M4 = _strassen_rec(A22, B21 - B11)
        M5 = _strassen_rec(A11 + A12, B22)
        M6 = _strassen_rec(A21 - A11, B11 + B12)
        M7 = _strassen_rec(A12 - A22, B21 + B22)

        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6

        # Assemble result
        out = Matrix.zeroes(n, n, dtype=X.dtype)
        for r in range(mid):
            for c in range(mid):
                out.M[r][c] = C11.M[r][c]
                out.M[r][c + mid] = C12.M[r][c]
                out.M[r + mid][c] = C21.M[r][c]
                out.M[r + mid][c + mid] = C22.M[r][c]

        return out

    res_pad = _strassen_rec(A_pad, B_pad)
    # Unpad to original M x N
    return res_pad[0:M, 0:N]


class LazyMatrix:
    """Lazy evaluation tree for chained matrix operations."""

    def __init__(
        self,
        eval_fn: Callable[[], Matrix],
        shape: Tuple[int, int],
        dtype: str = "float",
    ) -> None:
        self._eval_fn = eval_fn
        self._shape = shape
        self.dtype = dtype

    @classmethod
    def from_matrix(cls, mat: Matrix) -> "LazyMatrix":
        return cls(lambda: mat.copy(), mat.shape(), mat.dtype)

    def shape(self) -> Tuple[int, int]:
        return self._shape

    def evaluate(self) -> Matrix:
        """Evaluate the expression graph and return a concrete Matrix."""
        return self._eval_fn()

    def eval(self) -> Matrix:
        return self.evaluate()

    def __add__(self, other: Union["LazyMatrix", Matrix]) -> "LazyMatrix":
        other_lazy = (
            other if isinstance(other, LazyMatrix) else LazyMatrix.from_matrix(other)
        )
        return LazyMatrix(
            lambda: self.evaluate() + other_lazy.evaluate(),
            self._shape,
            self.dtype,
        )

    def __sub__(self, other: Union["LazyMatrix", Matrix]) -> "LazyMatrix":
        other_lazy = (
            other if isinstance(other, LazyMatrix) else LazyMatrix.from_matrix(other)
        )
        return LazyMatrix(
            lambda: self.evaluate() - other_lazy.evaluate(),
            self._shape,
            self.dtype,
        )

    def __matmul__(self, other: Union["LazyMatrix", Matrix]) -> "LazyMatrix":
        other_lazy = (
            other if isinstance(other, LazyMatrix) else LazyMatrix.from_matrix(other)
        )
        new_shape = (self._shape[0], other_lazy._shape[1])
        return LazyMatrix(
            lambda: self.evaluate() @ other_lazy.evaluate(),
            new_shape,
            self.dtype,
        )

    def __mul__(self, scalar: TinyMatrixNumeric) -> "LazyMatrix":
        return LazyMatrix(
            lambda: self.evaluate() * scalar,
            self._shape,
            self.dtype,
        )

    def __rmul__(self, scalar: TinyMatrixNumeric) -> "LazyMatrix":
        return self.__mul__(scalar)

    @property
    def T(self) -> "LazyMatrix":
        return LazyMatrix(
            lambda: self.evaluate().T,
            (self._shape[1], self._shape[0]),
            self.dtype,
        )


def to_numpy(A: Matrix) -> Any:
    """Convert TinyMatrix to NumPy array if numpy is installed."""
    try:
        import numpy as np

        return np.array(A.M, dtype=A.dtype if A.dtype != "decimal" else object)
    except ImportError as e:
        raise ImportError("NumPy is not installed") from e


def from_numpy(arr: Any) -> Matrix:
    """Create a TinyMatrix from a NumPy array."""
    data = arr.tolist()
    dtype = "float"
    if hasattr(arr, "dtype"):
        if "int" in str(arr.dtype):
            dtype = "int"
        elif "complex" in str(arr.dtype):
            dtype = "complex"
    return Matrix(matrix=data, dtype=dtype)

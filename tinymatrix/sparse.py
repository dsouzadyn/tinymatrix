"""Sparse matrix implementations (CSR and CSC) for TinyMatrix."""

from typing import List, Tuple, Union

from .exceptions import ShapeError
from .matrix import Matrix
from .types import TinyMatrixNumeric


class CSRMatrix:
    """Compressed Sparse Row (CSR) matrix format."""

    def __init__(
        self,
        values: List[TinyMatrixNumeric],
        col_indices: List[int],
        row_offsets: List[int],
        shape: Tuple[int, int],
        dtype: str = "float",
    ) -> None:
        self.values = values
        self.col_indices = col_indices
        self.row_offsets = row_offsets
        self.m, self.n = shape
        self.dtype = dtype

    @classmethod
    def from_dense(cls, A: Matrix) -> "CSRMatrix":
        """Construct CSRMatrix from a dense Matrix."""
        values = []
        col_indices = []
        row_offsets = [0]

        for r in range(A.m):
            for c in range(A.n):
                val = A.M[r][c]
                if abs(complex(val)) > 1e-15:
                    values.append(val)
                    col_indices.append(c)
            row_offsets.append(len(values))

        return cls(values, col_indices, row_offsets, A.shape(), A.dtype)

    def to_dense(self) -> Matrix:
        """Convert CSRMatrix to dense Matrix."""
        mat = Matrix.zeroes(self.m, self.n, dtype=self.dtype)
        for r in range(self.m):
            start = self.row_offsets[r]
            end = self.row_offsets[r + 1]
            for idx in range(start, end):
                c = self.col_indices[idx]
                mat.M[r][c] = mat.cast(self.values[idx])
        return mat

    def shape(self) -> Tuple[int, int]:
        return self.m, self.n

    def nnz(self) -> int:
        """Number of non-zero stored elements."""
        return len(self.values)

    def __getitem__(self, idx: Tuple[int, int]) -> TinyMatrixNumeric:
        r, c = idx
        if not (0 <= r < self.m and 0 <= c < self.n):
            raise IndexError("Index out of bounds")
        start = self.row_offsets[r]
        end = self.row_offsets[r + 1]
        for i in range(start, end):
            if self.col_indices[i] == c:
                return self.values[i]
        return 0.0

    def __matmul__(self, other: Union[Matrix, "CSRMatrix"]) -> Matrix:
        """Sparse-dense or sparse-sparse matrix multiplication."""
        if isinstance(other, CSRMatrix):
            other_dense = other.to_dense()
        else:
            other_dense = other

        if self.n != other_dense.m:
            raise ShapeError(
                f"Cannot multiply: ({self.m}*{self.n}) @ ({other_dense.m}*{other_dense.n})"
            )

        res = Matrix.zeroes(self.m, other_dense.n, dtype=self.dtype)
        for r in range(self.m):
            start = self.row_offsets[r]
            end = self.row_offsets[r + 1]
            for idx in range(start, end):
                c = self.col_indices[idx]
                val = self.values[idx]
                for j in range(other_dense.n):
                    res.M[r][j] = res.cast(res.M[r][j] + val * other_dense.M[c][j])

        return res

    def to_csc(self) -> "CSCMatrix":
        return CSCMatrix.from_dense(self.to_dense())


class CSCMatrix:
    """Compressed Sparse Column (CSC) matrix format."""

    def __init__(
        self,
        values: List[TinyMatrixNumeric],
        row_indices: List[int],
        col_offsets: List[int],
        shape: Tuple[int, int],
        dtype: str = "float",
    ) -> None:
        self.values = values
        self.row_indices = row_indices
        self.col_offsets = col_offsets
        self.m, self.n = shape
        self.dtype = dtype

    @classmethod
    def from_dense(cls, A: Matrix) -> "CSCMatrix":
        """Construct CSCMatrix from a dense Matrix."""
        values = []
        row_indices = []
        col_offsets = [0]

        for c in range(A.n):
            for r in range(A.m):
                val = A.M[r][c]
                if abs(complex(val)) > 1e-15:
                    values.append(val)
                    row_indices.append(r)
            col_offsets.append(len(values))

        return cls(values, row_indices, col_offsets, A.shape(), A.dtype)

    def to_dense(self) -> Matrix:
        """Convert CSCMatrix to dense Matrix."""
        mat = Matrix.zeroes(self.m, self.n, dtype=self.dtype)
        for c in range(self.n):
            start = self.col_offsets[c]
            end = self.col_offsets[c + 1]
            for idx in range(start, end):
                r = self.row_indices[idx]
                mat.M[r][c] = mat.cast(self.values[idx])
        return mat

    def shape(self) -> Tuple[int, int]:
        return self.m, self.n

    def nnz(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: Tuple[int, int]) -> TinyMatrixNumeric:
        r, c = idx
        if not (0 <= r < self.m and 0 <= c < self.n):
            raise IndexError("Index out of bounds")
        start = self.col_offsets[c]
        end = self.col_offsets[c + 1]
        for i in range(start, end):
            if self.row_indices[i] == r:
                return self.values[i]
        return 0.0

    def to_csr(self) -> "CSRMatrix":
        return CSRMatrix.from_dense(self.to_dense())

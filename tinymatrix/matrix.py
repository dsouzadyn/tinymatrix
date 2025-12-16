from typing import List, Optional, Tuple, Union, overload

from .exceptions import ShapeError
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
            if not all(len(row) == len(matrix[0]) for row in matrix):
                raise ShapeError("All rows must have equal length")

            self.M = [[self.cast(x) for x in row] for row in matrix]
            self.m = len(matrix)
            self.n = len(matrix[0])
        else:
            if m is None or n is None:
                raise ValueError("Provide either matrix or (m, n)")
            self.m = m
            self.n = n
            self.M = [[self.cast(0) for _ in range(n)] for _ in range(m)]

    def _is_valid_scalar(self, value):
        try:
            self.cast(value)
            return True
        except Exception:
            return False

    @staticmethod
    def zeroes(m: int, n: int, dtype: str = "float") -> "Matrix":
        return Matrix(m, n, dtype=dtype)

    @staticmethod
    def ones(m: int, n: int, dtype: str = "float") -> "Matrix":
        mat = Matrix(m, n, dtype=dtype)
        for row in range(m):
            for col in range(n):
                mat[row, col] = mat.cast(1)

        return mat

    @staticmethod
    def identity(n: int, dtype: str = "float") -> "Matrix":
        mat = Matrix(n, n, dtype=dtype)
        for row in range(n):
            mat[row, row] = mat.cast(1)

        return mat

    def shape(self) -> Tuple[int, int]:
        return self.m, self.n

    def copy(self) -> "Matrix":
        return Matrix(matrix=[row[:] for row in self.M], dtype=self.dtype)

    def astype(self, dtype: str) -> "Matrix":
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")

        cast = DTYPES[dtype]
        data = [[cast(x) for x in row] for row in self.M]
        return Matrix(matrix=data, dtype=dtype)

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

                data = [[self.M[row][col] for col in cols] for row in rows]

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

    def __repr__(self) -> str:
        return "\n".join("[" + " ".join(str(x) for x in row) + "]" for row in self.M)

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, other: "Matrix") -> "Matrix":
        if self.shape() != other.shape():
            raise ShapeError("Matrix size mismatch for addition")

        result = Matrix(self.m, self.n, dtype=self.dtype)
        for row in range(self.m):
            for col in range(self.n):
                a = self.cast(self[row, col])
                b = self.cast(other[row, col])
                result[row, col] = self.cast(a + b)

        return result

    def __sub__(self, other: "Matrix") -> "Matrix":
        if self.shape() != other.shape():
            raise ShapeError("Matrix size mismatch for subtraction")

        result = Matrix(self.m, self.n, dtype=self.dtype)
        for row in range(self.m):
            for col in range(self.n):
                a = self.cast(self[row, col])
                b = self.cast(other[row, col])
                result[row, col] = self.cast(a - b)

        return result

    def __mul__(self, scalar: TinyMatrixNumeric) -> "Matrix":
        if not self._is_valid_scalar(scalar):
            raise TypeError("Matrix can only be multiplied by a scalar")

        result = Matrix(self.m, self.n, dtype=self.dtype)
        scalar_cast = self.cast(scalar)
        for row in range(self.m):
            for col in range(self.n):
                result[row, col] = self.cast(self[row, col]) * scalar_cast

        return result

    def __rmul__(self, scalar: TinyMatrixNumeric) -> "Matrix":
        return self.__mul__(scalar)

    def __neg__(self) -> "Matrix":
        return self * -1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return False
        if self.shape() != other.shape():
            return False

        for row in range(self.m):
            for col in range(self.n):
                if self[row, col] != other[row, col]:
                    return False

        return True

    def __matmul__(self, other: "Matrix") -> "Matrix":
        if self.n != other.m:
            raise ShapeError(
                f"Cannot multiply: ({self.m}*{self.n}) @ ({other.m}*{other.n})"
            )

        result = Matrix(self.m, other.n, dtype=self.dtype)
        for row in range(self.m):
            for k in range(self.n):
                aik = self.cast(self[row, k])
                for col in range(other.n):
                    result[row, col] = self.cast(result[row, col] + aik * other[k, col])

        return result

    @property
    def T(self) -> "Matrix":
        result = Matrix(self.n, self.m, dtype=self.dtype)
        for row in range(self.m):
            for col in range(self.n):
                result[col, row] = self[row, col]

        return result

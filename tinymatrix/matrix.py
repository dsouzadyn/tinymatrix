from .exceptions import ShapeError


class Matrix:
    def __init__(self, m=None, n=None, matrix=None):
        if matrix is not None:
            if not all(len(row) == len(matrix[0]) for row in matrix):
                raise ShapeError("All rows must have equal length")

            self.M = [list(row) for row in matrix]
            self.m = len(matrix)
            self.n = len(matrix[0])
        else:
            if m is None or n is None:
                raise ValueError("Provide either matrix or (m, n)")
            self.m = m
            self.n = n
            self.M = [[0.0 for _ in range(n)] for _ in range(m)]

    @staticmethod
    def zeroes(m, n):
        return Matrix(m, n)

    @staticmethod
    def ones(m, n):
        mat = Matrix(m, n)
        for row in range(m):
            for col in range(n):
                mat[row, col] = 1.0

        return mat

    @staticmethod
    def identity(n):
        mat = Matrix(n, n)
        for row in range(n):
            mat[row, row] = 1.0

        return mat

    def shape(self):
        return self.m, self.n

    def copy(self):
        return Matrix(matrix=[row[:] for row in self.M])

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            row, col = idx

            if isinstance(row, slice) or isinstance(col, slice):
                rows = range(*row.indices(self.m)) if isinstance(row, slice) else [row]
                cols = range(*col.indices(self.n)) if isinstance(col, slice) else [col]

                data = [[self.M[row][col] for col in cols] for row in rows]

                return Matrix(matrix=data)

            return self.M[row][col]
        elif isinstance(idx, int):
            return Matrix(matrix=[self.M[idx][:]])
        else:
            raise TypeError("Invalid index type")

    def __setitem__(self, idx, value):
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
                        self.M[ri][cj] = value.M[i][j]
                return

            if not isinstance(value, (int, float)):
                raise TypeError("Single element assignment requires a number")

            self.M[row][col] = float(value)
            return

        elif isinstance(idx, int):
            if not isinstance(value, (list, tuple)):
                raise TypeError("Row assignment requires a list or tuple")
            if len(value) != self.n:
                raise ShapeError("Row length mismatch")

            self.M[idx] = list(value)
            return
        else:
            raise TypeError("Invalid index type")

    def __repr__(self):
        return "\n".join(
            "[" + " ".join(f"{x:.2f}" for x in row) + "]" for row in self.M
        )

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        if self.shape() != other.shape():
            raise ShapeError("Matrix size mismatch for addition")

        result = Matrix(self.m, self.n)
        for row in range(self.m):
            for col in range(self.n):
                result[row, col] = self[row, col] + other[row, col]

        return result

    def __sub__(self, other):
        if self.shape() != other.shape():
            raise ShapeError("Matrix size mismatch for subtraction")

        result = Matrix(self.m, self.n)
        for row in range(self.m):
            for col in range(self.n):
                result[row, col] = self[row, col] - other[row, col]

        return result

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Matrix can only be multiplied by a scalar")

        result = Matrix(self.m, self.n)
        for row in range(self.m):
            for col in range(self.n):
                result[row, col] = self[row, col] * scalar

        return result

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        return self * -1

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if self.shape() != other.shape():
            return False

        for row in range(self.m):
            for col in range(self.n):
                if self[row, col] != other[row, col]:
                    return False

        return True

    def __matmul__(self, other):
        if self.n != other.m:
            raise ShapeError(
                f"Cannot multiply: ({self.m}*{self.n}) @ ({other.m}*{other.n})"
            )

        result = Matrix(self.m, other.n)
        for row in range(self.m):
            for k in range(self.n):
                aik = self[row, k]
                for col in range(other.n):
                    result[row, col] += aik * other[k, col]

        return result

    @property
    def T(self):
        result = Matrix(self.n, self.m)
        for row in range(self.m):
            for col in range(self.n):
                result[col, row] = self[row, col]

        return result

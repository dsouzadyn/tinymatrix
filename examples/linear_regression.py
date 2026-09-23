"""Linear regression with least squares using TinyMatrix."""

from tinymatrix import Matrix, lstsq


def run_linear_regression():
    # Fit line: y = m*x + c
    # Data points: (1, 2), (2, 2.8), (3, 3.6), (4, 4.5), (5, 5.1)
    x_coords = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_coords = [2.0, 2.8, 3.6, 4.5, 5.1]

    # Design matrix: A = [[x_i, 1]]
    A = Matrix(matrix=[[x, 1.0] for x in x_coords])
    b = Matrix(matrix=[[y] for y in y_coords])

    # Solve least squares: min ||A @ params - b||
    params, residual = lstsq(A, b)

    slope = params[0, 0]
    intercept = params[1, 0]

    print("Linear Regression Results:")
    print(f"  Slope (m):     {slope:.4f}")
    print(f"  Intercept (c): {intercept:.4f}")
    print(f"  Residual norm: {residual:.4f}")

    # Predictions
    preds = A @ params
    print("\nActual vs Predicted:")
    for actual, pred in zip(y_coords, preds.flatten(), strict=False):
        print(f"  Actual: {actual:.2f} | Predicted: {pred:.2f}")


if __name__ == "__main__":
    run_linear_regression()

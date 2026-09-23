"""1D/2D Kalman filter tracking simulation using TinyMatrix."""

from tinymatrix import Matrix


class KalmanFilter:
    def __init__(
        self, F: Matrix, H: Matrix, Q: Matrix, R: Matrix, P: Matrix, x: Matrix
    ):
        self.F = F  # State transition model
        self.H = H  # Observation model
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Estimate error covariance
        self.x = x  # Initial state estimate

    def predict(self):
        # x = F @ x
        self.x = self.F @ self.x
        # P = F @ P @ F.T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: Matrix):
        # Innovation y = z - H @ x
        y = z - self.H @ self.x
        # Innovation covariance S = H @ P @ H.T + R
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain K = P @ H.T @ S.inv()
        K = self.P @ self.H.T @ S.inv()
        # Updated state x = x + K @ y
        self.x = self.x + K @ y
        ident = Matrix.identity(self.P.m)
        self.P = (ident - K @ self.H) @ self.P


def run_kalman_demo():
    # 1D position and velocity: x = [pos, vel].T
    dt = 1.0
    F = Matrix(matrix=[[1.0, dt], [0.0, 1.0]])
    H = Matrix(matrix=[[1.0, 0.0]])  # Observe position only
    Q = Matrix(matrix=[[0.01, 0.0], [0.0, 0.01]])
    R = Matrix(matrix=[[0.5]])  # Measurement noise variance
    P = Matrix.identity(2)
    x = Matrix(matrix=[[0.0], [1.0]])  # Start at 0 with velocity 1

    kf = KalmanFilter(F, H, Q, R, P, x)

    # Simulated noisy GPS measurements
    measurements = [1.1, 2.3, 2.9, 4.2, 5.0, 5.8, 7.1, 8.2]

    print("Kalman Filter 1D Tracking:")
    for step, z_val in enumerate(measurements):
        kf.predict()
        z = Matrix(matrix=[[z_val]])
        kf.update(z)
        est_pos = kf.x[0, 0]
        est_vel = kf.x[1, 0]
        print(
            f"  Step {step + 1}: Meas = {z_val:.2f} | Est Pos = {est_pos:.2f} | Est Vel = {est_vel:.2f}"
        )


if __name__ == "__main__":
    run_kalman_demo()

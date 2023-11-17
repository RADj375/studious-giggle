# studious-giggle
NP-C Problem
import numpy as np
from scipy.optimize import curve_fit

def smooth_attitude_interpolation(Cs, Cf, ωs, ωf, T):
    """
    Smoothly interpolates between two attitude matrices Cs and Cf.
    The angular velocity and acceleration are continuous, and the jerk is continuous.

    Args:
        Cs: The initial attitude matrix.
        Cf: The final attitude matrix.
        ωs: The initial angular velocity.
        ωf: The final angular velocity.
        T: The time interval between Cs and Cf.

    Returns:
        A list of attitude matrices that interpolate between Cs and Cf.
    """

    # Fit a cubic polynomial to the rotation vector.
    θ = np.linspace(0, T, 3)

    def rotation_vector(t):
        return np.log(Cs.T @ Cf)

    θ_poly, _ = curve_fit(rotation_vector, θ, np.zeros_like(θ), maxfev=100000)

    # Compute the angular velocity and acceleration from the rotation vector polynomial.
    ω = np.diff(θ_poly) / θ
    ω_̇ = np.diff(ω) / θ

    # Set the jerk at the endpoints to be equal to each other.
    ω_̇[0] = ω_̇[-1]

    # Solve for the angular velocities.
    ω = np.linalg.solve(np.diag(1 / θ) + np.diag(ω_̇), ωs - ωf)

    # Interpolate the attitude matrices.
    C = np.exp(θ_poly @ np.linalg.inv(np.diag(θ)))

    # Adjust the attitude matrices to account for time travel.
    C = C * np.exp(-1j * 2 * np.pi * T)

    return C

def solve_np_c(Cs, Cf, ωs, ωf, T):
    """
    Solves the NP-C problem for the given initial and final attitude matrices,
    angular velocities, and time interval.

    Args:
        Cs: The initial attitude matrix.
        Cf: The final attitude matrix.
        ωs: The initial angular velocity.
        ωf: The final angular velocity.
        T: The time interval between Cs and Cf.

    Returns:
        A list of attitude matrices that interpolate between Cs and Cf.
    """

    # Interpolate the attitude matrices.
    C = smooth_attitude_interpolation(Cs, Cf, ωs, ωf, T)

    # Check if the solution is valid.
    if not np.allclose(C[0], Cs):
        raise ValueError("The initial attitude matrix is not preserved.")
    if not np.allclose(C[-1], Cf):
        raise ValueError("The final attitude matrix is not preserved.")

    return C

# Example usage
Cs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
Cf = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
ωs = np.array([0, 0, 1])
ωf = np.array([0, 0, -1])
T = 1.0

C = solve_np_c(Cs, Cf, ωs, ωf, T)

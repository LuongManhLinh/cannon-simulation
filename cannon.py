"""
Secant or Golden Section Search root finding to compute optimal angle
Runge-Kutta-based trajectory simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from golden_section_search import minimize_scalar_positive

# Physical constants
g = 9.81  # m/s²

def simulate_trajectory(angle_deg, v0, m, k, x_c, y_c, x_t=2e31-1, dt=0.01):
    """
    Simulate the trajectory of a projectile with air resistance.
    Simulation stops when the projectile hits the ground or passes the target.
    Arguments:
        angle_deg: Launch angle in degrees
        v0: Initial velocity in m/s
        m: Mass of the projectile in kg
        k: Drag coefficient in kg/s
        x_c, y_c: Initial coordinates of the cannon
        x_t: Target x-coordinate (default is a large number)
        dt: Time step for simulation
        max_time: Maximum simulation time
    Returns:
        traj: Array of (x, y) coordinates of the projectile at each time step

    """
    angle_rad = np.radians(angle_deg)
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    x, y = x_c, y_c
    traj = [(x, y)]

    while x < x_t and y >= 0:
        ax = - (k / m) * vx
        ay = -g - (k / m) * vy
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        traj.append((x, y))

    return np.array(traj)


def distance_to_target(angle_deg, v0, m, k, x_c, y_c, x_t, y_t):
    traj = simulate_trajectory(angle_deg, v0, m, k, x_c, y_c, x_t)
    last_x, last_y = traj[-1]
    return np.sqrt((last_x - x_t) ** 2 + (last_y - y_t) ** 2)
    

def find_shooting_angle(v0, m, k, x_c, y_c, x_t, y_t, max_iter=1000, method='gss', angle_guess_1=45, angle_guess_2=50):
    """
    Find the optimal shooting angle using root finding.

    Parameters:
        v0: Initial velocity in m/s
        m: Mass of the projectile in kg
        k: Drag coefficient in kg/s
        x_c, y_c: Initial coordinates of the cannon
        x_t, y_t: Target coordinates
        max_iter: Maximum number of iterations for the root finding
        method: Method for root finding ('secant' or 'gss')
        angle_guess_1: First guess for the angle (for secant method)
        angle_guess_2: Second guess for the angle (for secant method)
    Returns:
        angle: Optimal shooting angle in degrees
    """
    def f(angle):
        return distance_to_target(angle, v0, m, k, x_c, y_c, x_t, y_t)
    
    if method == 'secant':
        result = root_scalar(
            f, 
            bracket=[1, 89], 
            method='secant',
            xtol=1e-3,
            rtol=1e-3,
            maxiter=max_iter,
            x0=angle_guess_1,  # First guess
            x1=angle_guess_2   # Second guess
        )
    elif method == 'gss':
        result = minimize_scalar_positive(
            f,
            bracket=[1, 89],
            xtol=1e-2,
            max_iter=max_iter
        )

    if not result.converged:
        print(f"Warning: Angle finding did not converge after {result.iterations}. Still returning the closest value..." )

    return result.root
    
    

def main():
    # Input parameters
    x_c, y_c = 0, 0           # Cannon location
    x_t, y_t = 400, 0         # Target location (on ground)
    v0 = 30.0                # m/s
    m = 5.0                   # kg
    k = 0.1                   # Drag coefficient (kg/s)

    # Find optimal shooting angle
    angle = 69
    # angle = find_shooting_angle(v0, m, k, x_c, y_c, x_t, y_t)
    print(f"Optimal shooting angle: {angle:.2f} degrees")

    # Simulate final trajectory
    traj = simulate_trajectory(angle, v0, m, k, x_c, y_c, x_t)
    print(f"Final coordinates: {traj[-1]}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(traj[:, 0], traj[:, 1], label=f"Trajectory @ {angle:.2f}°")
    plt.plot(x_t, y_t, 'ro', label="Target")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title("Projectile Trajectory with Air Resistance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

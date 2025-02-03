import numpy as np
from scipy.integrate import solve_ivp
from itertools import product
import matplotlib.pyplot as plt
from .newton import newton_method, jacobian
from dsgrn_boolean.models.hill import hill
import os

def integrate_system(system, x0, t_span=(0, 20), rtol=1e-4):
    """Integrate ODE system to find stable equilibria"""
    def event(t, x):
        return np.linalg.norm(system(x)) - 1e-4  # Relaxed tolerance
    event.terminal = True
    event.direction = -1
    
    sol = solve_ivp(
        lambda t, x: system(x),
        t_span,
        x0,
        method='RK45',
        events=event,
        rtol=rtol,
        atol=1e-6,
        max_step=0.1  # Control step size
    )
    
    # Check if system has stabilized by looking at final derivatives
    final_deriv = np.linalg.norm(system(sol.y[:, -1]))
    has_converged = final_deriv < 1e-3 or sol.status == 1
    
    return sol.y[:, -1], has_converged

def plot_phase_space(L, U, T, d, specific_points, stable_points, unstable_points):
    """
    Create a comprehensive phase space plot showing:
    - Initial points
    - Trajectories
    - Stable/unstable equilibria
    - Nullclines
    """
    
    def system(x):
        return hill(L, U, T, d)[0](x)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Add nullclines
    x_max = 1.5*(U[0,0] + U[1,0])
    y_max = 1.5*(U[0,1] * U[1,1])
    x_range = (0, x_max)
    y_range = (0, y_max)
    
    n_points = 100
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    derivatives = np.array([system(p) for p in points])
    
    dx = derivatives[:, 0].reshape(n_points, n_points)
    dy = derivatives[:, 1].reshape(n_points, n_points)
    
    ax.contour(X, Y, dx, levels=[0], colors='blue', alpha=0.5, linestyles='--', label='x-nullcline')
    ax.contour(X, Y, dy, levels=[0], colors='red', alpha=0.5, linestyles='--', label='y-nullcline')
    
    specific_points_np = np.array(specific_points)
    ax.plot(specific_points_np[:, 0], specific_points_np[:, 1], 'k.', markersize=5)  # Initial points in black
    
    # Plot stable and unstable points
    if stable_points:
        stable_x = [p[0] for p in stable_points]
        stable_y = [p[1] for p in stable_points]
        ax.scatter(stable_x, stable_y, c='g', s=100, marker='*', 
                  label=f'Stable ({len(stable_points)})', zorder=5)
    
    if unstable_points:
        unstable_x = [p[0] for p in unstable_points]
        unstable_y = [p[1] for p in unstable_points]
        ax.scatter(unstable_x, unstable_y, c='r', s=100, marker='x', 
                  label=f'Unstable ({len(unstable_points)})', zorder=5)
    
    # Customize plot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Phase Space Analysis (d={d})\nFound {len(stable_points)} stable, {len(unstable_points)} unstable points')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def find_stable_states(L, U, T, d, previous_states=None, use_newton_backup=False, visualize=False):
    """
    Find stable equilibrium points using ODE integration and Newton's method.
    
    Args:
        L: Lower bounds matrix
        U: Upper bounds matrix
        T: Threshold matrix
        d: Hill coefficient
        previous_states: List of states from previous d value (for continuity)
        use_newton_backup: Whether to use Newton's method as backup
        visualize: Whether to create and return a phase space plot
        
    Returns:
        tuple: (stable_points, unstable_points) if not visualize
        tuple: (stable_points, unstable_points, figure) if visualize
    """
    
    def system(x):
        return hill(L, U, T, d)[0](x)
    
    unique_points = []
    
    def is_new_point(point, existing_points, tolerance=1e-3):
        """Check if point is significantly different from existing points"""
        return not any(np.linalg.norm(point - p) < tolerance for p in existing_points)
    
    # 2. Define x and y coordinates separately
    x_coords = [
        T[0,0],              # Threshold
        T[0,1],              # Threshold
        L[0,0] + L[1,0],     # x-nullcline
        U[0,0] + L[1,0],     # x-nullcline
        L[0,0] + U[1,0],     # x-nullcline
        U[0,0] + U[1,0]      # x-nullcline
    ]
    
    y_coords = [
        T[1,0],              # Threshold
        T[1,1],              # Threshold
        L[0,1] * L[1,1],     # y-nullcline
        U[0,1] * L[1,1],     # y-nullcline
        L[0,1] * U[1,1],     # y-nullcline
        U[0,1] * U[1,1]      # y-nullcline
    ]
    
    # Create all possible combinations
    specific_points = [np.array([x, y]) for x, y in product(x_coords, y_coords)]
    
    # Try each initial point using Newton's method
    for x0 in specific_points:
        x, converged, _ = newton_method(system, x0)
        if converged and is_new_point(x, unique_points):
            unique_points.append(x)
    
    # Classify stability
    stable_points = []
    unstable_points = []
    for point in unique_points:
        J = jacobian(system, point)
        eigenvalues = np.linalg.eigvals(J)
        if all(np.real(eigenvalues) < 0):
            if is_new_point(point, stable_points):
                stable_points.append(point)
        else:
            if is_new_point(point, unstable_points):
                unstable_points.append(point)
    
    # Backup strategies if not enough points found
    if len(stable_points) < 3:
        # Integrate from all specific_points, collect unique endpoints, and use Newton's method
        integrated_points = []
        for x0 in specific_points:
            x_integrated, converged = integrate_system(system, x0)
            if converged and is_new_point(x_integrated, integrated_points):
                integrated_points.append(x_integrated)
        for x_integrated in integrated_points:
            x, converged, _ = newton_method(system, x_integrated)
            J = jacobian(system, x)
            eigenvalues = np.linalg.eigvals(J)
            if converged and is_new_point(x, unique_points) and all(np.real(eigenvalues) < 0):
                stable_points.append(x)
    
    if len(stable_points) < 3:
        x0 = np.array([T[0,1], T[1,0]])
        vector_field_value = system(x0)
        vector_field_norm = np.linalg.norm(vector_field_value)
    
    # Create visualization if requested
    if visualize:
        fig = plot_phase_space(L, U, T, d, specific_points, stable_points, unstable_points)
        filename = f"phase_space_d_{d}.png"
        filepath = os.path.join("phase_space_plots", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath)
        return stable_points, unstable_points, fig
    
    return stable_points, unstable_points 
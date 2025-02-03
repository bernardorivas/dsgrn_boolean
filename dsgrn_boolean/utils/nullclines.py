import numpy as np
import matplotlib.pyplot as plt
from .newton import newton_method
from dsgrn_boolean.models.hill import HillFunction
from dsgrn_boolean.models.hill import hill
from scipy.integrate import solve_ivp
from itertools import product

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
        max_step=0.1
    )
    
    final_deriv = np.linalg.norm(system(sol.y[:, -1]))
    has_converged = final_deriv < 1e-3 or sol.status == 1
    
    return sol.y[:, -1], has_converged

def is_new_point(point, existing_points, rtol=1e-8):
    """Check if point is significantly different from existing points"""
    return not any(np.allclose(point, p, rtol=rtol) for p in existing_points)

def plot_nullclines(L, U, T, d, n_points=1000):
    """
    Plot nullclines of the system:
    x' = -x + h11(x) + h21(y)
    y' = -y + h12(x) * h22(y)
    
    Args:
        L: Lower bounds matrix
        U: Upper bounds matrix
        T: Threshold matrix
        d: Hill function steepness parameter
        n_points: Number of points for grid discretization (default=1000)
        
    Returns:
        List of equilibrium points found
    """
    # Create Hill functions
    h11 = HillFunction(L[0,0], U[0,0], T[0,0], d)
    h21 = HillFunction(L[1,0], U[1,0], T[1,0], d)
    h12 = HillFunction(U[0,1], L[0,1], T[0,1], d)
    h22 = HillFunction(L[1,1], U[1,1], T[1,1], d)
    
    # Create grid
    x_max = 1.5*(U[0,0] + U[1,0])
    y_max = 1.5*(U[0,1] * U[1,1])
    x = np.linspace(0, x_max, n_points)
    y = np.linspace(0, y_max, n_points)
    X, Y = np.meshgrid(x, y)
    
    # First nullcline: x' = 0 => x = h11(x) + h21(y)
    Z1 = h11(X) + h21(Y) - X
    
    # Second nullcline: y' = 0 => y = h12(x) * h22(y)
    Z2 = h12(X) * h22(Y) - Y
    
    # Plot
    plt.figure(figsize=(10, 10))
    
    # Create contours and get the collections for legend
    x_nullcline = plt.contour(X, Y, Z1, levels=[0], colors='blue')
    y_nullcline = plt.contour(X, Y, Z2, levels=[0], colors='red')
    
    # Create proxy artists for the legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', label="x-nullcline"),
        Line2D([0], [0], color='red', label="y-nullcline"),
        Line2D([0], [0], color='black', marker='o', label='Stable equilibrium', 
               linestyle='None', markersize=10),
        Line2D([0], [0], color='black', marker='o', label='Unstable equilibrium',
               linestyle='None', markersize=10, fillstyle='none')
    ]
    
    # Add threshold lines without labels
    plt.axvline(x=T[0,0], color='lightgray', linestyle='--', alpha=0.5)
    plt.axvline(x=T[0,1], color='lightgray', linestyle='--', alpha=0.5)
    plt.axhline(y=T[1,0], color='lightgray', linestyle='--', alpha=0.5)
    plt.axhline(y=T[1,1], color='lightgray', linestyle='--', alpha=0.5)
    
    # Set axis limits explicitly
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    
    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Nullclines (d={d})')
    
    # Remove grid, keep only axes
    plt.grid(False)
    
    # Get system and jacobian
    system, jacobian = hill(L, U, T, d)
    
    # Define specific points to try
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
    zeros = []
    
    # Step 1: Try newton's method on specific points
    for x0 in specific_points:
        x, converged, _ = newton_method(system, x0, df=jacobian)
        if converged and is_new_point(x, zeros):
            zeros.append(x)
    
    # Step 2: Try grid around specific points
    if len(zeros) < 3:
        for center in specific_points:
            grid_size = 5
            perturbations = np.linspace(-0.5, 0.5, grid_size)
            for dx in perturbations:
                for dy in perturbations:
                    x0 = center + np.array([dx, dy])
                    x, converged, _ = newton_method(system, x0, df=jacobian)
                    if converged and is_new_point(x, zeros):
                        zeros.append(x)
    
    # Step 3: Try forward integration
    if len(zeros) < 3:
        additional_points = [
            np.array([0, U[1,1]]),
            np.array([U[0,0], 0]),
            np.array([U[0,0]/2, U[1,1]/2])
        ]
        
        for x0 in additional_points:
            x_integrated, converged = integrate_system(system, x0)
            if converged:
                x, converged, _ = newton_method(system, x_integrated, df=jacobian)
                if converged and is_new_point(x, zeros):
                    zeros.append(x)
    
    # Plot equilibrium points
    for x in zeros:
        # Get stability
        J = jacobian(x)
        eigenvals = np.linalg.eigvals(J)
        stable = all(np.real(eigenvals) < 0)
        
        # Plot points
        if stable:
            plt.plot(x[0], x[1], 'ko', markersize=10)
        else:
            plt.plot(x[0], x[1], 'ko', fillstyle='none', markersize=10)
    
    # Add single legend with all elements
    plt.legend(handles=legend_elements, loc='best')
    plt.show()
    
    return zeros
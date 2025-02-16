import numpy as np
import matplotlib.pyplot as plt
from .newton import newton_method
from dsgrn_boolean.models.hill import *
from itertools import product

def is_new_point(point, existing_points, rtol=1e-8):
    """Check if point is significantly different from existing points"""
    return not any(np.allclose(point, p, rtol=rtol) for p in existing_points)

def plot_nullclines(L, U, T, d, n_points=1000):
    """
    Plot nullclines of the system:
    x' = -x + h00(x) + h10(y)
    y' = -y + h01(x) * h11(y)
    
    Args:
        L: Lower bounds matrix
        U: Upper bounds matrix
        T: Threshold matrix
        d: Hill function steepness parameter
        n_points: Number of points for grid discretization (default=1000)
        
    Returns:
        fig : figure of the nullclines
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create Hill functions
    h00 = HillFunction(L[0,0], U[0,0], T[0,0], d)
    h10 = HillFunction(L[1,0], U[1,0], T[1,0], d)
    h01 = HillFunction(U[0,1], L[0,1], T[0,1], d)
    h11 = HillFunction(L[1,1], U[1,1], T[1,1], d)
    
    # Create grid
    x_max = 1.5*(U[0,0] + U[1,0])
    y_max = 1.5*(U[0,1] * U[1,1])
    x = np.linspace(0, x_max, n_points)
    y = np.linspace(0, y_max, n_points)
    X, Y = np.meshgrid(x, y)
    
    # First nullcline: x' = 0 
    Z1 = h00(X) + h10(Y) - X
    
    # Second nullcline: y' = 0
    Z2 = h01(X) * h11(Y) - Y
    
    # Plot
    ax.contour(X, Y, Z1, levels=[0], colors='blue')
    ax.contour(X, Y, Z2, levels=[0], colors='red')
    
    # Add threshold lines without labels
    ax.axvline(x=T[0,0], color='lightgray', linestyle='--', alpha=0.5)
    ax.axvline(x=T[0,1], color='lightgray', linestyle='--', alpha=0.5)
    ax.axhline(y=T[1,0], color='lightgray', linestyle='--', alpha=0.5)
    ax.axhline(y=T[1,1], color='lightgray', linestyle='--', alpha=0.5)
    
    # Set axis limits explicitly
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    # Add labels and legend
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Nullclines (d={d})')
    
    # Remove grid, keep only axes
    ax.grid(False)
    
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
            grid_size = 2
            perturbations = np.linspace(-1, 1, grid_size)
            for dx in perturbations:
                for dy in perturbations:
                    x0 = center + np.array([dx, dy])
                    x, converged, _ = newton_method(system, x0, df=jacobian)
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
            ax.plot(x[0], x[1], 'ko', markersize=10)
        else:
            ax.plot(x[0], x[1], 'ko', fillstyle='none', markersize=10)
    
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
    
    # Add single legend with all elements
    ax.legend(handles=legend_elements, loc='best')
    
    # Return both figure and equilibrium points
    return fig
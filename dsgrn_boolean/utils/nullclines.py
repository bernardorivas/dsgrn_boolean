import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dsgrn_boolean.models.hill import hill

def is_new_point(point, existing_points, rtol=1e-8):
    """Check if point is significantly different from existing points"""
    return not any(np.allclose(point, p, rtol=rtol) for p in existing_points)

def plot_nullclines(L, U, T, d, n_points=100):
    """
    Plot nullclines of the system:
    x' = -x + h00(x) + h10(y)
    y' = -y + h01(x) * h11(y)
    
    Args:
        L: Lower bounds
        U: Upper bounds
        T: Thresholds
        d: Hill coefficient 
        n_points: Number of points for grid discretization (default=100)
        
    Returns:
        fig : plot of the nullclines
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create system and jacobian
    system, _ = hill(L, U, T, d)
    
    # Create grid
    x_max = 1.5*(U[0,0] + U[1,0])
    y_max = 1.5*(U[0,1] * U[1,1])
    x = np.linspace(0, x_max, n_points)
    y = np.linspace(0, y_max, n_points)
    X, Y = np.meshgrid(x, y)
    
    # First nullcline: x' = 0
    Z1 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z1[i, j] = system(np.array([X[i, j], Y[i, j]]))[0]
    
    # Second nullcline: y' = 0
    Z2 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z2[i, j] = system(np.array([X[i, j], Y[i, j]]))[1]
    
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
    
    # legend
    legend_elements = [
        Line2D([0], [0], color='blue', label="x-nullcline"),
        Line2D([0], [0], color='red', label="y-nullcline")
    ]
    
    # Add single legend with all elements
    ax.legend(handles=legend_elements, loc='best')
    
    # Return both figure and equilibrium points
    return fig
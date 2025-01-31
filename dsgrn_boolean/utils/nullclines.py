import numpy as np
import matplotlib.pyplot as plt
from .newton import newton_method
from dsgrn_boolean.models.hill import HillFunction
from dsgrn_boolean.models.hill import hill

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
    
    # Add intersections (equilibria)
    system, jacobian = hill(L, U, T, d)
    
    # Find zeros using newton method with more initial conditions
    n_grid = 10 
    x_grid = np.linspace(0, x_max, n_grid)
    y_grid = np.linspace(0, y_max, n_grid)
    initial_conditions = [np.array([x, y]) for x in x_grid for y in y_grid]
    
    zeros = []
    print(f"\nEquilibria for d = {d}:")
    print("-" * 50)
    
    for x0 in initial_conditions:
        x, converged, _ = newton_method(system, x0, df=jacobian)
        if converged:
            is_new = True
            for z in zeros:
                if np.allclose(z, x, rtol=1e-8):
                    is_new = False
                    break
            if is_new:
                zeros.append(x)
                # Get stability
                J = jacobian(x)
                eigenvals = np.linalg.eigvals(J)
                stable = all(np.real(eigenvals) < 0)
                
                # Print stability information
                print(f"\nEquilibrium point: ({x[0]:.6f}, {x[1]:.6f})")
                print(f"Eigenvalues: {eigenvals[0]:.6f}, {eigenvals[1]:.6f}")
                print(f"Stability: {'Stable' if stable else 'Unstable'}")
                
                # Plot points
                if stable:
                    plt.plot(x[0], x[1], 'ko', markersize=10)
                else:
                    plt.plot(x[0], x[1], 'ko', fillstyle='none', markersize=10)
    
    # Add single legend with all elements
    plt.legend(handles=legend_elements, loc='best')
    plt.show()
    
    return zeros
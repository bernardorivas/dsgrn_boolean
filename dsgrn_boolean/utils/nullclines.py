import numpy as np
import matplotlib.pyplot as plt
from .newton import newton_method
from dsgrn_boolean.models.hill import HillFunction
from dsgrn_boolean.models.hill_model import hill  # Updated import

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
    plt.contour(X, Y, Z1, levels=[0], colors='blue', label="x' = 0")
    plt.contour(X, Y, Z2, levels=[0], colors='red', label="y' = 0")
    
    # Add threshold lines
    plt.axvline(x=T[0,0], color='lightgray', linestyle='--', alpha=0.5, label='T[0,0]')
    plt.axvline(x=T[0,1], color='lightgray', linestyle='--', alpha=0.5, label='T[0,1]')
    plt.axhline(y=T[1,0], color='lightgray', linestyle='--', alpha=0.5, label='T[1,0]')
    plt.axhline(y=T[1,1], color='lightgray', linestyle='--', alpha=0.5, label='T[1,1]')
    
    # Set axis limits explicitly
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    
    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Nullclines (d={d})')
    plt.legend()
    plt.grid(True)
    
    # Add intersections (equilibria)
    system, jacobian = hill(L, U, T, d)
    
    # Find zeros using newton method
    n_grid = 20  # coarser grid for initial conditions
    x_grid = np.linspace(0, x_max, n_grid)
    y_grid = np.linspace(0, y_max, n_grid)
    initial_conditions = [np.array([x, y]) for x in x_grid for y in y_grid]
    
    zeros = []
    for x0 in initial_conditions:
        x, converged, _ = newton_method(system, x0, df=jacobian)
        if converged:
            # Check if this zero is already found
            is_new = True
            for z in zeros:
                if np.allclose(z, x, rtol=1e-5):
                    is_new = False
                    break
            if is_new:
                zeros.append(x)
                # Get stability
                J = jacobian(x)
                eigenvals = np.linalg.eigvals(J)
                stable = all(np.real(eigenvals) < 0)
                # Plot stable points as filled circles, unstable as empty circles
                if stable:
                    plt.plot(x[0], x[1], 'ko', markersize=10, label='Stable equilibrium' if len(zeros)==1 else "")
                else:
                    plt.plot(x[0], x[1], 'ko', fillstyle='none', markersize=10, label='Unstable equilibrium' if len(zeros)==1 else "")
    
    plt.legend()
    plt.show()
    
    return zeros

def hill(L, U, T, d):
    """
    Create the system of equations and its Jacobian for the Hill function system.
    
    Args:
        L: Lower bounds matrix
        U: Upper bounds matrix
        T: Threshold matrix
        d: Hill function steepness parameter
        
    Returns:
        Tuple (system, jacobian) where:
            system: Function that computes the system of equations
            jacobian: Function that computes the Jacobian matrix
    """
    # Create Hill functions
    h11 = HillFunction(L[0,0], U[0,0], T[0,0], d)
    h21 = HillFunction(L[1,0], U[1,0], T[1,0], d)
    h12 = HillFunction(U[0,1], L[0,1], T[0,1], d)
    h22 = HillFunction(L[1,1], U[1,1], T[1,1], d)
    
    def system(x):
        return np.array([
            -x[0] + h11(x[0]) + h21(x[1]),
            -x[1] + h12(x[0]) * h22(x[1])
        ])
    
    def jacobian(x):
        return np.array([
            [-1 + h11.derivative(x[0]), h21.derivative(x[1])],
            [h12.derivative(x[0]) * h22(x[1]), -1 + h12(x[0]) * h22.derivative(x[1])]
        ])
    
    return system, jacobian 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from .dsgrn_sample_to_matrix import extract_parameter_matrices
from dsgrn_boolean.models.hill import hill
from itertools import product

def plot_trajectory(L, U, T, d, x0, t_span=(0, 10), n_points=1000):
    """
    Plot the trajectory of the ODE system from a given initial condition.
    
    Args:
        L, U, T: Parameter matrices
        d: Hill coefficient
        x0: Initial condition [x0, y0]
        t_span: Time span for integration
        n_points: Number of points for plotting
    """
    system, jacobian = hill(L, U, T, d)
    
    # Solve ODE
    t = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        lambda t, x: system(x),
        t_span,
        x0,
        t_eval=t,
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot trajectory in phase space
    ax1.plot(sol.y[0], sol.y[1], 'b-', label='Trajectory')
    ax1.plot(x0[0], x0[1], 'go', label='Initial point')
    ax1.plot(sol.y[0][-1], sol.y[1][-1], 'ro', label='Final point')
    
    # Add nullclines
    x_max = 1.5*(U[0,0] + U[1,0])
    y_max = 1.5*(U[0,1] * U[1,1])
    x = np.linspace(0, x_max, 100)
    y = np.linspace(0, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate nullclines
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = system([X[i,j], Y[i,j]])[0]
    ax1.contour(X, Y, Z, levels=[0], colors='blue', alpha=0.5, linestyles='--')
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = system([X[i,j], Y[i,j]])[1]
    ax1.contour(X, Y, Z, levels=[0], colors='red', alpha=0.5, linestyles='--')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Phase Space')
    ax1.legend()
    ax1.grid(True)
    
    # Plot time series
    ax2.plot(sol.t, sol.y[0], 'b-', label='x(t)')
    ax2.plot(sol.t, sol.y[1], 'r-', label='y(t)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.set_title('Time Series')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def debug_sample(network, sample, d=100):
    """
    Debug a specific sample by showing trajectories from different initial conditions.
    
    Args:
        network: DSGRN network
        sample: Parameter sample
        d: Hill coefficient to use
    Returns:
        list of figures
    """
    L, U, T = extract_parameter_matrices(sample, network)
    
    # Define x and y coordinates separately
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
    
    print(f"\nDebugging sample with d={d}")
    print("Parameter matrices:")
    print("L =\n", L)
    print("U =\n", U)
    print("T =\n", T)
    
    # Plot trajectory from each specific point
    figures = []
    for i, x0 in enumerate(specific_points):
        print(f"\nSimulating from initial point {i+1}: {x0}")
        fig = plot_trajectory(L, U, T, d, x0)
        plt.suptitle(f'Trajectory from point {i+1}: ({x0[0]:.2f}, {x0[1]:.2f})')
        figures.append(fig)
    
    return figures 
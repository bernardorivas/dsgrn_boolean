import DSGRN
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from ipywidgets import interact, FloatSlider, IntSlider
import ipywidgets as widgets
from dsgrn_boolean.utils.sample_management import load_samples
from dsgrn_boolean.utils.dsgrn_sample_to_matrix import extract_parameter_matrices
from dsgrn_boolean.models.hill import hill
from dsgrn_boolean.utils.newton import newton_method
from scipy.integrate import solve_ivp

def plot_phase_portrait(L, U, T, d, n_points=20):
    """Plot phase portrait with nullclines and trajectories."""
    system, jacobian = hill(L, U, T, d)
    
    # Create grid of points
    x_max = max(U[0,0] + U[1,0], U[0,1] * U[1,1]) * 1.2
    y_max = x_max
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])
    
    # Optimize trajectory calculations
    x_starts = np.linspace(0.1, x_max, n_points)[::2]  # Use every other point
    y_starts = np.linspace(0.1, y_max, n_points)[::2]
    t_span = (0, 30)  # Reduced time span
    
    # Vectorize initial conditions
    X0, Y0 = np.meshgrid(x_starts, y_starts)
    initial_conditions = np.column_stack((X0.ravel(), Y0.ravel()))
    
    # Plot trajectories more efficiently
    for ic in initial_conditions:
        sol = solve_ivp(
            lambda t, x: system(x),
            t_span,
            ic,
            method='RK45',
            rtol=1e-4,  # Reduced tolerance
            max_step=0.5,  # Increased max step
            dense_output=True
        )
        
        # Fewer points for dense output
        t_dense = np.linspace(0, sol.t[-1], 100)  # Reduced from 200
        y_dense = sol.sol(t_dense)
        
        # Plot trajectory with single call using alpha gradient
        points = y_dense.T
        ax.plot(points[:,0], points[:,1], 'gray', alpha=0.3, linewidth=0.8)
        ax.plot(ic[0], ic[1], 'k.', markersize=2, alpha=0.3)
    
    # Optimize nullcline calculations
    n_grid = 100  # Reduced from 200
    xx = np.linspace(0, x_max, n_grid)
    yy = np.linspace(0, y_max, n_grid)
    XX, YY = np.meshgrid(xx, yy)
    
    # Vectorize system evaluation
    points = np.stack((XX, YY), axis=-1)
    UV = np.apply_along_axis(system, 2, points)
    
    # Plot nullclines
    ax.contour(XX, YY, UV[:,:,0], levels=[0], colors='blue', alpha=0.7)
    ax.contour(XX, YY, UV[:,:,1], levels=[0], colors='red', alpha=0.7)
    
    # Optimize equilibria search
    n_search = 8  # Reduced from 10
    search_points = np.linspace(0, x_max, n_search)
    X0, Y0 = np.meshgrid(search_points, search_points)
    initial_guesses = np.column_stack((X0.ravel(), Y0.ravel()))
    
    equilibria = []
    for guess in initial_guesses:
        eq, converged, _ = newton_method(system, guess, df=jacobian)
        if converged and all(eq >= 0):
            if not any(np.allclose(eq, e) for e in equilibria):
                equilibria.append(eq)
                J = jacobian(eq)
                eigenvals = np.linalg.eigvals(J)
                if all(np.real(eigenvals) < 0):
                    ax.plot(eq[0], eq[1], 'go', markersize=10, label='Stable')
                else:
                    ax.plot(eq[0], eq[1], 'ro', markersize=10, label='Unstable')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Phase Portrait (d={d})')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    return fig

def interactive_phase_portrait(par_index=98, sample_index=0):
    """Create interactive phase portrait visualization."""
    # Load sample
    samples = load_samples(par_index, filtered=True, filter_tol=0.1)
    sample = samples[sample_index]
    
    # Setup network
    network = DSGRN.Network("""x : x + y : E
                              y : (~x) y : E""")
    
    # Extract matrices
    L, U, T = extract_parameter_matrices(sample, network)
    
    # Create interactive plot
    @interact(
        d=FloatSlider(min=1, max=100, step=1, value=20, 
                      description='Hill coefficient:'),
        n_points=IntSlider(min=10, max=50, step=5, value=20, 
                          description='Grid resolution:')
    )
    def update(d, n_points):
        plt.close('all')  # Clear previous plots
        fig = plot_phase_portrait(L, U, T, d, n_points)
        return fig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameter', type=int, default=98,
                      help='Parameter index (default: 98)')
    parser.add_argument('-s', '--sample', type=int, default=0,
                      help='Sample index (default: 0)')
    args = parser.parse_args()
    
    interactive_phase_portrait(args.parameter, args.sample) 
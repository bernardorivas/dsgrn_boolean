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
    
    # Plot trajectories from different initial conditions
    x_starts = np.linspace(0.1, x_max, n_points)
    y_starts = np.linspace(0.1, y_max, n_points)
    t_span = (0, 50)  # Time span for integration
    
    # Create a grid of initial conditions
    for x0 in x_starts[::2]:  # Use every other point to avoid overcrowding
        for y0 in y_starts[::2]:
            sol = solve_ivp(
                lambda t, x: system(x),
                t_span,
                [x0, y0],
                method='RK45',
                rtol=1e-6,
                max_step=0.1,
                dense_output=True
            )
            
            # Get dense output for smooth plotting
            t_dense = np.linspace(0, sol.t[-1], 200)
            y_dense = sol.sol(t_dense)
            
            # Plot trajectory with alpha gradient
            points = y_dense.T
            segments = np.array([[points[i], points[i+1]] for i in range(len(points)-1)])
            alpha_gradient = np.linspace(0.1, 0.8, len(segments))
            
            for segment, alpha in zip(segments, alpha_gradient):
                ax.plot(segment[:,0], segment[:,1], 'gray', alpha=alpha, linewidth=0.8)
            
            # Plot start point
            ax.plot(x0, y0, 'k.', markersize=2, alpha=0.3)
    
    # Plot nullclines
    xx = np.linspace(0, x_max, 200)
    yy = np.linspace(0, y_max, 200)
    XX, YY = np.meshgrid(xx, yy)
    
    UV = np.zeros((200, 200, 2))
    for i in range(200):
        for j in range(200):
            UV[i,j,:] = system([XX[i,j], YY[i,j]])
    
    # x-nullcline where dx/dt = 0
    ax.contour(XX, YY, UV[:,:,0], levels=[0], colors='blue', alpha=0.7)
    # y-nullcline where dy/dt = 0
    ax.contour(XX, YY, UV[:,:,1], levels=[0], colors='red', alpha=0.7)
    
    # Find and plot equilibria
    equilibria = []
    for x0 in np.linspace(0, x_max, 10):
        for y0 in np.linspace(0, y_max, 10):
            eq, converged, _ = newton_method(system, np.array([x0, y0]), df=jacobian)
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
    
    # Remove duplicate labels
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
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameter', type=int, default=98,
                      help='Parameter index (default: 98)')
    parser.add_argument('-s', '--sample', type=int, default=0,
                      help='Sample index (default: 0)')
    args = parser.parse_args()
    
    interactive_phase_portrait(args.parameter, args.sample) 
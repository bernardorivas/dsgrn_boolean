import DSGRN
from dsgrn_boolean.utils.sample_management import load_samples
from dsgrn_boolean.utils.find_stable_states import find_stable_states
import matplotlib.pyplot as plt
import os
import time
import numpy as np

def main():
    # Setup network and parameter
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    par_index = 98
    parameter = parameter_graph.parameter(par_index)
    
    # Load first sample
    samples = load_samples(par_index)
    first_sample = samples[4]
    
    # Extract matrices
    from dsgrn_boolean.utils.dsgrn_sample_to_matrix import extract_parameter_matrices
    L, U, T = extract_parameter_matrices(first_sample, network)
    
    # Find stable states with visualization
    stable, unstable, fig = find_stable_states(L, U, T, d=100, visualize=True)
    
    # Print detailed information about stable states
    print(f"\nFound {len(stable)} stable states:")
    for i, point in enumerate(stable, 1):
        print(f"Stable point {i}: x = {point[0]:.4f}, y = {point[1]:.4f}")
    
    if unstable:
        print(f"\nFound {len(unstable)} unstable states:")
        for i, point in enumerate(unstable, 1):
            print(f"Unstable point {i}: x = {point[0]:.4f}, y = {point[1]:.4f}")
    
    # Save figure
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(root_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"phase_space_analysis_{timestamp}.svg"
    filepath = os.path.join(figures_dir, filename)
    fig.savefig(filepath, format='svg', bbox_inches='tight')
    print(f"\nFigure saved as: {filename}")
    
    plt.show()

if __name__ == "__main__":
    main() 
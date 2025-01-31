import DSGRN
import os
import time
import matplotlib.pyplot as plt
from dsgrn_boolean.utils.hill_stable_analysis import analyze_stability_parallel, plot_stability_results
from dsgrn_boolean.utils.sample_management import load_samples

def main():
    # Setup network and parameter
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    par_index = 98
    parameter = parameter_graph.parameter(par_index)
    
    # Load and process samples
    samples = load_samples(par_index)[:1]  # First few samples
    
    # Run analysis
    d_range = range(100, 0, -5)
    results = analyze_stability_parallel(
        network,
        parameter,
        samples,
        d_range=d_range
    )
    
    # Create and save plot
    fig = plot_stability_results(results, d_range, par_index)
    
    # Save figure
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(root_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"stability_analysis_d{min(d_range)}-{max(d_range)}_{timestamp}.svg"
    filepath = os.path.join(figures_dir, filename)
    fig.savefig(filepath, format='svg', bbox_inches='tight')
    print(f"\nFigure saved as: {filename}")
    
    plt.show()

if __name__ == "__main__":
    main() 
import DSGRN
import DSGRN_utils
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from dsgrn_boolean.utils.hill_stable_analysis import analyze_stability_parallel
from dsgrn_boolean.utils.sample_management import load_samples
from multiprocessing import cpu_count
import argparse

def main(par_index: int, show_plot: bool = False):
    """
    Run stability analysis for a given parameter index.
    
    Args:
        par_index: DSGRN parameter index (default: 98)
        show_plot: Whether to show the plot (default: False)
    """
    # Fixed parameters for testing
    n_samples = 200
    d_min = 1
    d_max = 100
    d_step = 1
    
    # Set default d_range
    d_range = list(range(d_min, d_max + 1, d_step))
    
    # Setup network and parameter using DSGRN
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    parameter = parameter_graph.parameter(par_index)
    
    # Get the Conley Morse graph using DSGRN_utils
    morse_graph, stg, graded_complex = DSGRN_utils.ConleyMorseGraph(parameter)
    
    # Count the number of stable equilibria
    n_equilibria = sum(1 for v in morse_graph.vertices() if not morse_graph.adjacencies(v))
    print(f"Number of stable equilibria expected: {n_equilibria}")
    

    # Load and process samples
    samples = load_samples(par_index, filtered=True, filter_tol=0.1)
    samples = samples[:n_samples]
    
    # Run parallel analysis
    n_processes = min(cpu_count(), n_samples)
    start_time = time.time()
    results = analyze_stability_parallel(
        network,
        parameter,
        samples,
        n_equilibria,
        d_range=d_range,
        n_processes=n_processes
    )
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time/len(samples):.2f} seconds")
    
    # Calculate detailed statistics and store comprehensive data
    detailed_data = {
        "par_index": par_index,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "network_spec": net_spec,
        "d_range": d_range,
        "num_samples": len(samples),
        "by_d": {}
    }
    
    # Process results for each d value
    for d in d_range:
        sample_list = results['by_d'][d]
        num_samples = len(sample_list)
        num_success = sum(1 for stable_states in sample_list if len(stable_states) > 0)
        num_failures = num_samples - num_success
        
        # Store detailed statistics for this d value
        detailed_data["by_d"][str(d)] = {
            "num_samples": num_samples,
            "num_success": num_success,
            "num_failures": num_failures,
            "success_rate": num_success / num_samples,
            "failure_rate": num_failures / num_samples,
            "sample_results": [
                {
                    "sample_index": i,
                    "num_stable_states": len(stable_states),
                    "stable_states": [state.tolist() for state in stable_states],
                    "success": len(stable_states) == n_equilibria
                }
                for i, stable_states in enumerate(sample_list)
            ]
        }
    
    # Save detailed results
    stability_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stability_data')
    os.makedirs(stability_dir, exist_ok=True)
    
    filename = f"parindex_{par_index}_detailed_stability_{detailed_data['timestamp']}.json"
    filepath = os.path.join(stability_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {filepath}")
    
    # Plot and optionally show results
    plot_stability_results(results, d_range, par_index)
    if show_plot:
        plt.show()

def plot_stability_results(results, d_range, par_index):
    """Plot success rates for each d value"""
    d_values = sorted(results['by_d'].keys())
    success_rates = []
    
    for d in d_values:
        samples = results['by_d'][d]
        num_samples = len(samples)
        num_success = sum(1 for stable_states in samples if len(stable_states) > 0)
        success_rates.append(100 * num_success / num_samples)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(d_values, success_rates, color='skyblue')
    ax.axis([0, 100, 0, 100])
    ax.set_xlabel('Hill Coefficient (d)')
    ax.set_ylabel('Coherency rate (%)')
    ax.set_title(f'Coherency rate by Hill coefficient at parameter node {par_index}')
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    filename = f"coherency_rate_parameter_{par_index}.png"
    filepath = os.path.join(figures_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {filepath}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameter', type=int, default=98,
                      help='Parameter index (default: 98)')
    parser.add_argument('-s', '--show', action='store_true',
                      help='Show the plot (default: False)')
    args = parser.parse_args()
    main(args.parameter, args.show)
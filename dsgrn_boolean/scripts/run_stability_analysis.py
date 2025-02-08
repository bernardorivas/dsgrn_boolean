import DSGRN
import DSGRN_utils
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from dsgrn_boolean.utils.hill_stable_analysis import analyze_stability_parallel
from dsgrn_boolean.utils.sample_management import load_samples
from dsgrn_boolean.utils.dsgrn_sample_to_matrix import extract_parameter_matrices
from multiprocessing import cpu_count
import argparse

def main(par_index: int = 98, n_samples: int = 100, d_min: int = 0, d_max: int = 100, d_step: int = 1):
    # Set default d_range
    d_range = list(range(d_min, d_max + 1, d_step))  # Convert to list since some functions might expect a list
    
    # Setup network and parameter: 0, 49, 98, 147.
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    parameter = parameter_graph.parameter(par_index)
    
    morse_graph, stg, graded_complex = DSGRN_utils.ConleyMorseGraph(parameter)

    n_equilibria = 0
    for v in morse_graph.vertices():
        if not morse_graph.adjacencies(v):
            n_equilibria += 1

    print(f"Number of stable equilibria expected: {n_equilibria}")
    
    # Load and process samples
    samples = load_samples(par_index, filtered=True, filter_tol=0.1)
    samples = samples[:n_samples]
    
    
    # Pre-allocate processed_samples list
    processed_samples = [(extract_parameter_matrices(sample, network)) 
                        for sample in samples]
    
    # Run analysis
    n_processes = min(cpu_count(), len(processed_samples))
    print(f"Running stability analysis with {n_processes} processes.")
    results = analyze_stability_parallel(
        network,
        parameter,
        processed_samples,
        n_equilibria,
        d_range=d_range,
        n_processes=n_processes
    )
    
    # Calculate detailed statistics and store comprehensive data
    detailed_data = {
        "par_index": par_index,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "network_spec": net_spec,
        "d_range": d_range,
        "num_samples": len(samples),
        "by_d": {}
    }
    
    for d, sample_list in results['by_d'].items():
        num_samples = len(sample_list)
        success_array = np.array([len(stable_states) == n_equilibria for stable_states in sample_list])
        num_success = np.sum(success_array)
        num_failures = num_samples - num_success
        
        # Store detailed information for each d value
        detailed_data["by_d"][str(d)] = {  # Convert d to string for JSON compatibility
            "num_samples": num_samples,
            "num_success": int(num_success),
            "num_failures": int(num_failures),
            "success_rate": float(num_success / num_samples) if num_samples > 0 else 0,
            "failure_rate": float(num_failures / num_samples) if num_samples > 0 else 0,
            "sample_results": [
                {
                    "sample_index": i,
                    "num_stable_states": len(stable_states),
                    "stable_states": [state.tolist() for state in stable_states],  # Convert numpy arrays to lists
                    "success": len(stable_states) == n_equilibria
                }
                for i, stable_states in enumerate(sample_list)
            ]
        }
    
    # Save comprehensive data
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "stability_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save detailed data
    data_filename = f"parindex_{par_index}_detailed_stability_{detailed_data['timestamp']}.json"
    with open(os.path.join(data_dir, data_filename), "w") as f:
        json.dump(detailed_data, f, indent=2)
    
    # Create and save plot
    d_values = np.array(sorted(map(int, detailed_data["by_d"].keys())))
    success_rates = np.array([detailed_data["by_d"][str(d)]["success_rate"] * 100 for d in d_values])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(d_values, success_rates, color='skyblue')
    ax.set_xlabel('Hill Coefficient (d)')
    ax.set_ylabel('Matching Success (%)')
    ax.set_title(f'Matching Success Rates vs Hill Coefficient (Parameter {par_index})')
    ax.set_ylim(0, 110)
    ax.set_xlim(0, max(d_values)+1)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    figures_dir = os.path.join(root_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plot_filename = f"stability_analysis_matching_{detailed_data['timestamp']}.svg"
    fig.savefig(os.path.join(figures_dir, plot_filename), format='svg', bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run stability analysis for a given parameter index.')
    parser.add_argument('-p', '--par_index', type=int, default=98,
                      help='Parameter index to analyze (default: 98)')
    parser.add_argument('-n', '--n_samples', type=int, default=100,
                      help='Number of samples to analyze (default: 100)')
    parser.add_argument('--d_min', type=int, default=0,
                      help='Minimum Hill coefficient (default: 0)')
    parser.add_argument('--d_max', type=int, default=100,
                      help='Maximum Hill coefficient (default: 100)')
    parser.add_argument('--d_step', type=int, default=1,
                      help='Step size for Hill coefficient (default: 1)')
    args = parser.parse_args()
    
    main(args.par_index, args.n_samples, args.d_min, args.d_max, args.d_step)
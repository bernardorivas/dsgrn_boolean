import DSGRN
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from dsgrn_boolean.utils.hill_stable_analysis import analyze_stability_parallel
from dsgrn_boolean.utils.sample_management import load_samples
from dsgrn_boolean.utils.dsgrn_sample_to_matrix import extract_parameter_matrices

def main():
    # Setup network and parameter
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    par_index = 98
    parameter = parameter_graph.parameter(par_index)
    
    # Load and process samples
    # load_samples(par_index, network=None, parameter=None, force_regenerate=False, filtered=False, filter_tol=0.1):
    samples = load_samples(par_index, filtered=True, filter_tol=0.1)[:3]  # Use a single sample for now
    
    # Define d_range. Here we use primes from 2 up to 97 as an example.
    # d_range = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    d_range = range(1,101)
    
    # Convert samples to the format expected by analyze_stability_parallel.
    processed_samples = []
    for sample in samples:
        L, U, T = extract_parameter_matrices(sample, network)
        for d in d_range:
            processed_samples.append((L, U, T, d, None))
    
    # Run analysis (visualize is False because we only want the success/failure statistics)
    results = analyze_stability_parallel(
        network,
        parameter,
        processed_samples,
        d_range=d_range,
        visualize=False,
    )
    
    # Calculate success/failure rates.
    # We define a "success" for a given d if a sample yields exactly three stable states.
    success_data = {}  # key: d, value: statistics dictionary
    for d, sample_list in results['by_d'].items():
        num_samples = len(sample_list)
        num_success = sum(1 for stable_states in sample_list if len(stable_states) == 3)
        num_failures = num_samples - num_success
        success_data[d] = {
            "num_samples": num_samples,
            "num_success": num_success,
            "num_failures": num_failures,
            "success_rate": num_success / num_samples if num_samples > 0 else 0,
            "failure_rate": num_failures / num_samples if num_samples > 0 else 0,
        }
    
    # Store success_data along with the par_index into a JSON file so you don't have to re-run.
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "stability_data")
    os.makedirs(data_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    data_filename = f"parindex_{par_index}_stability_data_{timestamp}.json"
    data_filepath = os.path.join(data_dir, data_filename)
    with open(data_filepath, "w") as f:
        json.dump({"par_index": par_index, "success_data": success_data}, f, indent=2)
    print(f"Stability data saved to: {data_filepath}")
    
    # Create a plot where x-axis is Hill coefficient (d) and y-axis is matching success %
    d_values = sorted(success_data.keys())
    success_rates = [success_data[d]["success_rate"] * 100 for d in d_values]  # convert to percentages
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(d_values, success_rates, color='skyblue')
    
    ax.set_xlabel('Hill Coefficient (d)')
    ax.set_ylabel('Matching Success (%)')
    ax.set_title(f'Matching Success Rates vs Hill Coefficient (Parameter {par_index})')
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot in the figures directory
    figures_dir = os.path.join(root_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plot_filename = f"stability_analysis_matching_{timestamp}.svg"
    plot_filepath = os.path.join(figures_dir, plot_filename)
    fig.savefig(plot_filepath, format='svg', bbox_inches='tight')
    print(f"Figure saved as: {plot_filepath}")
    
    plt.show()

if __name__ == "__main__":
    main()
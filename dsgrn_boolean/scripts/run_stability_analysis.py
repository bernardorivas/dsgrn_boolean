import DSGRN
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from dsgrn_boolean.utils.hill_stable_analysis import analyze_stability_parallel
from dsgrn_boolean.utils.sample_management import load_samples
from dsgrn_boolean.utils.dsgrn_sample_to_matrix import extract_parameter_matrices
from multiprocessing import cpu_count

def main():
    # Setup network and parameter
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    par_index = 98
    parameter = parameter_graph.parameter(par_index)
    
    # Load and process samples
    samples = load_samples(par_index, filtered=True, filter_tol=0.1)[:200]  # Remove slice to process all samples

    # Optimize d_range: Use fewer points but strategically placed
    # Focus on regions where changes are more likely to occur
    # d_range = list(range(1, 15)) + \
    #           list(range(15, 51, 5)) + \
    #           list(range(60, 101, 10))  # More sparse at higher values
    d_range = list(range(1,200))

    # Pre-allocate processed_samples list for better memory efficiency
    processed_samples = []
    processed_samples = [(extract_parameter_matrices(sample, network)) 
                        for sample in samples]
    
    # Run analysis with optimal number of processes
    n_processes = min(cpu_count(), len(processed_samples))  # Don't use more processes than samples
    results = analyze_stability_parallel(
        network,
        parameter,
        processed_samples,
        d_range=d_range,
        n_processes=n_processes
    )
    
    # Vectorized success rate calculation
    success_data = {}
    for d, sample_list in results['by_d'].items():
        num_samples = len(sample_list)
        success_array = np.array([len(stable_states) == 3 for stable_states in sample_list])
        num_success = np.sum(success_array)
        num_failures = num_samples - num_success
        
        success_data[d] = {
            "num_samples": num_samples,
            "num_success": int(num_success),  # Convert from np.int64 for JSON serialization
            "num_failures": int(num_failures),
            "success_rate": float(num_success / num_samples) if num_samples > 0 else 0,
            "failure_rate": float(num_failures / num_samples) if num_samples > 0 else 0,
        }
    
    # File operations
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save data
    data_dir = os.path.join(root_dir, "stability_data")
    os.makedirs(data_dir, exist_ok=True)
    data_filename = f"parindex_{par_index}_stability_data_{timestamp}.json"
    with open(os.path.join(data_dir, data_filename), "w") as f:
        json.dump({"par_index": par_index, "success_data": success_data}, f, indent=2)
    
    # Create and save plot
    d_values = np.array(sorted(success_data.keys()))
    success_rates = np.array([success_data[d]["success_rate"] * 100 for d in d_values])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(d_values, success_rates, color='skyblue')
    ax.set_xlabel('Hill Coefficient (d)')
    ax.set_ylabel('Matching Success (%)')
    ax.set_title(f'Matching Success Rates vs Hill Coefficient (Parameter {par_index})')
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    figures_dir = os.path.join(root_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plot_filename = f"stability_analysis_matching_{timestamp}.svg"
    fig.savefig(os.path.join(figures_dir, plot_filename), format='svg', bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    main()
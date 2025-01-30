import DSGRN
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import argparse

from dsgrn_boolean.utils.hill_analysis import analyze_hill_coefficients
from dsgrn_boolean.utils.sample_management import load_samples

def analyze_without_plotting(network, parameter, samples, d_range):
    """Wrapper for analyze_hill_coefficients that suppresses plotting"""
    # Temporarily redirect stdout to suppress print statements
    import sys
    from io import StringIO
    temp_stdout = StringIO()
    sys.stdout = temp_stdout
    
    # Run analysis
    results, summary, optimal_d, sample_results = analyze_hill_coefficients(
        network, parameter, samples, d_range
    )
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    
    return results, summary, optimal_d, sample_results

def main():
    # Define the network specification
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    
    # Create DSGRN network and parameter graph
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    
    # Parameter node we want to analyze
    par_index = 98
    parameter = parameter_graph.parameter(par_index)
    
    # Load all three types of samples
    print("\nLoading samples...")
    samples_unfiltered = load_samples(par_index, filtered=False)
    samples_filtered_01 = load_samples(par_index, filtered=True, filter_tol=0.1)
    samples_filtered_10 = load_samples(par_index, filtered=True, filter_tol=1.0)

    # For quick testing, use only the first 15 samples
    samples_unfiltered = samples_unfiltered[:1]
    samples_filtered_01 = samples_filtered_01[:1]
    samples_filtered_10 = samples_filtered_10[:1]
    
    # Define d range for analysis
    d_range = range(100, 101)
    
    # Run analysis for each set
    print("\nAnalyzing unfiltered samples...")
    results_unfiltered, _, _, _ = analyze_without_plotting(network, parameter, samples_unfiltered, d_range)
    print("\nAnalyzing filtered samples (tol=0.1)...")
    results_filtered_01, _, _, _ = analyze_without_plotting(network, parameter, samples_filtered_01, d_range)
    print("\nAnalyzing filtered samples (tol=1.0)...")
    results_filtered_10, summary, optimal_d, sample_results = analyze_without_plotting(network, parameter, samples_filtered_10, d_range)
    
    # Plot results as grouped bar graph
    plt.figure(figsize=(15, 8))
    bar_width = 0.25
    
    # Calculate bar positions
    r1 = np.arange(len(d_range))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    plt.bar(r1, results_unfiltered, bar_width, label='Unfiltered', color='steelblue', alpha=0.8)
    plt.bar(r2, results_filtered_01, bar_width, label='Filtered (tol=0.1)', color='forestgreen', alpha=0.8)
    plt.bar(r3, results_filtered_10, bar_width, label='Filtered (tol=1.0)', color='indianred', alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Hill coefficient (d)', fontsize=12)
    plt.ylabel('Percentage of matches (%)', fontsize=12)
    plt.title(f'Comparison of Sample Analyses for Parameter Node {par_index}', fontsize=14)
    
    # Set axis properties
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Set x-axis ticks in the middle of the groups
    plt.xticks([r + bar_width for r in range(len(d_range))], d_range)
    
    # Add legend
    plt.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nUnfiltered:")
    print(f"Best match: {max(results_unfiltered):.1f}% at d = {d_range[np.argmax(results_unfiltered)]}")
    print(f"Worst match: {min(results_unfiltered):.1f}% at d = {d_range[np.argmin(results_unfiltered)]}")
    
    print("\nFiltered (tol=0.1):")
    print(f"Best match: {max(results_filtered_01):.1f}% at d = {d_range[np.argmax(results_filtered_01)]}")
    print(f"Worst match: {min(results_filtered_01):.1f}% at d = {d_range[np.argmin(results_filtered_01)]}")
    
    print("\nFiltered (tol=1.0):")
    print(f"Best match: {max(results_filtered_10):.1f}% at d = {d_range[np.argmax(results_filtered_10)]}")
    print(f"Worst match: {min(results_filtered_10):.1f}% at d = {d_range[np.argmin(results_filtered_10)]}")
    print(f"Expected equilibria: {summary['expected_eq']}")

if __name__ == "__main__":
    main() 
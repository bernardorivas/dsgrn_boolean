import DSGRN
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import argparse

from dsgrn_boolean.utils.hill_analysis import analyze_hill_coefficients
from dsgrn_boolean.utils.sample_management import load_samples

def analyze_and_plot(network, parameter, samples, d_range, par_index, filtered=False, filter_tol=0.1, plot=True):
    """
    Run analysis and optionally plot results
    """
    # Run analysis
    results, summary, optimal_d, sample_results = analyze_hill_coefficients(
        network, 
        parameter, 
        samples, 
        d_range
    )
    
    if plot:
        # Plot results as a bar graph
        plt.figure(figsize=(10, 6))
        plt.bar(d_range, results, 
                width=0.8,            
                alpha=0.8,            
                color='steelblue',    
                edgecolor='black',    
                linewidth=0.5)        

        # Create title based on what analysis we're running
        if filtered:
            title = f'Parameter Node {par_index} (Filtered, tol={filter_tol})'
        else:
            title = f'Parameter Node {par_index}'

        plt.xlabel('Hill coefficient (d)', fontsize=12)
        plt.ylabel('Percentage of matches (%)', fontsize=12)
        plt.title(f'Percentage of Samples Matching DSGRN Equilibria Count\n{title}', fontsize=14)
        plt.ylim(0, 100)
        plt.xlim(0,max(d_range))

        # Add horizontal grid lines only at the integer ticks
        plt.yticks(range(0, 101, 10))
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Customize tick labels
        plt.tick_params(axis='both', which='major', labelsize=10)

        plt.show()

    # Print final statistics
    print(f"\nSummary ({title if 'title' in locals() else 'Analysis'}):")
    print(f"Expected equilibria: {summary['expected_eq']}")
    print(f"Best match percentage: {max(results):.1f}% at d = {d_range[np.argmax(results)]}")
    print(f"Worst match percentage: {min(results):.1f}% at d = {d_range[np.argmin(results)]}")
    
    return results, summary, optimal_d, sample_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Hill coefficient analysis')
    parser.add_argument('--filtered', action='store_true',
                       help='Use filtered samples')
    parser.add_argument('--filter_tol', type=float, default=0.1,
                       help='Filter tolerance (default: 0.1)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')
    args = parser.parse_args()
    
    # Define the network specification
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    
    # Create DSGRN network and parameter graph
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    
    # Parameter node we want to analyze
    par_index = 98
    parameter = parameter_graph.parameter(par_index)
    
    # Load samples with filtering option
    samples = load_samples(
        par_index, 
        filtered=args.filtered, 
        filter_tol=args.filter_tol
    )
    # Use first 30 samples for test analysis
    samples_30 = samples[:30]

    # Use first 100 samples for quicker analysis
    samples_100 = samples[:100]

    # Use first 1000 samples for overnight analysis
    samples_1000 = samples[:1000]
    
    # Define d range for analysis
    d_range = range(100, 101)
    
    # Run analysis
    samples = samples_30
    results, summary, optimal_d, sample_results = analyze_and_plot(
        network, 
        parameter, 
        samples, 
        d_range,
        par_index,
        filtered=args.filtered,
        filter_tol=args.filter_tol,
        plot=not args.no_plot
    )
    
    # Example: Track a specific sample across all d values
    sample_id = sample_results[d_range[0]]['failures'][0]  # First failing sample
    print(f"\nTracking sample {sample_id}:")
    matching_d = [d for d in d_range if sample_id in sample_results[d]['matches']]
    failing_d = [d for d in d_range if sample_id in sample_results[d]['failures']]
    print(f"Matching d values: {matching_d}")
    print(f"Failing d values: {failing_d}")

if __name__ == "__main__":
    main() 
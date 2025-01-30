import DSGRN
import argparse
from dsgrn_boolean.utils.hill_analysis import (
    analyze_hill_coefficients, 
    create_comparison_plot
)
from dsgrn_boolean.utils.sample_management import load_samples
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def main():
    parser = argparse.ArgumentParser(description='Compare Hill coefficient analyses')
    parser.add_argument('--samples', type=int, default=30)
    args = parser.parse_args()
    
    # Setup network and parameter
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    par_index = 98
    parameter = parameter_graph.parameter(par_index)
    
    # Load samples
    print("\nLoading samples...")
    samples_unf = load_samples(par_index, filtered=False)[:args.samples]
    samples_f01 = load_samples(par_index, filtered=True, filter_tol=0.1)[:args.samples]
    samples_f10 = load_samples(par_index, filtered=True, filter_tol=1.0)[:args.samples]
    
    # Run analyses without individual plots
    d_range = range(100, 101)
    print("\nAnalyzing samples...")
    results_unf, summary_unf, _, _ = analyze_hill_coefficients(network, parameter, samples_unf, d_range, show_plot=False)
    results_f01, summary_01, _, _ = analyze_hill_coefficients(network, parameter, samples_f01, d_range, show_plot=False)
    results_f10, summary_10, _, _ = analyze_hill_coefficients(network, parameter, samples_f10, d_range, show_plot=False)
    
    # Create comparison plot
    fig = create_comparison_plot(d_range, results_unf, results_f01, results_f10, par_index)
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for name, results, summary in [
        ("Unfiltered", results_unf, summary_unf),
        ("Filtered (tol=0.1)", results_f01, summary_01),
        ("Filtered (tol=1.0)", results_f10, summary_10)
    ]:
        print(f"\n{name}:")
        print(f"Best match: {max(results):.1f}% at d = {d_range[np.argmax(results)]}")
        print(f"Worst match: {min(results):.1f}% at d = {d_range[np.argmin(results)]}")
        print(f"Expected equilibria: {summary['expected_eq']}")

if __name__ == "__main__":
    main() 
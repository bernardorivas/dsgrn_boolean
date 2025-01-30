import DSGRN
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

from dsgrn_boolean.utils.hill_analysis import analyze_hill_coefficients
from dsgrn_boolean.utils.sample_management import load_samples

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
    
    # Load samples for this parameter node
    samples = load_samples(par_index)
    
    # Use first 100 samples for quicker analysis
    samples_100 = samples[:100]

    # Use first 1000 samples for overnight analysis
    samples_1000 = samples[:1000]
    
    # Define d range for analysis
    d_range = range(1, 201)
    
    # Run analysis
    results, summary, optimal_d, sample_results = analyze_hill_coefficients(
        network, 
        parameter, 
        samples_1000, 
        d_range
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
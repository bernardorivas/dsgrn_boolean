import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from .find_stable_states import find_stable_states
from .dsgrn_sample_to_matrix import extract_parameter_matrices
import matplotlib.pyplot as plt

def process_sample(args):
    """Parallel processing wrapper for sample analysis"""
    sample_idx, network, sample, d_range = args
    L, U, T = extract_parameter_matrices(sample, network)
    results = []
    prev_states = None
    
    for d in reversed(d_range):  # Process high d first
        stable, _ = find_stable_states(L, U, T, d, prev_states)
        # Store whether we found exactly 3 stable states
        results.append((d, len(stable) == 3))  # Changed to store boolean match
        prev_states = stable  # Carry states forward
        
    return sample_idx, sorted(results, reverse=True)

def analyze_stability_parallel(network, parameter, samples, d_range=range(100, 0, -1), n_processes=None):
    """
    Parallel stability analysis with continuity tracking.
    """
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    
    # Prepare arguments for parallel processing
    args = [(i, network, s, d_range) for i, s in enumerate(samples)]
    
    with Pool(n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_sample, args),
            total=len(samples),
            desc="Processing samples"
        ))
    
    # Reorganize results by d-value
    d_results = {d: [] for d in d_range}
    for sample_idx, sample_data in results:
        for d, matches in sample_data:
            d_results[d].append(matches)
    
    # Calculate percentage of matches for each d
    return {
        'by_sample': results,
        'by_d': {d: (sum(matches) / len(matches)) * 100 for d, matches in d_results.items()}
    }

def plot_stability_results(results, d_range, par_index, expected_stable=3):
    """
    Create a bar plot showing percentage of samples with correct number of stable equilibria.
    
    Args:
        results: Output from analyze_stability_parallel
        d_range: Range of d values used
        par_index: Parameter node index for title
        expected_stable: Number of expected stable equilibria
    """
    # Process results 
    d_values = sorted(results['by_d'].keys())
    percentages = [results['by_d'][d] for d in d_values] 
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(d_values, percentages,
                   width=0.8,
                   color='steelblue',
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=0.5)
    
    # Style matching hill_analysis
    plt.xlabel('Hill Coefficient (d)', fontsize=12)
    plt.ylabel('Samples with 3 Stable Equilibria (%)', fontsize=12)
    plt.title(f'Stability Analysis for Parameter Node {par_index}', fontsize=14)
    plt.ylim(0, 100)
    plt.xlim(min(d_range)-1, max(d_range)+1)
    plt.yticks(range(0, 101, 10))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom',
                 fontsize=8)
    
    return plt.gcf() 
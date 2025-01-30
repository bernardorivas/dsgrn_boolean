import DSGRN
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from itertools import product
from tqdm import tqdm
from multiprocessing import Pool

from .dsgrn_sample_to_matrix import extract_parameter_matrices
from .newton import newton_method
from dsgrn_boolean.models.hill import hill

def process_parameter_set(args):
    """
    Helper function to process a single parameter set.
    Must be at module level for multiprocessing.
    """
    L, U, T, d, expected_eq, sample_index = args
    
    # Define x and y coordinates separately
    x_coords = [
        T[0,0],              # Threshold
        T[0,1],              # Threshold
        L[0,0] + L[1,0],     # x-nullcline
        U[0,0] + L[1,0],     # x-nullcline
        L[0,0] + U[1,0],     # x-nullcline
        U[0,0] + U[1,0]      # x-nullcline
    ]
    
    y_coords = [
        T[1,0],              # Threshold
        T[1,1],              # Threshold
        L[0,1] * L[1,1],     # y-nullcline
        U[0,1] * L[1,1],     # y-nullcline
        L[0,1] * U[1,1],     # y-nullcline
        U[0,1] * U[1,1]      # y-nullcline
    ]
    
    # Create all possible combinations
    specific_points = [np.array([x, y]) for x, y in product(x_coords, y_coords)]
    
    n_zeros = count_zeros(L, U, T, d, expected_eq, specific_points=specific_points)
    return sample_index, (n_zeros == expected_eq)

def analyze_hill_coefficients(network, parameter, samples, d_range=range(1, 51), n_processes=None):
    """
    Analyze how different Hill coefficients match DSGRN equilibria predictions.
    """
    if n_processes is None:
        n_processes = max(1, os.cpu_count() - 1)
    
    # Get expected number of equilibria
    expected_eq = len(DSGRN.EquilibriumCells(parameter))
    
    # Timing estimates
    samples_per_process = len(samples) / n_processes
    time_per_batch = 11.0  # seconds (measured from actual runtime)
    total_batches = len(d_range)
    estimated_seconds = time_per_batch * total_batches
    
    # Determine if we should show detailed output
    show_detailed = len(d_range) <= 10
    
    print(f"Processing {len(samples)} samples for {len(d_range)} d values using {n_processes} processes")
    print(f"Estimated runtime: {estimated_seconds:.1f} seconds ({estimated_seconds/60:.1f} minutes)")
    
    # Pre-compute all parameter matrices
    print("Pre-computing parameter matrices...")
    parameter_matrices = []
    for sample in tqdm(samples):
        L, U, T = extract_parameter_matrices(sample, network)
        parameter_matrices.append((L, U, T))
    
    # Analyze each d value
    results = []
    sample_results = {}
    
    with Pool(n_processes) as pool:
        for d in tqdm(d_range, desc="Testing Hill coefficients"):
            args = [(L, U, T, d, expected_eq, i) for i, (L, U, T) in enumerate(parameter_matrices)]
            sample_matches = pool.map(process_parameter_set, args)
            
            matches = sum(1 for match in sample_matches if match[1])
            percentage = (matches / len(samples)) * 100
            results.append(percentage)
            
            sample_results[d] = {
                'matches': [i for i, (i, match) in enumerate(sample_matches) if match],
                'failures': [i for i, (i, match) in enumerate(sample_matches) if not match]
            }
            
            # Only show detailed output if d_range is small
            if show_detailed:
                print(f"d={d}: {percentage:.1f}% match ({matches}/{len(samples)} samples)")
                print(f"  Matched samples: {sample_results[d]['matches']}")
                print(f"  Failed samples: {sample_results[d]['failures']}")
    
    # Find optimal d value
    optimal_d = d_range[np.argmax(results)]
    
    # Create summary statistics
    summary = {
        "expected_equilibria": expected_eq,
        "best_match": max(results),
        "best_match_d": optimal_d,
        "worst_match": min(results),
        "worst_match_d": d_range[np.argmin(results)]
    }
    
    # Create and save visualization
    plt.figure(figsize=(10, 6))
    plt.bar(d_range, results,
            width=0.8,
            alpha=0.8,
            color='steelblue',
            edgecolor='black',
            linewidth=0.5)

    plt.xlabel('Hill coefficient (d)', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Percentage of samples whose number of equilibria matches DSGRN equilibria count', fontsize=14)
    plt.ylim(0, 100)
    plt.xlim(0, max(d_range))

    plt.yticks(range(0, 101, 10))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # Create figures directory if it doesn't exist
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(root_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save figure with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"hill_analysis_d{min(d_range)}-{max(d_range)}_{timestamp}.svg"
    filepath = os.path.join(figures_dir, filename)
    plt.savefig(filepath, format='svg', bbox_inches='tight')
    print(f"\nFigure saved as: {filename}")
    
    plt.show()
    
    # Print summary
    print(f"\nSummary:")
    print(f"Expected equilibria: {summary['expected_equilibria']}")
    print(f"Best match: {summary['best_match']:.1f}% at d = {summary['best_match_d']}")
    print(f"Worst match: {summary['worst_match']:.1f}% at d = {summary['worst_match_d']}")
    
    return results, summary, optimal_d, sample_results

def count_zeros(L, U, T, d, expected_eq, n_grid=20, specific_points=None):
    """
    Count number of distinct zeros for given parameters.
    
    Args:
        L: Lower bounds matrix
        U: Upper bounds matrix
        T: Threshold matrix
        d: Hill coefficient
        expected_eq: Expected number of equilibria
        n_grid: Number of grid points for initial conditions (default: 20)
        specific_points: List of additional initial conditions to try first
        
    Returns:
        int: Number of distinct equilibrium points found
    """
    system, jacobian = hill(L, U, T, d)
    
    zeros = []
    
    # Try specific points first if provided
    if specific_points is not None:
        for x0 in specific_points:
            x, converged, _ = newton_method(system, x0, df=jacobian)
            if converged:
                # Check if this zero is already found
                is_new = True
                for z in zeros:
                    if np.allclose(z, x, rtol=1e-5):
                        is_new = False
                        break
                if is_new:
                    zeros.append(x)
                    # Early stopping if we found the expected number of equilibria
                    if len(zeros) == expected_eq:
                        return len(zeros)
    
    # If we haven't found all expected equilibria, try grid points
    x_max = U[0,0] + U[0,1]
    y_max = U[1,0] * U[1,1]
    x_grid = np.linspace(0, x_max, n_grid)
    y_grid = np.linspace(0, y_max, n_grid)
    initial_conditions = [np.array([x, y]) for x in x_grid for y in y_grid]
    
    for x0 in initial_conditions:
        x, converged, _ = newton_method(system, x0, df=jacobian)
        if converged:
            # Check if this zero is already found
            is_new = True
            for z in zeros:
                if np.allclose(z, x, rtol=1e-5):
                    is_new = False
                    break
            if is_new:
                zeros.append(x)
                # Early stopping if we found the expected number of equilibria
                if len(zeros) == expected_eq:
                    return len(zeros)
    
    return len(zeros)
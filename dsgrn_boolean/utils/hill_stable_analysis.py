import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from .find_stable_states import find_stable_states
from .dsgrn_sample_to_matrix import extract_parameter_matrices
import matplotlib.pyplot as plt
import os
from dsgrn_boolean.models.hill import hill
from dsgrn_boolean.utils.newton import newton_method

def process_sample(args):
    """
    Process a single sample.
    Args:
      args: a tuple (L, U, T, d, extra) where:
         - L, U, T are the parameter matrices,
         - d is the Hill exponent,
         - extra is a placeholder for future use.
    Returns:
      A list of stable equilibrium points found for this sample.
    """
    L, U, T, d, extra = args
    system, jacobian = hill(L, U, T, d)
    
    # Define the grid for initial conditions; use a coarse grid for speed.
    try:
        x_max = 1.5 * (U[0,0] + U[1,0])
        y_max = 1.5 * (U[0,1] * U[1,1])
    except Exception as e:
        # Fallback if grid definition encounters issues.
        x_max = 1.0
        y_max = 1.0
        
    n_grid = 10  # coarse grid to speed up; adjust if needed
    x_grid = np.linspace(0, x_max, n_grid)
    y_grid = np.linspace(0, y_max, n_grid)
    initial_conditions = [np.array([x, y]) for x in x_grid for y in y_grid]

    stable_equilibria = []
    for x0 in initial_conditions:
        x_eq, converged, _ = newton_method(system, x0, df=jacobian)
        if converged:
            # Avoid duplicate equilibria.
            if not any(np.allclose(x_eq, eq, rtol=1e-8) for eq in stable_equilibria):
                # Determine stability.
                J = jacobian(x_eq)
                eigenvals = np.linalg.eigvals(J)
                if all(np.real(eigenvals) < 0):
                    stable_equilibria.append(x_eq)
    return stable_equilibria

def analyze_stability_parallel(network, parameter, processed_samples, d_range, visualize=False):
    """
    Analyze stability for the provided samples in parallel.
    
    Args:
        network: The DSGRN network (unused in this snippet but provided for context).
        parameter: The parameter from the DSGRN parameter graph.
        processed_samples: A list of tuples (L, U, T, d, extra) as input to process_sample.
        d_range: The range of Hill coefficients (for grouping results).
        visualize (bool): If True, create plots (not used in this multiprocessing version).
        
    Returns:
        A dictionary with keys 'by_d' where for each d the value is a list of lists containing
        the stable equilibria for that sample.
    """
    total = len(processed_samples)
    pool = mp.Pool(processes=mp.cpu_count())
    chunk_size = max(1, total // mp.cpu_count())

    import time
    start_time = time.time()
    results = []
    progress_bar = tqdm(total=total, desc="Processing samples", dynamic_ncols=True)
    
    # Process samples with progress tracking
    for stable_eq in pool.imap_unordered(process_sample, processed_samples, chunksize=chunk_size):
        results.append(stable_eq)
        progress_bar.update(1)
        
        # Calculate time statistics
        elapsed = time.time() - start_time
        percent_complete = len(results) / total
        remaining = (elapsed / percent_complete - elapsed) if percent_complete > 0 else 0
        progress_bar.set_postfix({"ETA": f"{remaining:.1f} sec"})
    
    progress_bar.close()
    pool.close()
    pool.join()

    # Group results by Hill coefficient
    results_by_d = {}
    for sample_info, stable_eq in zip(processed_samples, results):
        d = sample_info[3]
        results_by_d.setdefault(d, []).append(stable_eq)
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.1f} seconds")
    
    return {"by_d": results_by_d}

def plot_stability_results(results, d_range, par_index):
    """
    Plot the stability results as a bar chart of percentage match.
    For each Hill coefficient (d), we display the percentage of samples
    that have a given number of stable states.
    
    Args:
        results (dict): The dictionary returned by analyze_stability_parallel.
        d_range (range): The range of d values that were analyzed.
        par_index (int): The parameter index (for title display).
    
    Returns:
        plt.Figure: The matplotlib figure object.
    """
    
    d_values = sorted(results['by_d'].keys())
    
    # For each d value, count how many samples have a given number of stable states.
    percentages_by_d = {}  # This will be {d: {num_stable: percentage, ...}, ...}
    
    for d in d_values:
        sample_list = results['by_d'][d]
        num_samples = len(sample_list)
        counts = {}
        for stable_states in sample_list:
            num_stable = len(stable_states)
            counts[num_stable] = counts.get(num_stable, 0) + 1
        # Convert counts to percentages
        percentages = {num: count / num_samples for num, count in counts.items()}
        percentages_by_d[d] = percentages
    
    # Determine the set of all possible "number of stable states" across all d
    all_stable_numbers = set()
    for perc in percentages_by_d.values():
        all_stable_numbers.update(perc.keys())
    all_stable_numbers = sorted(all_stable_numbers)
    
    # Create grouped bar chart data
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # The width of the bars and positions: we arrange each d's bars side-by-side.
    num_d = len(d_values)
    bar_width = 0.8 / num_d
    
    for i, d in enumerate(d_values):
        # For each possible number of stable states, get the percentage (or 0 if missing)
        percentages = [percentages_by_d[d].get(n, 0) for n in all_stable_numbers]
        # Compute positions shifted for each d group
        positions = np.array(all_stable_numbers, dtype=float) + i * bar_width
        ax.bar(positions, percentages, width=bar_width, label=f'd={d}')
    
    ax.set_xticks(np.array(all_stable_numbers) + 0.8/2)
    ax.set_xticklabels(all_stable_numbers)
    ax.set_xlabel('Number of Stable States')
    ax.set_ylabel('Percentage of Samples')
    ax.set_title(f'Stability Analysis Matching Results (Parameter {par_index})')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small', ncol=2)
    
    return fig

# For testing, you can run this module directly.
if __name__ == "__main__":
    # Dummy test inputs.
    L = np.array([[1, 0], [0, 1]])
    U = np.array([[2, 0], [0, 2]])
    T = np.array([[1.5, 0], [0, 1.5]])
    processed_samples = [(L, U, T, d, None) for d in range(1, 6)]
    results = analyze_stability_parallel(None, None, processed_samples, d_range=range(1,6))
    print(results)
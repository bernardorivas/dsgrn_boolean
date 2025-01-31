import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from .find_stable_states import find_stable_states
from .dsgrn_sample_to_matrix import extract_parameter_matrices
import matplotlib.pyplot as plt
import os
from dsgrn_boolean.models.hill import hill
from dsgrn_boolean.utils.newton import newton_method
from scipy.integrate import solve_ivp
import time
from multiprocessing import Pool

def integrate_system(system, x0, t_span=(0, 20), rtol=1e-4):
    """Integrate ODE system to find stable equilibria"""
    def event(t, x):
        return np.linalg.norm(system(x)) - 1e-4  # Relaxed tolerance
    event.terminal = True
    event.direction = -1
    
    sol = solve_ivp(
        lambda t, x: system(x),
        t_span,
        x0,
        method='RK45',
        events=event,
        rtol=rtol,
        atol=1e-6,
        max_step=0.1
    )
    
    final_deriv = np.linalg.norm(system(sol.y[:, -1]))
    has_converged = final_deriv < 1e-3 or sol.status == 1
    
    return sol.y[:, -1], has_converged

def is_new_point(point, existing_points, rtol=1e-8):
    """Check if a point is significantly different from existing points"""
    return not any(np.allclose(point, p, rtol=rtol) for p in existing_points)

def process_sample(args):
    """
    Process a single sample with enhanced stable state finding strategy.
    """
    L, U, T, d, prev_states = args  # Note: prev_states is now used
    system, jacobian = hill(L, U, T, d)
    
    print(f"\nProcessing sample for d={d}")
    
    stable_equilibria = []
    
    # Step 0: If we have previous states, try them first
    if prev_states is not None and len(prev_states) > 0:
        print(f"Trying previous states from d>{d}: {prev_states}")
        for x0 in prev_states:
            x_eq, converged, _ = newton_method(system, x0, df=jacobian)
            if converged:
                J = jacobian(x_eq)
                eigenvals = np.linalg.eigvals(J)
                if all(np.real(eigenvals) < 0) and is_new_point(x_eq, stable_equilibria):
                    stable_equilibria.append(x_eq)
                    print(f"Found stable state from previous d at {x_eq}")
    
    # If we haven't found all three stable states, proceed with the other strategies
    if len(stable_equilibria) < 3:
        # Step 1: Try specific points
        specific_points = [
            np.array([T[0,1], T[1,0]]),  # Point from threshold matrix
            np.array([0., 0.]),          # Origin
            np.array([U[0,0], U[1,1]])   # Upper bounds point
        ]
        
        print(f"Trying specific points: {specific_points}")
        
        for x0 in specific_points:
            x_eq, converged, _ = newton_method(system, x0, df=jacobian)
            if converged:
                J = jacobian(x_eq)
                eigenvals = np.linalg.eigvals(J)
                if all(np.real(eigenvals) < 0) and is_new_point(x_eq, stable_equilibria):
                    stable_equilibria.append(x_eq)
                    print(f"Found stable state at {x_eq}")
    
        # Step 2: If still haven't found all three, try grid around specific points
        if len(stable_equilibria) < 3:
            print("Didn't find all stable states, trying grid around specific points...")
            for center in specific_points:
                grid_size = 5
                perturbations = np.linspace(-0.5, 0.5, grid_size)
                for dx in perturbations:
                    for dy in perturbations:
                        x0 = center + np.array([dx, dy])
                        x_eq, converged, _ = newton_method(system, x0, df=jacobian)
                        if converged:
                            J = jacobian(x_eq)
                            eigenvals = np.linalg.eigvals(J)
                            if all(np.real(eigenvals) < 0) and is_new_point(x_eq, stable_equilibria):
                                stable_equilibria.append(x_eq)
                                print(f"Found additional stable state at {x_eq}")
        
        # Step 3: If still haven't found all three, try forward integration
        if len(stable_equilibria) < 3:
            print("Still missing stable states, trying forward integration...")
            additional_points = [
                np.array([0, U[1,1]]),
                np.array([U[0,0], 0]),
                np.array([U[0,0]/2, U[1,1]/2])
            ]
            
            for x0 in additional_points:
                x_integrated, converged = integrate_system(system, x0)
                if converged:
                    x_eq, converged, _ = newton_method(system, x_integrated, df=jacobian)
                    if converged:
                        J = jacobian(x_eq)
                        eigenvals = np.linalg.eigvals(J)
                        if all(np.real(eigenvals) < 0) and is_new_point(x_eq, stable_equilibria):
                            stable_equilibria.append(x_eq)
                            print(f"Found stable state through integration at {x_eq}")
    
    print(f"Final count: Found {len(stable_equilibria)} stable states for d={d}")
    return stable_equilibria

def process_sample_sequence(sample, d_range):
    """
    Process a single sample through all d values in sequence (large to small).
    
    Args:
        sample: tuple (L, U, T) matrices for this sample
        d_range: list of d values to process
    Returns:
        dict: Results for this sample, keyed by d
    """
    L, U, T = sample
    d_values = sorted(d_range, reverse=True)  # Ensure large to small
    results = {}
    prev_states = None
    
    for d in d_values:
        stable_eq = process_sample((L, U, T, d, prev_states))
        results[d] = stable_eq
        prev_states = stable_eq
    
    return results

def analyze_stability_parallel(network, parameter, samples, d_range, n_processes=None):
    """
    Parallel analysis of samples, where each process handles a complete
    d-sequence for its assigned samples.
    
    Args:
        network: DSGRN network
        parameter: DSGRN parameter
        samples: list of (L, U, T) samples to process
        d_range: list of d values to process
        n_processes: number of processes to use
    Returns:
        dict: Combined results from all processes
    """
    if n_processes is None:
        n_processes = os.cpu_count()
    
    # Split samples into chunks for each process
    chunk_size = len(samples) // n_processes + (1 if len(samples) % n_processes else 0)
    sample_chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]
    
    # Process chunks in parallel
    with Pool(processes=n_processes) as pool:
        chunk_results = list(tqdm(
            pool.starmap(
                process_sample_sequence,
                [(chunk, d_range) for chunk in sample_chunks]
            ),
            total=len(sample_chunks),
            desc="Processing sample chunks"
        ))
    
    # Combine results from all chunks
    combined_results = {"by_d": {}}
    for d in d_range:
        combined_results["by_d"][d] = []
        for chunk_result in chunk_results:
            combined_results["by_d"][d].extend(chunk_result[d])
    
    return combined_results

def plot_stability_results(results, d_range, par_index):
    """Plot success rates for each d value"""
    d_values = sorted(results['by_d'].keys())
    success_rates = []
    
    for d in d_values:
        samples = results['by_d'][d]
        num_samples = len(samples)
        num_success = sum(1 for stable_states in samples if len(stable_states) > 0)
        success_rates.append(100 * num_success / num_samples)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(d_values, success_rates, color='skyblue')
    
    ax.set_xlabel('Hill Coefficient (d)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Stability Analysis Success Rates (Parameter {par_index})')
    ax.grid(True, alpha=0.3)
    
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
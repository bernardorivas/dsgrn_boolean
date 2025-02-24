import numpy as np
import time

from scipy.integrate import solve_ivp
from multiprocessing import Pool
from tqdm import tqdm
from itertools import product

from dsgrn_boolean.models.hill import hill
from dsgrn_boolean.utils.newton import newton_method
from dsgrn_boolean.utils.dsgrn_sample_to_matrix import extract_parameter_matrices

def integrate_system(system, x0, t_span=(0, 20), rtol=1e-4):
    """Integrate ODE system to find stable equilibria"""
    sol = solve_ivp(
        lambda t, x: system(x),
        t_span,
        x0,
        method='RK45',
        rtol=rtol,
        atol=1e-6,
        max_step=0.1
    )
    return sol.y[:, -1], np.linalg.norm(system(sol.y[:, -1])) < 1e-4

def is_new_point(point, existing_points, rtol=1e-4):
    if not existing_points:
        return True
    return not any(np.allclose(point, p, rtol=rtol) for p in existing_points)

def find_equilibria(args):
    """Find stable equilibria for a single parameter set at given Hill coefficient."""
    L, U, T, d, prev_states, n_equilibria = args
    system, jacobian = hill(L, U, T, d)
    stable_equilibria = []
    
    # Try previous states first
    if prev_states is not None and len(prev_states) > 0:
        for x0 in prev_states:
            x_eq, converged, _ = newton_method(system, x0, df=jacobian)
            if converged:
                J = jacobian(x_eq)
                eigenvals = np.linalg.eigvals(J)
                if all(np.real(eigenvals) < 0) and is_new_point(x_eq, stable_equilibria):
                    stable_equilibria.append(x_eq)
    
    # If needed, try additional points
    if len(stable_equilibria) < n_equilibria:
        new_equilibria = find_additional_equilibria(system, jacobian, L, U, T, n_equilibria)
        for p in new_equilibria: 
            if is_new_point(p, stable_equilibria):
                stable_equilibria.append(p)
    
    return stable_equilibria

def find_stable_equilibria_in_parallel(network, samples, n_equilibria, d_range, n_processes=None):
    """Analyze stability of samples in parallel."""
    # Process samples
    processed_samples = [extract_parameter_matrices(sample, network) 
                        for sample in samples]

    # Create tasks list - one task per sample
    tasks = [(sample, d_range, n_equilibria) for sample in processed_samples]
    
    # Process samples in parallel
    print("\nStarting computation...")
    
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_sample, tasks),
            total=len(tasks),
            desc="Processing samples"
        ))
    
    # Combine results
    combined_results = {"by_d": {d: [] for d in d_range}}
    for sample_result in results:
        for d, stable_states in sample_result.items():
            combined_results["by_d"][d].append(stable_states)
    
    return combined_results

def process_single_sample(args):
    """Process all d-values for a single sample."""
    sample, d_range, n_equilibria = args
    L, U, T = sample
    results = {}
    
    prev_states = None
    for d in sorted(d_range, reverse=True):
        results[d] = find_equilibria((L, U, T, d, prev_states, n_equilibria))
        prev_states = results[d]
    # print(f"Sample processed: {results}")    
    return results

def find_additional_equilibria(system, jacobian, L, U, T, n_equilibria):
    """Find additional equilibria using various strategies."""
    stable_equilibria = []
    
    # Step 1: Try specific points
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
    
    for x0 in specific_points:
        x_eq, converged, _ = newton_method(system, x0, df=jacobian)
        if converged:
            J = jacobian(x_eq)
            eigenvals = np.linalg.eigvals(J)
            if all(np.real(eigenvals) < 0) and is_new_point(x_eq, stable_equilibria):
                stable_equilibria.append(x_eq)
                if len(stable_equilibria) == n_equilibria:
                    return stable_equilibria
    
    # Step 2: If still haven't found all, try some random points in the square
    if len(stable_equilibria) < n_equilibria:
        x_max = 1.1 * max(x_coords)
        y_max = 1.1 * max(y_coords)
        num_random_points = 100
        random_points = [np.array([np.random.uniform(0, x_max), np.random.uniform(0, y_max)]) for _ in range(num_random_points)]
        for x0 in random_points:
            x_eq, converged, _ = newton_method(system, x0, df=jacobian)
            if converged:
                J = jacobian(x_eq)
                eigenvals = np.linalg.eigvals(J)
                if all(np.real(eigenvals) < 0) and is_new_point(x_eq, stable_equilibria):
                    stable_equilibria.append(x_eq)
                    if len(stable_equilibria) == n_equilibria:
                        return stable_equilibria

    # Step 3: If still haven't found all, try a grid in the rectangle
    # if len(stable_equilibria) < n_equilibria:
    #     grid_size = 10
    #     x_points = np.linspace(0, 1.1*max(x_coords), grid_size)[1:-1]
    #     y_points = np.linspace(0, 1.1*max(y_coords), grid_size)[1:-1]
    #     grid = [np.array([x, y]) for x, y in product(x_points, y_points)]
    #     for x0 in grid:
    #         x_eq, converged, _ = newton_method(system, x0, df=jacobian)
    #         if converged:
    #             J = jacobian(x_eq)
    #             eigenvals = np.linalg.eigvals(J)
    #             if all(np.real(eigenvals) < 0) and is_new_point(x_eq, stable_equilibria):
    #                 stable_equilibria.append(x_eq)
    #                 if len(stable_equilibria) == n_equilibria:
    #                     return stable_equilibria

    # Step 4: If still haven't found all, try forward integration
    # if len(stable_equilibria) < n_equilibria:
    #     for x0 in specific_points:
    #         x_integrated, _ = integrate_system(system, x0)
    #         x_eq, converged, _ = newton_method(system, x_integrated, df=jacobian)
    #         if converged:
    #             J = jacobian(x_eq)
    #             eigenvals = np.linalg.eigvals(J)
    #             if all(np.real(eigenvals) < 0) and is_new_point(x_eq, stable_equilibria):
    #                 stable_equilibria.append(x_eq)
    #                 if len(stable_equilibria) == n_equilibria:
    #                     return stable_equilibria
    
    return stable_equilibria
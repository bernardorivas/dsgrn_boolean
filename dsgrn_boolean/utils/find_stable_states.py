import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from .newton import newton_method
from dsgrn_boolean.models.hill import hill

def integrate_system(system, x0, t_span=(0, 10), rtol=1e-6):
    """Integrate ODE system to find stable equilibria"""
    def event(t, x):
        return np.linalg.norm(system(x))
    event.terminal = True
    event.direction = -1
    
    sol = solve_ivp(
        lambda t, x: system(x),
        t_span,
        x0,
        method='RK45',
        events=event,
        rtol=rtol,
        atol=1e-8
    )
    
    if sol.status == 1:  # Event occurred (converged)
        return sol.y[:, -1], True
    return sol.y[:, -1], False

def find_stable_states(L, U, T, d, previous_states=None, use_newton_backup=False):
    """
    Find stable equilibrium points using ODE integration and Newton's method.
    
    Args:
        L: Lower bounds matrix
        U: Upper bounds matrix
        T: Threshold matrix
        d: Hill coefficient
        previous_states: List of states from previous d value (for continuity)
        use_newton_backup: Whether to use Newton's method as backup
        
    Returns:
        tuple: (stable_points, unstable_points) containing lists of stable/unstable equilibria
    """
    system, jacobian = hill(L, U, T, d)
    unique_points = []
    
    # 1. Try previous states first
    if previous_states:
        for x0 in previous_states:
            x, converged = integrate_system(system, x0)
            if converged and not any(np.allclose(x, p) for p in unique_points):
                unique_points.append(x)
    
    # 2. Integrate from specific points
    specific_points = [
        np.array([T[0,0], T[1,0]]),
        np.array([T[0,1], T[1,1]]),
        np.array([L[0,0] + L[1,0], L[0,1] * L[1,1]]),
        np.array([U[0,0] + L[1,0], U[0,1] * L[1,1]]),
        np.array([L[0,0] + U[1,0], L[0,1] * U[1,1]]),
        np.array([U[0,0] + U[1,0], U[0,1] * U[1,1]])
    ]
    
    for x0 in specific_points:
        x, converged = integrate_system(system, x0)
        if converged and not any(np.allclose(x, p) for p in unique_points):
            unique_points.append(x)
    
    # 3. Newton backup if needed
    if use_newton_backup and len(unique_points) < 3:
        print("Using Newton backup search...")
        search_regions = [
            (T[0,1]-h, T[0,1]+h, T[1,0]-h, T[1,0]+h),
            (L[0,0]+L[1,0]-h, L[0,0]+L[1,0]+h, U[0,1]*U[1,1]-h, U[0,1]*U[1,1]+h),
            (L[0,0]+L[1,0]-h, L[0,0]+L[1,0]+h, U[0,1]*L[1,1]-h, U[0,1]*L[1,1]+h)
        ]
        
        for xmin, xmax, ymin, ymax in search_regions:
            x_vals = np.linspace(xmin, xmax, n_grid)
            y_vals = np.linspace(ymin, ymax, n_grid)
            for x in x_vals:
                for y in y_vals:
                    x0 = np.array([x, y])
                    x, converged = integrate_system(system, x0)
                    if converged and not any(np.allclose(x, p) for p in unique_points):
                        unique_points.append(x)
    
    # Classify stability
    stable_points = []
    unstable_points = []
    for point in unique_points:
        J = jacobian(point)
        eigenvalues = np.linalg.eigvals(J)
        if all(np.real(eigenvalues) < 0):
            stable_points.append(point)
        else:
            unstable_points.append(point)
    
    return stable_points, unstable_points 
import numpy as np

def newton_method(f, x0, max_iter=100, df=None, abs_tol=1e-6, rel_tol=1e-12):
    """
    Implements Newton's method for a system of equations f(x) = 0, where f: R^n -> R^n.

    Args:
        f: The function representing the system of equations. 
           Takes a numpy array (x) of shape (n,) and returns a numpy array of shape (n,).
        x0: The initial guess, a numpy array of shape (n,).
        tol: The tolerance for convergence.
        max_iter: The maximum number of iterations.
        df: (Optional) The Jacobian matrix of f, a function that takes a numpy array (x) 
            of shape (n,) and returns a numpy array of shape (n, n). 
            If not provided, finite differences will be used (see alternatives below).
        abs_tol: The absolute tolerance for convergence.
        rel_tol: The relative tolerance for convergence.

    Returns:
        A tuple (x, converged, iter_count):
            x: The approximated root, a numpy array of shape (n,).
            converged: A boolean indicating whether the method converged.
            iter_count: The number of iterations performed.
    """

    x = x0.copy()
    converged = False
    iter_count = 0

    for _ in range(max_iter):
        iter_count += 1

        if df is not None:
            J = df(x)
        else:
            # Alternative: Use finite differences to approximate the Jacobian
            J = approximate_jacobian(f, x) 

        fx = f(x)

        # Solve J * delta_x = -f(x) for delta_x
        try:
            delta_x = np.linalg.solve(J, -fx)
        except np.linalg.LinAlgError:
            print("Singular Jacobian encountered. Newton's method may fail to converge.")
            return x, False, iter_count

        x_new = x + delta_x

        # Check convergence conditions
        if np.linalg.norm(delta_x) < abs_tol or np.linalg.norm(delta_x) < rel_tol * np.linalg.norm(x_new):
            converged = True
            x = x_new
            break

        x = x_new

    # if not converged:
    #     raise RuntimeError("Newton's method did not converge after max_iter iterations.")

    return x, converged, iter_count


def approximate_jacobian(f, x, h=1e-5):
    """
    Approximates the Jacobian matrix of f using finite differences.

    Args:
        f: The function, takes a numpy array (x) and returns a numpy array.
        x: The point at which to evaluate the Jacobian, a numpy array.
        h: The step size for finite differences.

    Returns:
        The approximated Jacobian matrix, a numpy array of shape (n, n).
    """
    n = len(x)
    J = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x_plus_h = x.copy()
            x_minus_h = x.copy()
            
            x_plus_h[j] += h
            x_minus_h[j] -= h
            
            J[i, j] = (f(x_plus_h)[i] - f(x_minus_h)[i]) / (2 * h)

    return J

def jacobian(f, x):
    """
    Computes the Jacobian matrix of f using finite differences.
    
    Args:
        f: The function, takes a numpy array (x) and returns a numpy array.
        x: The point at which to evaluate the Jacobian, a numpy array.
        
    Returns:
        The Jacobian matrix, a numpy array of shape (n, n).
    """
    return approximate_jacobian(f, x)
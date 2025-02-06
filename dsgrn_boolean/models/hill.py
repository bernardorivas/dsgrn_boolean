import numpy as np
import warnings

# Suppress overflow warnings globally for this module
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

class HillFunction:
    def __init__(self, L, U, theta, d):
        """
        Initializes a Hill function.

        Args:
            L (float): The lower value.
            U (float): The upper value.
            theta (float): The threshold value.
            d (float): The Hill coefficient/exponent.
        """
        self.L = L
        self.U = U
        self.theta = theta
        self.d = d

    def _compute_ratio(self, x):
        """Safely compute (x/theta)^d using vectorized operations."""
        ratio = np.zeros_like(x, dtype=float)
        positive = x > 0
        
        if np.any(positive):
            # Vectorized computation
            log_ratio = self.d * np.log(np.where(positive, x/self.theta, 1.0))
            ratio[positive] = np.exp(np.clip(log_ratio[positive], -700, 700))
        
        return ratio

    def __call__(self, x):
        """Evaluates the Hill function using vectorized operations."""
        x = np.asarray(x)
        ratio = self._compute_ratio(x)
        
        # Vectorized computation
        result = np.full_like(x, self.L)
        result += np.where(np.isinf(ratio), 
                          self.U - self.L,  # where infinite
                          (self.U - self.L) * ratio / (1 + ratio))  # elsewhere
        return result

    def derivative(self, x):
        """Evaluates the derivative using vectorized operations."""
        x = np.asarray(x)
        ratio = self._compute_ratio(x)
        
        result = np.zeros_like(x)
        valid = (x > 0) & ~np.isinf(ratio) & (ratio != 0)
        result[valid] = (self.U - self.L) * self.d * ratio[valid] / (x[valid] * (1 + ratio[valid])**2)
        return result

def hill(L, U, T, d):
    """
    Create the Hill function system and its Jacobian.
    
    Args:
        L: Lower bounds matrix
        U: Upper bounds matrix
        T: Threshold matrix
        d: Hill function steepness parameter
        
    Returns:
        Tuple (system, jacobian) where:
            system: returns f(x) in x'=f(x)
            jacobian: returns df/dx
    """
    # Create Hill functions
    h00 = HillFunction(L[0,0], U[0,0], T[0,0], d)
    h10 = HillFunction(L[1,0], U[1,0], T[1,0], d)
    h01 = HillFunction(U[0,1], L[0,1], T[0,1], d)
    h11 = HillFunction(L[1,1], U[1,1], T[1,1], d)
    
    def system(x):
        """Compute right-hand side of the ODE system."""
        return np.array([
            -x[0] + h00(x[0]) + h10(x[1]),
            -x[1] + h01(x[0]) * h11(x[1])
        ])
    
    def jacobian(x):
        """Compute the Jacobian matrix."""
        return -np.eye(2) + np.array([
            [h00.derivative(x[0]),     h10.derivative(x[1])],
            [h01.derivative(x[0])*h11(x[1]),   h01(x[0])*h11.derivative(x[1])]
        ])
    
    return system, jacobian
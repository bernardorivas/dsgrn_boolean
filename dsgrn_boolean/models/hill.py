import numpy as np

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

    def __call__(self, x):
        """
        Evaluates the Hill function.

        Args:
            x (float or numpy array): Input value(s).

        Returns:
            float or numpy array: The Hill function value(s) at x.
        """
        if isinstance(x, (float, int)):
            if x < 0:
                return self.L
            else:
                return self.L + (self.U - self.L) * (x**self.d) / (self.theta**self.d + x**self.d)
        else:
            result = np.where(x < 0, self.L, self.L + (self.U - self.L) * (x**self.d) / (self.theta**self.d + x**self.d))
            return result

    def derivative(self, x):
        """
        Evaluates the derivative of the Hill function.

        Args:
            x (float or numpy array): Input value(s).

        Returns:
            float or numpy array: The derivative of the Hill function at x.
        """
        if isinstance(x, (float, int)):
            if x < 0:
                return 0.0
            else:
                return (self.U - self.L) * self.d * self.theta**self.d * x**(self.d - 1) / (self.theta**self.d + x**self.d)**2
        else:
            result = np.where(x < 0, 0.0, (self.U - self.L) * self.d * self.theta**self.d * x**(self.d - 1) / (self.theta**self.d + x**self.d)**2)
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
    h11 = HillFunction(L[0,0], U[0,0], T[0,0], d)
    h21 = HillFunction(L[1,0], U[1,0], T[1,0], d)
    h12 = HillFunction(U[0,1], L[0,1], T[0,1], d)
    h22 = HillFunction(L[1,1], U[1,1], T[1,1], d)
    
    def system(x):
        """Compute right-hand side of the ODE system."""
        return np.array([
            -x[0] + h11(x[0]) + h21(x[1]),
            -x[1] + h12(x[0]) * h22(x[1])
        ])
    
    def jacobian(x):
        """Compute the Jacobian matrix."""
        return np.array([
            [-1 + h11.derivative(x[0]), h21.derivative(x[1])],
            [h12.derivative(x[0]) * h22(x[1]), -1 + h12(x[0]) * h22.derivative(x[1])]
        ])
    
    return system, jacobian
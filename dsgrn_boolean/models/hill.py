import numpy as np
import jax.numpy as jnp

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

def HillSystem(L, U, T, d):
    """
    Create the system of equations and its Jacobian for the Hill function system.
    
    Args:
        L: Lower bounds matrix
        U: Upper bounds matrix
        T: Threshold matrix
        d: Hill function steepness parameter
        
    Returns:
        Tuple (system, jacobian) where:
            system: Function that computes the system of equations
            jacobian: Function that computes the Jacobian matrix
    """
    # Create Hill functions
    h11 = HillFunction(L[0,0], U[0,0], T[0,0], d)
    h21 = HillFunction(L[1,0], U[1,0], T[1,0], d)
    h12 = HillFunction(U[0,1], L[0,1], T[0,1], d)
    h22 = HillFunction(L[1,1], U[1,1], T[1,1], d)
    
    def system(x):
        x1, x2 = x
        
        # First equation: x1'
        dx1 = -x1 + h11(x1) + h21(x2)
        
        # Second equation: x2'
        dx2 = -x2 + h12(x1) * h22(x2)
        
        return np.array([dx1, dx2])
    
    def jacobian(x):
        x1, x2 = x
        
        # Compute derivatives
        dh11 = h11.derivative(x1)
        dh21 = h21.derivative(x2)
        dh12 = h12.derivative(x1)
        dh22 = h22.derivative(x2)
        
        # Jacobian matrix
        J = np.zeros((2, 2))
        
        # df1/dx1
        J[0,0] = -1 + dh11
        # df1/dx2
        J[0,1] = dh21
        
        # df2/dx1
        J[1,0] = dh12 * h22(x2)
        # df2/dx2
        J[1,1] = -1 + h12(x1) * dh22
        
        return J
    
    return system, jacobian
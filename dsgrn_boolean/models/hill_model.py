import numpy as np
from .hill import HillFunction

def hill(L, U, T, d):
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
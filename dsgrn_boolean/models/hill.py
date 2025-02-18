"""
Hill System Implementation for Numerical Analysis on DSGRN/Boolean project

This module provides a 2D Hill system model of the form:
    dx₁/dt = -x₁ + h₀₀(x₁) + h₁₀(x₂)
    dx₂/dt = -x₂ + h₀₁(x₁) * h₁₁(x₂)

Each Hill function is defined as:
    h(x) = A + (B - A)*x^d/(θ^d + x^d)
    where A is the value BEFORE the threshold, and B is the value AFTER the threshold.
    
Note:
  • h₀₁ is repressing, so it reverses the L and U order.
"""

import numpy as np
import warnings
from typing import Tuple, Callable
from numpy.typing import ArrayLike, NDArray

warnings.filterwarnings('ignore', category=RuntimeWarning)

class HillFunction:
    """A single Hill function h(x) = L + (U-L)xᵈ/(θᵈ + xᵈ)"""
    
    def __init__(self, L: float, U: float, theta: float, d: float) -> None:
        if theta <= 0 or d <= 0:
            raise ValueError("θ and d must be positive")
            
        self.L = float(L)
        self.U = float(U)
        self.theta = float(theta)
        self.d = float(d)

    def _compute_ratio(self, x: ArrayLike) -> NDArray:
        """Safely compute (x/θ)^d in log space"""
        ratio = np.zeros_like(x, dtype=float)
        positive = x > 0
        
        if np.any(positive):
            log_ratio = self.d * np.log(np.where(positive, x/self.theta, 1.0))
            ratio[positive] = np.exp(np.clip(log_ratio[positive], -700, 700))
        
        return ratio

    def __call__(self, x: ArrayLike) -> NDArray:
        """Evaluate h(x)"""
        x = np.asarray(x)
        ratio = self._compute_ratio(x)
        
        return self.L + np.where(np.isinf(ratio), 
                                self.U - self.L,
                                (self.U - self.L) * ratio / (1 + ratio))

    def derivative(self, x: ArrayLike) -> NDArray:
        """Evaluate h'(x)"""
        x = np.asarray(x)
        ratio = self._compute_ratio(x)
        
        result = np.zeros_like(x)
        valid = (x > 0) & ~np.isinf(ratio) & (ratio != 0)
        result[valid] = (self.U - self.L) * self.d * ratio[valid] / (x[valid] * (1 + ratio[valid])**2)
        return result

def hill(L: ArrayLike, U: ArrayLike, T: ArrayLike, d: float) -> Tuple[Callable, Callable]:
    """Create Hill system and its Jacobian for the 2D regulatory network"""
    L, U, T = map(np.asarray, (L, U, T))
    
    # Initialize Hill functions
    h00 = HillFunction(L[0,0], U[0,0], T[0,0], d)  # x0 → x0
    h10 = HillFunction(L[1,0], U[1,0], T[1,0], d)  # x1 → x0
    h01 = HillFunction(U[0,1], L[0,1], T[0,1], d)  # x0 → x1
    h11 = HillFunction(L[1,1], U[1,1], T[1,1], d)  # x1 → x1
    
    def system(x: ArrayLike) -> NDArray:
        """Right-hand side of dx/dt = f(x)"""
        return np.array([
            -x[0] + h00(x[0]) + h10(x[1]),      # dx0/dt
            -x[1] + h01(x[0]) * h11(x[1])       # dx1/dt
        ])
    
    def jacobian(x: ArrayLike) -> NDArray:
        """Jacobian matrix df/dx"""
        return -np.eye(2) + np.array([
            [h00.derivative(x[0]),               h10.derivative(x[1])],
            [h01.derivative(x[0])*h11(x[1]),     h01(x[0])*h11.derivative(x[1])]
        ])
    
    return system, jacobian
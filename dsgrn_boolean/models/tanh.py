import numpy as np

class TanhFunction:
    def __init__(self, L, U, theta, d):
        """
        Initializes a hyperbolic tangent (tanh) function.

        Args:
            L (float): The lower value.
            U (float): The upper value.
            theta (float): The threshold value (inflection point).
            d (float): The steepness coefficient.
        """
        self.L = L
        self.U = U
        self.theta = theta
        self.d = d

    def __call__(self, x):
        """
        Evaluates the tanh function.

        Args:
            x (float or numpy array): Input value(s).

        Returns:
            float or numpy array: The tanh function value(s) at x.
        """
        return self.L + (self.U - self.L) * 0.5 * (1 + np.tanh(self.d * (x - self.theta)))

    def derivative(self, x):
        """
        Evaluates the derivative of the tanh function.

        Args:
            x (float or numpy array): Input value(s).

        Returns:
            float or numpy array: The derivative of the tanh function at x.
        """
        # Derivative of tanh(x) is sech^2(x) = 1 - tanh^2(x)
        tanh_val = np.tanh(self.d * (x - self.theta))
        return (self.U - self.L) * 0.5 * self.d * (1 - tanh_val**2)
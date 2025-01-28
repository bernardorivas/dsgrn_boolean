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
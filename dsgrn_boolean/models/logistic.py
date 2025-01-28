import numpy as np

class LogisticFunction:
    def __init__(self, L, U, theta, d):
        """
        Initializes a logistic function.

        Args:
            L (float): The lower value.
            U (float): The upper value.
            theta (float): The threshold value.
            d (float): The logistic coefficient (controls the steepness).
        """
        self.L = L
        self.U = U
        self.theta = theta
        self.d = d

    def __call__(self, x):
        """
        Evaluates the logistic function.

        Args:
            x (float or numpy array): Input value(s).

        Returns:
            float or numpy array: The logistic function value(s) at x.
        """
        return self.L + (self.U - self.L) / (1 + np.exp(-self.d * (x - self.theta)))

    def derivative(self, x):
        """
        Evaluates the derivative of the logistic function.

        Args:
            x (float or numpy array): Input value(s).

        Returns:
            float or numpy array: The derivative of the logistic function at x.
        """
        exp_term = np.exp(-self.d * (x - self.theta))
        return (self.U - self.L) * self.d * exp_term / (1 + exp_term)**2
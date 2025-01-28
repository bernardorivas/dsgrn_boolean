import numpy as np

class StepFunction:
    def __init__(self, L, U, theta):
        """
        Initializes a step function.

        Args:
            L (float): The lower value of the step function.
            U (float): The upper value of the step function.
            theta (float): The threshold value.
        """
        self.L = L
        self.U = U
        self.theta = theta

    def __call__(self, x):
        """
        Evaluates the step function.

        Args:
            x (float or numpy array): Input value(s).

        Returns:
            float or numpy array: The step function value(s) at x.
        """
        if isinstance(x, (float, int)):
            return self.L if x < self.theta else self.U
        else:
            return np.where(x < self.theta, self.L, self.U)

    def derivative(self, x):
        """
        Evaluates the derivative of the step function.

        Args:
            x (float or numpy array): Input value(s).

        Returns:
            float or numpy array: The derivative of the step function at x.
        """
        if isinstance(x, (float, int)):
            if x == self.theta:
                return float('inf')  # Derivative is undefined (infinite) at the step
            else:
                return 0.0
        else:
            return np.where(x == self.theta, float('inf'), 0.0)
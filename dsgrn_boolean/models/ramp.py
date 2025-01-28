import numpy as np

class RampFunction:
    def __init__(self, L, U, theta, h):
        """
        Initializes a ramp function.

        Args:
            L (float): The lower value.
            U (float): The upper value.
            theta (float): The threshold value.
            h (float): Half the width of the [theta-h, theta+h] interval.
        """
        self.L = L
        self.U = U
        self.theta = theta
        self.h = h

    def __call__(self, x):
        """
        Evaluates the ramp function.

        Args:
            x (float or numpy array): Input value(s).

        Returns:
            float or numpy array: The ramp function value(s) at x.
        """
        if isinstance(x, (float, int)):
            if x < self.theta - self.h:
                return self.L
            elif x > self.theta + self.h:
                return self.U
            else:
                return self.L + (self.U - self.L) * (x - (self.theta - self.h)) / (2 * self.h)
        else:  # Handle numpy array input
            return np.where(x < self.theta - self.h, self.L,
                            np.where(x > self.theta + self.h, self.U,
                                     self.L + (self.U - self.L) * (x - (self.theta - self.h)) / (2 * self.h)))

    def derivative(self, x):
        """
        Evaluates the derivative of the ramp function.

        Args:
            x (float or numpy array): Input value(s).

        Returns:
            float or numpy array: The derivative of the ramp function at x.
        """
        if isinstance(x, (float, int)):
            if x < self.theta - self.h or x > self.theta + self.h:
                return 0.0
            else:
                return (self.U - self.L) / (2 * self.h)
        else:  # Handle numpy array input
            return np.where((x >= self.theta - self.h) & (x <= self.theta + self.h),
                            (self.U - self.L) / (2 * self.h), 0.0)
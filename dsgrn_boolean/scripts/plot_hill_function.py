import numpy as np
import matplotlib.pyplot as plt
from dsgrn_boolean.models.hill import HillFunction

def plot_hill_function(hill_function, T, n_points=100):
    """
    Plots a Hill function from 0 to 2*T.

    Args:
        hill_function (HillFunction): The Hill function to plot.
        T (float): The threshold value.
        n_points (int): Number of points to use for plotting.
    """
    x = np.linspace(0, 2 * T, n_points)
    y = hill_function(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("h(x)")
    plt.title("Hill Function Plot")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Example usage:
    L = 0.0
    U = 1.0
    T = 1.0
    d = 2.0
    hill_func = HillFunction(L, U, T, d)
    plot_hill_function(hill_func, T)
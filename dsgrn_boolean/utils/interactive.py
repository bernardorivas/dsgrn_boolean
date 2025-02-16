from ipywidgets import interact, FloatSlider
import ipywidgets as widgets
from .nullclines import plot_nullclines
import matplotlib.pyplot as plt

def plot_nullclines_interactive(L, U, T, d_max, n_points=20):
    """
    Create an interactive plot of nullclines with a slider to control the Hill function steepness.
    
    Args:
        L: Lower bounds matrix
        U: Upper bounds matrix
        T: Threshold matrix
        d_max: Maximum value for the Hill function exponent
    Returns:
        Interactive widget with slider controlling parameter d and displaying nullclines plot
    """
    
    @interact(d=FloatSlider(
        value=5.0,
        min=1.0,
        max=d_max,
        step=1,
        description='d:',
        continuous_update=False  # Only update when slider is released
    ))
    def update(d):
        plt.close('all')  # Close any existing figures
        return plot_nullclines(L, U, T, d, n_points)
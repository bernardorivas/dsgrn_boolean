from ipywidgets import interact, FloatSlider
import ipywidgets as widgets
from .nullclines import plot_nullclines

def plot_nullclines_interactive(L, U, T):
    """
    Create an interactive plot of nullclines with a slider to control the Hill function steepness.
    
    Args:
        L: Lower bounds matrix
        U: Upper bounds matrix
        T: Threshold matrix
    
    Returns:
        Interactive widget with slider controlling parameter d and displaying nullclines plot
    """
    
    @interact(d=FloatSlider(
        value=5.0,
        min=1.0,
        max=30.0,
        step=0.5,
        description='d:',
        continuous_update=False  # Only update when slider is released
    ))
    def update(d):
        zeros = plot_nullclines(L, U, T, d)
        return zeros
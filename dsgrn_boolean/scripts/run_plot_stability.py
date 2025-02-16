from dsgrn_boolean.utils.plot_stability import plot_multiple_parameters
import os

# Get the path to stability_data directory
stability_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stability_data')

# Plot all parameters in the stability_data directory
plot_multiple_parameters(stability_dir)
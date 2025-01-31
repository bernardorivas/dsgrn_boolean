import DSGRN
import os
import time
import webbrowser
from dsgrn_boolean.utils.sample_management import load_samples
from dsgrn_boolean.utils.simulation_debug import debug_sample

def main():
    # Setup network and parameter
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    par_index = 98
    parameter = parameter_graph.parameter(par_index)
    
    # Load first sample
    samples = load_samples(par_index)
    first_sample = samples[0]
    
    # Create figures directory if it doesn't exist
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(root_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Debug the sample and save figures
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    figs = debug_sample(network, first_sample, d=100)
    
    # Save and open each figure
    for i, fig in enumerate(figs):
        filename = f"debug_trajectory_{i+1}_{timestamp}.svg"
        filepath = os.path.join(figures_dir, filename)
        fig.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"\nFigure {i+1} saved as: {filename}")
        webbrowser.open(f'file://{filepath}')

if __name__ == "__main__":
    main() 
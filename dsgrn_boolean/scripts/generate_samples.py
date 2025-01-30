import DSGRN
import os
from tqdm import tqdm
import json

def generate_samples_file(net_spec, par_index, n_samples=10000):
    """
    Generate sample file for a specific parameter index.
    
    Args:
        net_spec: Network specification
        par_index: Parameter node index
        n_samples: Number of samples (default: 10000)
    """
    # Setup network
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    
    # Get parameter and setup sampler
    parameter = parameter_graph.parameter(par_index)
    sampler = DSGRN.ParameterSampler(network)
    
    # Get the project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate samples s
    filename = os.path.join(data_dir, f"parameter_samples_node_{par_index}.txt")
    print(f"\nGenerating {n_samples} samples for parameter node {par_index}...")
    
    samples = [sampler.sample(parameter) for _ in tqdm(range(n_samples))]
    
    with open(filename, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved samples to {filename}")

def generate_all_samples(net_spec):
    """Generate sample files for all parameter indices we're interested in."""
    # List of parameter indices we want to analyze
    par_indices = [0, 49, 98, 147]  # Add or modify indices as needed
    
    for par_index in par_indices:
        generate_samples_file(net_spec, par_index)

if __name__ == "__main__":
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    generate_all_samples(net_spec) 
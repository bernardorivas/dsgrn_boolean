import DSGRN
import os
from tqdm import tqdm
import json
import numpy as np
from dsgrn_boolean.utils.dsgrn_sample_to_matrix import extract_parameter_matrices

def generate_samples_file(net_spec, par_index, n_samples=10000, filter_tol=None):
    """
    Generate sample file for a specific parameter index with optional filtering.
    
    Args:
        net_spec: Network specification
        par_index: Parameter node index
        n_samples: Number of samples (default: 10000)
        filter_tol: Threshold tolerance for filtering (None for no filtering)
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
    
    if filter_tol is not None:
        # Generate filtered samples
        tol_str = str(filter_tol).replace('.', '_')
        filename = os.path.join(
            data_dir,
            f"filtered_parameter_samples_node_{par_index}_tol_{tol_str}.txt"
        )
        print(f"\nGenerating {n_samples} filtered samples with tolerance {filter_tol}...")
        
        filtered_samples = []
        pbar = tqdm(total=n_samples)
        
        while len(filtered_samples) < n_samples:
            batch_size = min(1000, n_samples - len(filtered_samples))
            new_samples = [sampler.sample(parameter) for _ in range(batch_size)]
            
            for sample in new_samples:
                L, U, T = extract_parameter_matrices(sample, network)
                diff1 = abs(T[0,0] - T[0,1])
                diff2 = abs(T[1,0] - T[1,1])
                if min(diff1, diff2) > filter_tol:
                    filtered_samples.append(sample)
                    pbar.update(1)
                    
                    if len(filtered_samples) >= n_samples:
                        break
        
        pbar.close()
        
        with open(filename, 'w') as f:
            for sample in filtered_samples:
                f.write(sample + '\n')
                
        print(f"Saved {len(filtered_samples)} filtered samples to {filename}")
        
    else:
        # Generate unfiltered samples
        filename = os.path.join(data_dir, f"parameter_samples_node_{par_index}.txt")
        print(f"\nGenerating {n_samples} samples for parameter node {par_index}...")
        
        samples = [sampler.sample(parameter) for _ in tqdm(range(n_samples))]
        
        with open(filename, 'w') as f:
            for sample in samples:
                f.write(sample + '\n')
        print(f"Saved samples to {filename}")

def generate_all_samples(net_spec, filter_tol=None):
    """Generate sample files for all parameter indices with optional filtering."""
    par_indices = [0, 49, 98, 147]
    
    for par_index in par_indices:
        generate_samples_file(net_spec, par_index, filter_tol=filter_tol)

if __name__ == "__main__":
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate parameter samples')
    parser.add_argument('--filter_tol', type=float, default=None,
                       help='Threshold tolerance for filtering (default: None for no filtering)')
    
    args = parser.parse_args()
    generate_all_samples(net_spec, filter_tol=args.filter_tol) 
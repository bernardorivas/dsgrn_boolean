import os
import argparse
from tqdm import tqdm
import DSGRN
from dsgrn_boolean.utils.dsgrn_sample_to_matrix import extract_parameter_matrices  # Added missing import

# Constants
DEFAULT_NUM_SAMPLES = 10000
DATA_DIR = 'data'
NET_SPEC = """x : x + y : E
              y : (~x) y : E"""

def get_data_dir():
    """Get the project data directory."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir_full = os.path.join(root_dir, DATA_DIR)
    os.makedirs(data_dir_full, exist_ok=True)
    return data_dir_full

def get_samples(net_spec: str, par_index: int, n_samples: int, filter_tol: float = None) -> list:
    """
    Collects n_samples valid parameter samples for a given parameter index.
    If filter_tol is provided, the sample's (L, U, T) matrices are extracted and
    the sample is only accepted if min(|T[i,j]-T[k,l]|) > filter_tol (for some indices).
    
    Args:
        net_spec: Network specification string.
        par_index: Parameter index to sample.
        n_samples: Desired number of samples.
        filter_tol: If provided, sample must satisfy threshold tolerance.
    
    Returns:
        A list of valid sample strings.
    """
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    parameter = parameter_graph.parameter(par_index)
    sampler = DSGRN.ParameterSampler(network)
    
    valid_samples = []
    pbar = tqdm(total=n_samples, desc="Collecting valid samples")
    
    while len(valid_samples) < n_samples:
        sample = sampler.sample(parameter)
        if filter_tol is not None:
            L, U, T = extract_parameter_matrices(sample, network)
            diff1 = abs(T[0,0] - T[0,1])
            diff2 = abs(T[1,0] - T[1,1])
            if min(diff1, diff2) > filter_tol:
                valid_samples.append(sample)
                pbar.update(1)
        else:
            valid_samples.append(sample)
            pbar.update(1)
    
    pbar.close()
    return valid_samples

# Optional: a helper to save the samples
def save_samples(samples: list, par_index: int, filter_tol: float = None) -> None:
    """
    Save the provided samples to a file.
    """
    data_dir = get_data_dir()
    if filter_tol is not None:
        tol_str = str(filter_tol).replace('.', '_')
        filename = os.path.join(data_dir, f"filtered_parameter_samples_node_{par_index}_tol_{tol_str}.txt")
    else:
        filename = os.path.join(data_dir, f"parameter_samples_node_{par_index}.txt")
    
    with open(filename, 'w') as f:
        for sample in samples:
            f.write(sample + '\n')
    print(f"Saved {len(samples)} samples to {filename}")

def load_samples(par_index, network=None, parameter=None, filter_tol=None):
    """
    Load pre-computed parameter samples from file.
    
    Args:
        par_index: Parameter node index
        network: DSGRN network object (only needed if regenerating)
        parameter: DSGRN parameter node (only needed if regenerating)
        force_regenerate: If True, regenerate samples even if file exists
        filtered: Whether to load filtered samples
        filter_tol: Threshold tolerance for filtered samples (default: 0.1)
    
    Returns:
        List of parameter samples as JSON strings
    """
    # Get the project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, DATA_DIR)
    
    # Determine filename based on filtering options
    if filtered:
        tol_str = str(filter_tol).replace('.', '_')
        filename = os.path.join(
            data_dir, 
            f"filtered_parameter_samples_node_{par_index}_tol_{tol_str}.txt"
        )
    else:
        filename = os.path.join(data_dir, f"parameter_samples_node_{par_index}.txt")
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            samples = [line.strip() for line in f]
    return samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect and save parameter samples.')
    parser.add_argument('--par_index', type=int, default=98, help='Parameter index')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to collect')
    parser.add_argument('--filter_tol', type=float, default=None, help='Threshold tolerance for filtering')
    
    args = parser.parse_args()
    samples = get_samples(NET_SPEC, args.par_index, args.n_samples, args.filter_tol)
    save_samples(samples, args.par_index, args.filter_tol)
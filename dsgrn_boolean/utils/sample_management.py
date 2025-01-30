import os
import json
from tqdm import tqdm
import DSGRN

def load_samples(par_index, network=None, parameter=None, force_regenerate=False):
    """
    Load pre-computed parameter samples from file.
    If file doesn't exist or force_regenerate is True, generate and save samples.
    
    Args:
        par_index: Parameter node index
        network: DSGRN network object (only needed if regenerating)
        parameter: DSGRN parameter node (only needed if regenerating)
        force_regenerate: If True, regenerate samples even if file exists
    
    Returns:
        List of parameter samples
    """
    # Get the project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')
    filename = os.path.join(data_dir, f"parameter_samples_node_{par_index}.txt")
    
    if os.path.exists(filename) and not force_regenerate:
        with open(filename, 'r') as f:
            samples = [json.loads(line.strip()) for line in f]
        print(f"Loaded {len(samples)} samples for parameter node {par_index}")
        return samples
    else:
        if network is None or parameter is None:
            raise ValueError("network and parameter must be provided when generating new samples")
        
        print(f"Generating 10000 samples for parameter node {par_index}...")
        sampler = DSGRN.ParameterSampler(network)
        samples = [sampler.sample(parameter) for _ in tqdm(range(10000))]
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Save samples
        with open(filename, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        print(f"Saved samples to {filename}")
        return samples 
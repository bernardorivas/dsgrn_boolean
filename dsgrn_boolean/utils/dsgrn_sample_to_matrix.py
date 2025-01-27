import json
import numpy as np

def extract_parameter_matrices(par_sample, network):
    """
    Extract L, U, and T matrices from a parameter sample.
    
    Args:
        par_sample: A JSON string containing parameter information
        network: A DSGRN network object
        
    Returns:
        A tuple (L, U, T) containing the parameter matrices:
            L: Lower bounds matrix
            U: Upper bounds matrix 
            T: Threshold matrix
    """
    # Initialize matrices
    D = network.size()
    L = np.zeros([D, D])
    U = np.zeros([D, D])
    T = np.zeros([D, D])

    # Parse parameter sample JSON
    sample_dict = json.loads(par_sample)

    # Extract parameter values
    for key, value in sample_dict['Parameter'].items():
        # Get parameter type (L, U, or T)
        par_type = key[0]
        
        # Extract node names from key
        node_names = [name.strip() for name in key[2:-1].split('->')]
        
        # Convert node names to indices
        node_indices = [network.index(node) for node in node_names]
        
        # Assign value to appropriate matrix
        if par_type == 'L':
            L[tuple(node_indices)] = value
        elif par_type == 'U':
            U[tuple(node_indices)] = value
        else:  # T
            T[tuple(node_indices)] = value

    return L, U, T 
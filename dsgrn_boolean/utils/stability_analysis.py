import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

def analyze_stability_data(par_index: int, d_values: Optional[List[int]] = None) -> Dict:
    """
    Analyze stability data for a given parameter index across different d values.
    
    Args:
        par_index: Parameter node index
        d_values: Optional list of d values to analyze. If None, analyzes all available d values.
        
    Returns:
        Dictionary containing analysis results:
        {
            'mean_success_rate': float,
            'std_success_rate': float,
            'max_success_rate': float,
            'min_success_rate': float,
            'best_d_values': List[int],  # d values with highest success rate
            'success_rates': Dict[str, float],  # success rate for each d value
            'sample_counts': Dict[str, int]  # number of samples for each d value
        }
    """
    # Get the project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stability_dir = os.path.join(root_dir, 'stability_data')
    
    # Find all stability data files for this parameter index
    stability_files = [f for f in os.listdir(stability_dir) 
                      if f.startswith(f'parindex_{par_index}_stability_data_')]
    
    if not stability_files:
        raise FileNotFoundError(f"No stability data found for parameter index {par_index}")
    
    # Use the most recent file
    latest_file = sorted(stability_files)[-1]
    file_path = os.path.join(stability_dir, latest_file)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    success_data = data['success_data']
    
    # Filter d values if specified
    if d_values is not None:
        success_data = {str(d): data for d, data in success_data.items() if int(d) in d_values}
    
    # Calculate statistics
    success_rates = {d: data['success_rate'] for d, data in success_data.items()}
    sample_counts = {d: data['num_samples'] for d, data in success_data.items()}
    
    rates = np.array(list(success_rates.values()))
    mean_rate = np.mean(rates)
    std_rate = np.std(rates)
    max_rate = np.max(rates)
    min_rate = np.min(rates)
    
    # Find d values with highest success rate
    best_rate = max_rate
    best_d_values = [int(d) for d, rate in success_rates.items() 
                    if abs(rate - best_rate) < 1e-10]
    
    return {
        'mean_success_rate': float(mean_rate),
        'std_success_rate': float(std_rate),
        'max_success_rate': float(max_rate),
        'min_success_rate': float(min_rate),
        'best_d_values': best_d_values,
        'success_rates': success_rates,
        'sample_counts': sample_counts
    } 
import os
import json
from typing import Dict, List, Optional

def analyze_failures_at_d(d_value: int) -> None:
    """
    Analyze and print which parameter indices failed at a specific d value.
    Reads from the JSON files saved by run_stability_analysis.py
    
    Args:
        d_value: The Hill coefficient value to analyze
    """
    # Get the project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stability_dir = os.path.join(root_dir, 'stability_data')
    
    # Find all stability data files
    stability_files = [f for f in os.listdir(stability_dir) 
                      if f.startswith('parindex_') and 'detailed_stability' in f]
    
    if not stability_files:
        print("No stability data files found.")
        return
    
    print(f"\nAnalysis for d = {d_value}:")
    print("-" * 50)
    
    # Analyze each parameter's data
    for filename in stability_files:
        with open(os.path.join(stability_dir, filename), 'r') as f:
            data = json.load(f)
        
        par_index = data['par_index']
        d_str = str(d_value)
        
        if d_str in data['by_d']:
            d_data = data['by_d'][d_str]
            success_rate = d_data['success_rate'] * 100
            
            print(f"\nParameter Index {par_index}:")
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Failed Samples: {d_data['num_failures']}/{d_data['num_samples']}")
            
            if d_data['num_failures'] > 0:
                print("\nFailed Sample Details:")
                for result in d_data['sample_results']:
                    if not result['success']:
                        print(f"  Sample {result['sample_index']}:")
                        print(f"    Found {result['num_stable_states']} stable states")
                        if result['stable_states']:
                            print("    Stable states found:")
                            for state in result['stable_states']:
                                print(f"      {state}")

if __name__ == "__main__":
    # Example usage
    analyze_failures_at_d(42) 
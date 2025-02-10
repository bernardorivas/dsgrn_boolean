import os
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

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

def plot_from_saved_data(par_indices: List[int] = None) -> None:
    """Create plots from saved JSON data files."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stability_dir = os.path.join(root_dir, 'stability_data')
    figures_dir = os.path.join(root_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Find relevant stability data files
    all_files = sorted([f for f in os.listdir(stability_dir) 
                       if f.startswith('parindex_') and 'detailed_stability' in f])
    
    if par_indices:
        files = [f for f in all_files if any(f'parindex_{p}_' in f for p in par_indices)]
    else:
        files = all_files
    
    if not files:
        print("No matching stability data files found.")
        return
    
    for filename in files:
        filepath = os.path.join(stability_dir, filename)
        print(f"\nProcessing file: {filename}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            par_index = data['par_index']
            d_values = sorted(int(d) for d in data['by_d'].keys())
            success_rates = []
            
            print(f"\nData for parameter {par_index}:")
            print(f"Number of d values: {len(d_values)}")
            print(f"d range: {min(d_values)} to {max(d_values)}")
            
            for d in d_values:
                d_data = data['by_d'][str(d)]
                # Count successes based on the 'success' field in sample_results
                num_success = sum(1 for result in d_data['sample_results'] if result['success'])
                success_rate = (num_success / d_data['num_samples']) * 100
                success_rates.append(success_rate)
                print(f"d={d}: success_rate={success_rate:.1f}% ({num_success}/{d_data['num_samples']} samples)")
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(d_values, success_rates, color='skyblue', width=0.8)
            ax.set_xlim(-1, max(d_values)+1)
            ax.set_ylim(0, 100)
            ax.set_xlabel('Hill Coefficient (d)')
            ax.set_ylabel('Coherency rate (%)')
            ax.set_title(f'Coherency rate by Hill coefficient at parameter node {par_index}')
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_filename = f"coherency_rate_parameter_{par_index}_from_saved.png"
            fig.savefig(os.path.join(figures_dir, plot_filename), bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Plot saved for parameter {par_index}")
            
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file {filename}:")
            print(f"Error details: {str(e)}")
        except Exception as e:
            print(f"Error processing file {filename}:")
            print(f"Error details: {str(e)}")

if __name__ == "__main__":
    # Example usage
    # analyze_failures_at_d(42)
    # Example usage: plot specific parameters
    plot_from_saved_data([0, 49, 98, 147]) 
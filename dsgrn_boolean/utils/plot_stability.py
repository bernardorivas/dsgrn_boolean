import json
import matplotlib.pyplot as plt
import os

def plot_stability_from_json(json_path: str, show_plot: bool = False) -> None:
    """
    Create a stability analysis plot from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing stability analysis data
        show_plot: Whether to display the plot (default: False)
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract necessary data
    par_index = data['par_index']
    d_values = sorted(int(d) for d in data['by_d'].keys())
    success_rates = []
    
    # Calculate success rates for each d value
    for d in d_values:
        d_data = data['by_d'][str(d)]
        num_success = sum(1 for result in d_data['sample_results'] if result['success'])
        success_rate = (num_success / d_data['num_samples']) * 100
        success_rates.append(success_rate)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(d_values, success_rates, color='skyblue')
    ax.axis([0, max(d_values), 0, 100])
    ax.set_xlabel('Hill Coefficient (d)')
    ax.set_ylabel('Coherency rate (%)')
    ax.set_title(f'Coherency rate by Hill coefficient at parameter node {par_index}')
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(json_path))), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Use JSON filename (without extension) for the figure
    json_filename = os.path.basename(json_path)
    filename = os.path.splitext(json_filename)[0] + '.png'
    filepath = os.path.join(figures_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def plot_multiple_parameters(json_dir: str, par_indices: list = None, show_plot: bool = False) -> None:
    """
    Create stability analysis plots for multiple parameter files.
    
    Args:
        json_dir: Directory containing JSON stability analysis files
        par_indices: List of parameter indices to plot (if None, plots all)
        show_plot: Whether to display the plots (default: False)
    """
    # Find all stability data files
    all_files = [f for f in os.listdir(json_dir) 
                 if f.startswith('parindex_') and f.endswith('.json')]
    
    # Filter files by parameter indices if specified
    if par_indices is not None:
        files = [f for f in all_files if any(f'parindex_{p}_' in f for p in par_indices)]
    else:
        files = all_files
    
    if not files:
        print("No matching stability data files found.")
        return
    
    # Process each file
    for filename in files:
        filepath = os.path.join(json_dir, filename)
        print(f"\nProcessing: {filename}")
        plot_stability_from_json(filepath, show_plot)

if __name__ == "__main__":
    # Example usage:
    stability_dir = "path/to/stability_data"
    
    # Plot a single file
    # plot_stability_from_json("path/to/stability_data/parindex_98_detailed_stability_20240315-123456.json")
    
    # Plot multiple parameters
    # plot_multiple_parameters(stability_dir, par_indices=[0, 49, 98, 147]) 
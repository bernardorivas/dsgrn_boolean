import os
import json
import matplotlib.pyplot as plt

def plot_coherency(json_path: str) -> None:
    """
    Create a coherency analysis from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing coherency analysis
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract necessary data
    par_index = data['par_index']
    d_values = sorted(int(d) for d in data['d_range'])
    success_rates = []
    
    # Calculate success rates for each d value
    for d in d_values:
        d_str = str(d)
        d_data = data['by_d'][d_str]
        num_success = sum(1 for result in d_data['sample_results'] if result['success'])
        success_rate = (num_success / data['num_samples']) * 100
        success_rates.append(success_rate)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(d_values, success_rates, color='skyblue')
    ax.axis([0, 100])
    ax.set_xlabel('Hill Coefficient (d)')
    ax.set_ylabel('Coherency rate (%)')
    ax.set_title(f'Coherency rate by Hill coefficient at parameter node {par_index}')
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Use JSON filename (without extension) for the figure
    json_filename = os.path.basename(json_path)
    filename = os.path.splitext(json_filename)[0] + '.png'
    filepath = os.path.join(figures_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {filepath}")
    plt.close(fig)

if __name__ == "__main__":
    # List files in the stability_data directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "stability_data")
    files = os.listdir(data_dir)
    
    # Select the JSON files in the directory
    json_files = [f for f in files if f.endswith('.json')]
    if not json_files:
        print("No JSON files found in the stability_data directory.")
    else:
        for json_file in json_files:
            filepath = os.path.join(data_dir, json_file)
            plot_stability_from_json(filepath)

import os
import json
import matplotlib.pyplot as plt

def plot_stability_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    
    par_index = data["par_index"]
    success_data = data["success_data"]
    
    # Create a plot where x-axis is Hill coefficient (d) and y-axis is matching success %
    d_values = sorted(success_data.keys())
    success_rates = [success_data[d]["success_rate"] * 100 for d in d_values]  # convert to percentages
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(d_values, success_rates, color='skyblue')
    
    ax.set_xlabel('Hill Coefficient (d)')
    ax.set_ylabel('Matching Success (%)')
    ax.set_title(f'Matching Success Rates vs Hill Coefficient (Parameter {par_index})')
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()

if __name__ == "__main__":
    # List files in the stability_data directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "stability_data")
    files = os.listdir(data_dir)
    
    # Select the first JSON file in the directory
    json_files = [f for f in files if f.endswith('.json')]
    if not json_files:
        print("No JSON files found in the stability_data directory.")
    else:
        example_file = os.path.join(data_dir, json_files[0])
        plot_stability_data(example_file)

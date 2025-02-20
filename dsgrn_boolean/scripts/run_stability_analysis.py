import DSGRN
import DSGRN_utils
import os
import time
import json
import argparse
from multiprocessing import cpu_count

from dsgrn_boolean.utils.find_equilibria import find_stable_equilibria_in_parallel
from dsgrn_boolean.utils.sample_management import load_samples

# Constants
DEFAULT_PARAMETER_INDEX = 0 # par_list = [0, 49, 98, 147]
DEFAULT_NUM_SAMPLES = 1000 # 100
DEFAULT_D_MIN = 1 
DEFAULT_D_MAX = 100 # 100
DEFAULT_D_STEP = 1
NETWORK_SPEC = """x : x + y : E
                  y : (~x) y : E"""
RESULTS_DIR = "stability_data"

def count_stable_states(parameter: DSGRN.Parameter) -> int:
    """Count the number of expected stable states using DSGRN_utils."""
    morse_graph, _, _ = DSGRN_utils.ConleyMorseGraph(parameter)
    n_stable = sum(1 for v in morse_graph.vertices() if not morse_graph.adjacencies(v))
    print(f"Number of stable states expected: {n_stable}")
    return n_stable

def save_results(par_index: int, network_spec: str, d_range: list, samples: list, results: dict, n_stable: int) -> None:
    """Creates detailed data dictionary and saves it to a JSON file."""
    detailed_data = {
        "par_index": par_index,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "network_spec": network_spec,
        "d_range": d_range,
        "num_samples": len(samples),
        "by_d": {}
    }

    for d in d_range:
        sample_list = results['by_d'][d]
        detailed_data["by_d"][str(d)] = {
            "sample_results": [
                {
                    "sample_index": i,
                    "success": len(stable_states) == n_stable
                }
                for i, stable_states in enumerate(sample_list)
            ]
        }

    root_dir = os.path.dirname(os.path.dirname(__file__))
    stability_dir = os.path.join(root_dir, RESULTS_DIR)
    os.makedirs(stability_dir, exist_ok=True)

    filename = f"parindex_{detailed_data['par_index']}_detailed_stability_{detailed_data['timestamp']}.json"
    filepath = os.path.join(stability_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(detailed_data, f, indent=2)

    print(f"\nDetailed results saved to: {filepath}")

def main(par_index: int = DEFAULT_PARAMETER_INDEX) -> None:
    """
    Main function to run stability analysis for a given parameter index.
    """
    # Configuration
    num_samples = DEFAULT_NUM_SAMPLES
    d_range = list(range(DEFAULT_D_MIN, DEFAULT_D_MAX + 1, DEFAULT_D_STEP))

    # Load samples
    samples = load_samples(par_index, filter_tol=0.1)[:num_samples]

    # Setup network and parameter using DSGRN
    network = DSGRN.Network(NETWORK_SPEC)
    parameter_graph = DSGRN.ParameterGraph(network)
    parameter = parameter_graph.parameter(par_index)

    # Count stable states
    n_stable = count_stable_states(parameter)

    # Run newton's method for each parameter sample and hill coefficient
    start_time = time.time()
    results = find_stable_equilibria_in_parallel(
        network,
        samples,
        n_stable,
        d_range=d_range,
        n_processes=min(cpu_count(), len(samples))
    )
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time/len(samples):.2f} seconds")

    # Save the results for a posteriori analysis
    save_results(par_index, NETWORK_SPEC, d_range, samples, results, n_stable)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stability analysis for a given parameter index.")
    parser.add_argument('-p', '--parameter', type=int, default=DEFAULT_PARAMETER_INDEX,
                        help=f'Parameter index (default: {DEFAULT_PARAMETER_INDEX})')
    args = parser.parse_args()
    main(args.parameter)
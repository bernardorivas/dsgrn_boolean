import DSGRN
import argparse
from dsgrn_boolean.utils.hill_analysis import analyze_hill_coefficients, create_single_plot
from dsgrn_boolean.utils.sample_management import load_samples

def analyze_and_plot(network, parameter, samples, d_range, par_index, filtered=False, filter_tol=0.1, plot=True, verbose=True):
    """Run analysis and optionally plot results"""
    # Run analysis
    results, summary, optimal_d, sample_results = analyze_hill_coefficients(
        network, parameter, samples, d_range, show_plot=plot
    )
    
    if plot:
        title = f'Parameter Node {par_index}' + (f' (Filtered, tol={filter_tol})' if filtered else '')
        create_single_plot(d_range, results, title)

    if verbose:
        print(f"\nSummary ({title if 'title' in locals() else 'Analysis'}):")
        print(f"Expected equilibria: {summary['expected_eq']}")
        print(f"Best match: {max(results):.1f}% at d = {d_range[np.argmax(results)]}")
        print(f"Worst match: {min(results):.1f}% at d = {d_range[np.argmin(results)]}")
    
    return results, summary, optimal_d, sample_results

def main():
    parser = argparse.ArgumentParser(description='Run Hill coefficient analysis')
    parser.add_argument('--filtered', action='store_true')
    parser.add_argument('--filter_tol', type=float, default=0.1)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--samples', type=int, default=30, 
                       help='Number of samples to analyze (30, 100, or 1000)')
    args = parser.parse_args()
    
    # Setup network and parameter
    net_spec = """x : x + y : E
                  y : (~x) y : E"""
    network = DSGRN.Network(net_spec)
    parameter_graph = DSGRN.ParameterGraph(network)
    par_index = 98
    parameter = parameter_graph.parameter(par_index)
    
    # Load and slice samples
    samples = load_samples(par_index, filtered=args.filtered, filter_tol=args.filter_tol)
    samples = samples[:args.samples]
    
    # Run analysis
    d_range = range(100, 101)
    results, summary, optimal_d, sample_results = analyze_and_plot(
        network, parameter, samples, d_range, par_index,
        filtered=args.filtered, filter_tol=args.filter_tol,
        plot=not args.no_plot
    )

if __name__ == "__main__":
    main() 
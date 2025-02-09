from dsgrn_boolean.scripts.run_stability_analysis import main
import matplotlib.pyplot as plt

def run_all():
    parameters = [0, 49, 98, 147]
    
    # Turn off interactive plotting
    plt.ioff()
    
    for p in parameters:
        print(f"\nRunning analysis for parameter {p}")
        print("=" * 50)
        main(p)
        plt.close('all')  # Close all figures to free memory

if __name__ == "__main__":
    run_all() 
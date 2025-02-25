import subprocess

par_list = [0, 49, 98, 147]
# par_list = [ 0, 98 ] # gotta run at 10000 samples
# par_list = [ 49, 147] # already ran at 10000 samples

for par_index in par_list:
    print(f"Running stability analysis for parameter index: {par_index}")
    subprocess.run(['python', '-m', 'dsgrn_boolean.scripts.run_stability_analysis', '-p', str(par_index)])

print("Finished running stability analysis for all parameter indices.")

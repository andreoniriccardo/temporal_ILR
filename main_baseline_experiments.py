import subprocess
import os
import yaml
import argparse
import pickle


settings = {"ME": False, # Mutually Exclusive setting
               "NME": True # Non-mutually Exclusive setting
               }
methods = {
    "ILR": True
    }


    
for setting in settings.keys():
    # Generate experiments
    if settings["ME"]:
        generate_experiment_ME_folder = os.path.join(os.getcwd(), "src", "utils")
        subprocess.run(["python", f"generate_experiments_baseline_ME.py"],
                            cwd=generate_experiment_ME_folder)
    if settings["NME"]:
        generate_experiment_NME_folder = os.path.join(os.getcwd(), "src", "utils")
        subprocess.run(["python", f"generate_experiments_baseline_NME.py"],
                            cwd=generate_experiment_NME_folder)
    
    # Folder paths
    folder_IRL = os.path.join(os.getcwd(), "_ILR")

    if (setting == 'ME') and (settings[setting]) :
        experiments_folder = f"experiments_baseline_{setting}"
        experiments = [name for name in os.listdir(experiments_folder) if (os.path.isdir(f"{experiments_folder}/{name}") and name.startswith("experiment_"))]
        
        for experiment in experiments:
            print("-----------------------------------------")
            print(f"Running experiment: {experiment}")
                        
            for method in methods.keys():
                if (method == "ILR") and (methods[method]):
                    print("Method: ILR")
                    subprocess.run(["python", f"run_experiment_baseline_ILR_{setting}.py",
                                    "--experiment", experiment],
                                    cwd=folder_IRL)
                
    
    if (setting == 'NME' and settings[setting]):
        experiments_folder = f"experiments_baseline_{setting}"
        experiments = [name for name in os.listdir(experiments_folder) if (os.path.isdir(f"{experiments_folder}/{name}") and name.startswith("experiment_"))]

        for experiment in experiments:
            print("-----------------------------------------")
            print(f"Running experiment: {experiment}")            

            for method in methods.keys():
                if (method == 'ILR') and (methods[method]):
                    print("Method: ILR")
                    subprocess.run(["python", f"run_experiment_baseline_ILR_{setting}.py",
                                    "--experiment", experiment],
                                    cwd=folder_IRL)
                


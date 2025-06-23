import random
import yaml
import os

from declare_formulas import formulas, formulas_names

# Set seed for reproducibility
random.seed(42)

# Configuration parameters
FORMULA_INDICES = list(range(len(formulas)))
n_symbols = 2
max_length = 4
num_experiments = 10
EXPERIMENT_FOLDER = "../../experiments_baseline_NME"
if not os.path.exists(f"{EXPERIMENT_FOLDER}"):
    os.makedirs(f"{EXPERIMENT_FOLDER}")

for formula_idx in FORMULA_INDICES:    
    for i in range(num_experiments):
        i_str = str(i) if len(str(i)) > 1 else "0"+str(i)
        config = {
        "experiment_id": f"experiment{i_str}_{formulas_names[formula_idx]}_{n_symbols}sym_{max_length}len",
        "formula1_name": (formulas_names[formula_idx]).replace("_a", "c0").replace("_b", "c1"),
        "formula2_name": '',
        "formula3_name": '',
        "formula1_index": formula_idx,
        "formula2_index": None,
        "formula3_index": None,
        "formula1": formulas[formula_idx].replace("a", "c0").replace("b", "c1"),
        "formula2": None,
        "formula3": None,
        "num_symbols": n_symbols,
        "max_length_traces": max_length,
        "train_size_traces": 0.5,
        "mutually_exclusive_symbols": False,
        "cnn_model": None,
        "hyperparameters": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 20
        },
        "seed": 20+i
        }
        experiment_id = config["experiment_id"]
        formula_idx_str = str(formula_idx) if len(str(formula_idx)) > 1 else "0"+str(formula_idx)
        folder_name = f"{EXPERIMENT_FOLDER}/experiment_f{formula_idx_str}_{i_str}_nonmutex"
        filename = f"config.yaml"
        os.makedirs(folder_name, exist_ok=True)
        filepath = os.path.join(folder_name, filename)
        
        # Write YAML file
        with open(filepath, 'w') as f:
            yaml.dump(config, f, sort_keys=False)   

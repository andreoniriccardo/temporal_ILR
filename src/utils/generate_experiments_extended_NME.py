import random
import yaml
import os

from declare_formulas import formulas, formulas_names

# Set seed for reproducibility
random.seed(42)

# Configuration parameters
N_SYMBOLS_OPTIONS = [2, 3, 4]
MAX_LENGTHS = [5, 10, 20]
NUM_FORMULAS_PER_SYMBOL = 5
FORMULA_INDICES = list(range(len(formulas)))
EXPERIMENT_FOLDER = "../../experiments_extended_NME"
if not os.path.exists(f"{EXPERIMENT_FOLDER}"):
    os.makedirs(f"{EXPERIMENT_FOLDER}")

def generate_formula_combinations(n_symbols, num_combinations):
    combinations = []
    for _ in range(num_combinations):
        if n_symbols == 2:
            indices = [random.choice(FORMULA_INDICES)]
        elif n_symbols == 3:
            indices = [random.choice(FORMULA_INDICES) for _ in range(2)]
        elif n_symbols == 4:
            indices = [random.choice(FORMULA_INDICES) for _ in range(3)]
        combinations.append(indices)
    return combinations

def create_config(n_symbols, formula_indices, max_length):
    formula_indices_new = formula_indices.copy()
    replacements = str.maketrans({"(": "", ")": "", "_": "", ",": "", " ":"-"})
    formula_orig_names = [formulas_names[i] for i in formula_indices_new]
    formula_names = [formulas_names[i].lower().translate(replacements) for i in formula_indices_new]
    logical_formulas = [formulas[i] for i in formula_indices_new]
    
    if n_symbols == 2:
        formula_names += ["", ""]
        logical_formulas += ["", ""]
        formula_indices_new += [None, None]
    elif n_symbols == 3:
        formula_names += [""]
        logical_formulas += [""]
        formula_indices_new += [None]
    
    # Create experiment ID
    active_formulas = formula_names[:n_symbols-1] if n_symbols > 2 else formula_names[:1]
    formula_part = "AND".join(active_formulas)

    config = {
        "experiment_id": f"experiment_{formula_part}_{n_symbols}sym_{max_length}len",
        "formula1_name": (formula_orig_names[0]).replace("_a", "c0").replace("_b", "c1"),
        "formula2_name": (formula_orig_names[1] if len(formula_orig_names) > 1 else "").replace("_a", "c1").replace("_b", "c2"),
        "formula3_name": (formula_orig_names[2] if len(formula_orig_names) > 2 else "").replace("_a", "c2").replace("_b", "c3"),
        "formula1_index": formula_indices_new[0],
        "formula2_index": formula_indices_new[1] if len(formula_indices_new) > 1 else None,
        "formula3_index": formula_indices_new[2] if len(formula_indices_new) > 2 else None,
        "formula1": logical_formulas[0].replace("a", "c0").replace("b", "c1"),
        "formula2": logical_formulas[1].replace("a", "c1").replace("b", "c2") if len(formula_indices_new) > 1 else None,
        "formula3": logical_formulas[2].replace("a", "c2").replace("b", "c3") if len(formula_indices_new) > 2 else None,
        "num_symbols": n_symbols,
        "max_length_traces": max_length,
        "train_size_traces": 0.5,
        "mutually_exclusive_symbols": False,
        "cnn_model": f"untrained_CNN_NME_state_dict_{n_symbols}sym.pth",
        "hyperparameters": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "num_epochs": 20
        },
        "seed": 21
    }
    return config


i = 0
for n_symbols in N_SYMBOLS_OPTIONS:
    formula_combinations = generate_formula_combinations(n_symbols, NUM_FORMULAS_PER_SYMBOL)
    
    for formula_indices in formula_combinations:
        for max_length in MAX_LENGTHS:
            config = create_config(n_symbols, formula_indices, max_length)
            experiment_id = config["experiment_id"]
            str_i = str(i) if len(str(i)) > 1 else "0"+str(i)
            folder_name = f"{EXPERIMENT_FOLDER}/experiment_"+str_i+"_"+str(config["num_symbols"])+"_sym"+"_"+str(config["max_length_traces"])+"_len"
            filename = f"config.yaml"
            os.makedirs(folder_name, exist_ok=True)
            filepath = os.path.join(folder_name, filename)
            
            # Write YAML file
            with open(filepath, 'w') as f:
                yaml.dump(config, f, sort_keys=False)
            
            i += 1

print(f"Generated {len(N_SYMBOLS_OPTIONS)*NUM_FORMULAS_PER_SYMBOL*len(MAX_LENGTHS)} configuration files.")
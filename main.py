import subprocess
import os
import yaml
import argparse
import pickle
from src.utils.create_dataset import generate_sample_traces_sym_ME, generate_sample_traces_sym_NME, create_complete_traces_sym_NME
from src.utils.logic.parser import LTLfParser as LTLfParserPL


settings = {"ME": True, # Mutually Exclusive setting
               "NME": True # Non-mutually Exclusive setting
               }
methods = {
    "ILR": True
    }






    
for setting in settings.keys():
    # Generate experiments
    if settings["ME"]:
        generate_experiment_ME_folder = os.path.join(os.getcwd(), "src", "utils")
        subprocess.run(["python", f"generate_experiments_ME.py"],
                            cwd=generate_experiment_ME_folder)
    if settings["NME"]:
        generate_experiment_NME_folder = os.path.join(os.getcwd(), "src", "utils")
        subprocess.run(["python", f"generate_experiments_NME.py"],
                            cwd=generate_experiment_NME_folder)
    
    # Folder paths
    folder_IRL = os.path.join(os.getcwd(), "_ILR")

    if (setting == 'ME') and (settings[setting]) :
        NUM_SAMPLES = 1000
        NUM_PASSES_IMG = 5
        experiments_folder = f"experiments_{setting}"
        experiments = [name for name in os.listdir(experiments_folder) if (os.path.isdir(f"{experiments_folder}/{name}") and name.startswith("experiment_"))]
        
        for experiment in experiments:
            print("-----------------------------------------")
            print(f"Running experiment: {experiment}")
            with open(f"experiments_{setting}/{experiment}/config.yaml", "r") as f:
                config = yaml.safe_load(f)
                config = argparse.Namespace(**config)
            # Create symbolic dataset
            alphabet = ["c" + str(i) for i in range(config.num_symbols)]
            # Retrieve formula
            if config.formula3 != '':
                formula = f"({config.formula1}) & ({config.formula2}) & ({config.formula3})"
            elif config.formula2 != '':
                formula = f"({config.formula1}) & ({config.formula2})"
            else:
                formula = config.formula1
            # Formula PL
            parserPL = LTLfParserPL(config.max_length_traces, alphabet)
            f = parserPL(formula)
            f_pl = f.to_propositional(parserPL.predicates, config.max_length_traces, 0)
            print('Generating symbolic dataset...')
            if not os.path.exists(f"experiments_{setting}/{experiment}/dataset/symbolic_dataset.pickle"):
                train_traces, test_traces, train_acceptance_tr, test_acceptance_tr = generate_sample_traces_sym_ME(config.max_length_traces, 
                                                        alphabet, 
                                                        f_pl, 
                                                        NUM_SAMPLES, 
                                                        config.train_size_traces,
                                                        config.seed
                                                        )
                symbolic_dataset = (train_traces, test_traces, train_acceptance_tr, test_acceptance_tr)
                # Export dataset
                # Crate dataset folder
                if not os.path.exists(f"experiments_{setting}/{experiment}/dataset"):
                    os.makedirs(f"experiments_{setting}/{experiment}/dataset")
                with open(f"experiments_{setting}/{experiment}/dataset/symbolic_dataset.pickle", "wb") as f:
                    pickle.dump(symbolic_dataset, f)
                print('Symbolic dataset generated and exported.')
            else:
                print('Symbolic dataset already exists.')

            
            for method in methods.keys():
                if (method == "ILR") and (methods[method]):
                    print("Method: ILR")
                    subprocess.run(["python", f"run_experiment_ILR_{setting}.py",
                                    "--experiment", experiment,
                                    "--num_samples", str(NUM_SAMPLES),
                                    "--num_passes_img", str(NUM_PASSES_IMG)],
                                    cwd=folder_IRL)
                
    
    if (setting == 'NME' and settings[setting]):
        NUM_PASSES_IMG = 1
        NUMBER_TRAIN_SEQUENCES = 2500
        PCT_LEN4 = 0.2
        experiments_folder = f"experiments_{setting}"
        experiments = [name for name in os.listdir(experiments_folder) if (os.path.isdir(f"{experiments_folder}/{name}") and name.startswith("experiment_"))]

        for experiment in experiments:
            print("-----------------------------------------")
            print(f"Running experiment: {experiment}")
            with open(f"experiments_{setting}/{experiment}/config.yaml", "r") as f:
                config = yaml.safe_load(f)
                config = argparse.Namespace(**config)
            # Create symbolic dataset
            alphabet = ["c" + str(i) for i in range(config.num_symbols)]
            # Retrieve formula
            if config.formula3 != '':
                formula = f"({config.formula1}) & ({config.formula2}) & ({config.formula3})"
            elif config.formula2 != '':
                formula = f"({config.formula1}) & ({config.formula2})"
            else:
                formula = config.formula1
            # Formula PL
            parserPL = LTLfParserPL(config.max_length_traces, alphabet)
            f = parserPL(formula)
            f_pl = f.to_propositional(parserPL.predicates, config.max_length_traces, 0)
            print('Generating symbolic dataset...')
            if not os.path.exists(f"experiments_{setting}/{experiment}/dataset/symbolic_dataset.pickle"):
                traces_t_train_complete, traces_t_test_complete, accepted_train_complete, accepted_test_complete = create_complete_traces_sym_NME(
                                                    max_length_generated=4,
                                                    max_length_formula=config.max_length_traces,
                                                    alphabet=alphabet, 
                                                    formula=f_pl, 
                                                    seed=config.seed,
                                                    train_size=config.train_size_traces, 
                                                    max_concurrent_symbols=2)
                traces_t_train_complete = traces_t_train_complete[:int(PCT_LEN4*NUMBER_TRAIN_SEQUENCES)]
                traces_t_test_complete = traces_t_test_complete[:int(PCT_LEN4*NUMBER_TRAIN_SEQUENCES)]
                accepted_train_complete = accepted_train_complete[:int(PCT_LEN4*NUMBER_TRAIN_SEQUENCES)]
                accepted_test_complete = accepted_test_complete[:int(PCT_LEN4*NUMBER_TRAIN_SEQUENCES)]

                NUM_SAMPLES = 2*NUMBER_TRAIN_SEQUENCES - 2*len(traces_t_train_complete)
                traces_t_train_long, traces_t_test_long, accepted_train_long, accepted_test_long = generate_sample_traces_sym_NME(
                                                            min_length_traces=2,
                                                            max_length_traces=config.max_length_traces,
                                                            alphabet=alphabet,
                                                            formula=f_pl,
                                                            num_samples=NUM_SAMPLES,
                                                            train_size=config.train_size_traces, 
                                                            seed=config.seed)
                
                train_traces = list(traces_t_train_complete) + list(traces_t_train_long)
                test_traces = list(traces_t_test_complete) + list(traces_t_test_long)
                train_acceptance_tr = list(accepted_train_complete) + list(accepted_train_long)
                test_acceptance_tr = list(accepted_test_complete) + list(accepted_test_long)
                symbolic_dataset = (train_traces, test_traces, train_acceptance_tr, test_acceptance_tr)
                # Export dataset
                # Crate dataset folder
                if not os.path.exists(f"experiments_{setting}/{experiment}/dataset"):
                    os.makedirs(f"experiments_{setting}/{experiment}/dataset")
                with open(f"experiments_{setting}/{experiment}/dataset/symbolic_dataset.pickle", "wb") as f:
                    pickle.dump(symbolic_dataset, f)
                print('Symbolic dataset generated and exported.')
            else:
                print('Symbolic dataset already exists.')

            for method in methods.keys():
                if (method == 'ILR') and (methods[method]):
                    print("Method: ILR")
                    subprocess.run(["python", f"run_experiment_ILR_{setting}.py",
                                    "--experiment", experiment,
                                    # "--num_samples", str(NUM_SAMPLES),
                                    "--num_passes_img", str(NUM_PASSES_IMG)],
                                    cwd=folder_IRL)
                


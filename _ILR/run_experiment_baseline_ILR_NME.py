# import absl.flags
import torchvision
import torch
import sys
import os
import yaml
import argparse
import json
import pickle
import random
import numpy as np
from flloat.parser.ltlf import LTLfParser as LTLfParserDFA


from logical_network import LogicalNetwork

# Get absolute path to src directory
current_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')

# Add to Python path
sys.path.insert(0, src_path)


from configs.global_config import DATA_FOLDER, MODELS_FOLDER
from utils.logic.parser import LTLfParser as LTLfParserPL
from utils.logic.formula import Predicate, IMPLIES
from utils.save_results import save_results 
from utils.save_model import save_model
from utils.create_dataset import create_complete_set_traces, create_image_sequence_dataset_non_mut_ex
from utils.classifier import CNN_NME

EXPERIMENTS_FOLDER = '../experiments_baseline_NME'
# -----------------------------------------------------------------------------

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--experiment_name", type=str, required=True)
args = arg_parser.parse_args()

experiment_name = args.experiment_name


with open(f"{EXPERIMENTS_FOLDER}/{experiment_name}/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    config = argparse.Namespace(**config)

print(config)

# Create results folder
if not os.path.exists(f"{EXPERIMENTS_FOLDER}/{experiment_name}/results"):
    os.makedirs(f"{EXPERIMENTS_FOLDER}/{experiment_name}/results")
# Create checkpoints folder
if not os.path.exists(f"{EXPERIMENTS_FOLDER}/{experiment_name}/checkpoints"):
    os.makedirs(f"{EXPERIMENTS_FOLDER}/{experiment_name}/checkpoints")
# Crate dataset folder
if not os.path.exists(f"{EXPERIMENTS_FOLDER}/{experiment_name}/dataset"):
    os.makedirs(f"{EXPERIMENTS_FOLDER}/{experiment_name}/dataset")

# Data
normalize = torchvision.transforms.Normalize(mean=(0.1307,),
                                                     std=(0.3081,))

transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize,
    ])

train_data = torchvision.datasets.MNIST(DATA_FOLDER, train=True, download=True, transform=transforms)
test_data = torchvision.datasets.MNIST(DATA_FOLDER, train=False, download=True, transform=transforms)

# Formula
if config.formula3 != None:
    formula = f"({config.formula1}) & ({config.formula2}) & ({config.formula3})"
elif config.formula2 != None:
    formula = f"({config.formula1}) & ({config.formula2})"
else:
    formula = config.formula1
print("Formula: ", formula)

# Seed
torch.manual_seed(config.seed)
torch.random.manual_seed(config.seed)
np.random.seed(config.seed)
torch.cuda.manual_seed(config.seed)
random.seed(config.seed)


# Dataset creation
parser = LTLfParserDFA()
ltl_formula_parsed = parser(formula)
dfa = ltl_formula_parsed.to_automaton() # DFA used only for dataset creation in the baseline experiments
alphabet = ["c" + str(i) for i in range(config.num_symbols)]

_, _, train_traces, test_traces, train_acceptance_tr, test_acceptance_tr = create_complete_set_traces(
                config.max_length_traces, alphabet, dfa, train_with_accepted_only=False,
                train_size=config.train_size_traces)
symbolic_dataset = (train_traces, test_traces, train_acceptance_tr, test_acceptance_tr)

# Formula PL
parserPL = LTLfParserPL(config.max_length_traces, alphabet)
f = parserPL(formula)
f_pl = f.to_propositional(parserPL.predicates, config.max_length_traces, 0)
y1 = Predicate('y1', config.max_length_traces*config.num_symbols)
k = IMPLIES([f_pl, y1])

train_img_seq, train_acceptance_img = create_image_sequence_dataset_non_mut_ex(train_data, config.num_symbols, train_traces,
                                                                                          train_acceptance_tr,print_size=True)
test_img_seq_hard, test_acceptance_img_hard = create_image_sequence_dataset_non_mut_ex(test_data, config.num_symbols,
                                                                                        test_traces,
                                                                                        test_acceptance_tr,print_size=True)
image_seq_dataset = (train_img_seq, train_acceptance_img, 
                     test_img_seq_hard, test_acceptance_img_hard)


# Model
# CNN hyperparameters
num_classes = config.num_symbols
num_channels = 1
nodes_linear = 54

data = image_seq_dataset
classifier = CNN_NME(num_channels, num_classes, nodes_linear, mutually_exc=config.mutually_exclusive_symbols, complete_initialization=False)
logical_nn = LogicalNetwork(classifier, 
                            k,
                            data=data, 
                            symbolic_dataset=symbolic_dataset, 
                            lr=config.hyperparameters["learning_rate"],
                            batch_size=config.hyperparameters["batch_size"],
                            seq_max_len=config.max_length_traces,
                            mutex=config.mutually_exclusive_symbols,
                            max_iterations_lrl=1,
                            baseline=True)


loss_list, train_image_classification_accuracy_list, test_image_classification_accuracy_list, time_list = logical_nn.train_classifier(config.hyperparameters["num_epochs"])

# Save results
save_results(loss_list,
              train_image_classification_accuracy_list,
                test_image_classification_accuracy_list, 
                time_list,
                f"{EXPERIMENTS_FOLDER}/{experiment_name}/results/metrics_ILR.json")


# Save model
save_model(logical_nn.nn_layer, f"{EXPERIMENTS_FOLDER}/{experiment_name}/checkpoints/model_ILR.pth")
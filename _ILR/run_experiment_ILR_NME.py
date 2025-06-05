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


from logical_network import LogicalNetwork

# Get absolute path to src directory
current_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')

# Add to Python path
sys.path.insert(0, src_path)


from configs.global_config import DATA_FOLDER, MODELS_FOLDER
from utils.logic.parser import LTLfParser as LTLfParserPL
from utils.save_results import save_results 
from utils.save_model import save_model
from utils.image_normalization import normalize_image_sequences_pytorch
from utils.create_dataset import create_image_sequence_dataset_sampling_NME
from utils.classifier import CNN_NME

EXPERIMENTS_FOLDER = '../experiments_NME'
# -----------------------------------------------------------------------------

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--experiment_name", type=str, required=True)
arg_parser.add_argument("--num_passes_img", type=int, required=True)
args = arg_parser.parse_args()

experiment_name = args.experiment_name
NUM_PASSES_IMG = args.num_passes_img


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
if config.formula3 != '':
    formula = f"({config.formula1}) & ({config.formula2}) & ({config.formula3})"
elif config.formula2 != '':
    formula = f"({config.formula1}) & ({config.formula2})"
else:
    formula = config.formula1
print("Formula: ", formula)


# Dataset creation
alphabet = ["c" + str(i) for i in range(config.num_symbols)]

# Loading symbolic dataset
with open(f"{EXPERIMENTS_FOLDER}/{experiment_name}/dataset/symbolic_dataset.pickle", "rb") as f:
    symbolic_dataset = pickle.load(f)
train_traces, test_traces, train_acceptance_tr, test_acceptance_tr = symbolic_dataset


# Seed
# torch.manual_seed(config.seed)
# torch.random.manual_seed(config.seed)
# np.random.seed(config.seed)
# torch.cuda.manual_seed(config.seed)
# random.seed(config.seed)

# Formula PL
parserPL = LTLfParserPL(config.max_length_traces, alphabet)
f = parserPL(formula)
f_pl = f.to_propositional(parserPL.predicates, config.max_length_traces, 0)

train_img_seq, train_symbolic_sequences, train_acceptance_img = create_image_sequence_dataset_sampling_NME(train_data, 
                                                                             len(alphabet),
                                                                             train_traces,
                                                                             train_acceptance_tr,
                                                                             num_passes=NUM_PASSES_IMG,
                                                                             seed=config.seed,
                                                                             shuffle=False)
test_img_seq, test_symbolic_sequences, test_acceptance_img = create_image_sequence_dataset_sampling_NME(test_data, 
                                                                             len(alphabet),
                                                                             test_traces,
                                                                             test_acceptance_tr,
                                                                             num_passes=NUM_PASSES_IMG,
                                                                             seed=config.seed,
                                                                             shuffle=False)

train_img_seq = normalize_image_sequences_pytorch(train_img_seq)
test_img_seq = normalize_image_sequences_pytorch(test_img_seq)
image_seq_dataset = (train_img_seq, train_symbolic_sequences, train_acceptance_img, 
                     test_img_seq, test_symbolic_sequences, test_acceptance_img)

unique_train_traces = set()
for t in train_traces:
    hashable = (tuple(t.shape), t.numpy().tobytes())
    unique_train_traces.add(hashable)
unique_test_traces = set()
for t in test_traces:
    hashable = (tuple(t.shape), t.numpy().tobytes())
    unique_test_traces.add(hashable)

dataset_stats = {
    "train": {
        "sym_traces_tot": len(train_traces),
        "sym_traces_unique": len(unique_train_traces), 
        "img_seq_tot": len(train_img_seq), 
        "img_seq_accepting": sum(train_acceptance_img),
        "img_seq_accepting_ratio": sum(train_acceptance_img) / len(train_img_seq) 
    },
    "test": {
        "sym_traces_tot": len(test_traces),
        "sym_traces_unique": len(unique_test_traces),
        "img_seq_tot": len(test_img_seq), 
        "img_seq_accepting": sum(test_acceptance_img), 
        "img_seq_accepting_ratio": sum(test_acceptance_img) / len(test_img_seq) 
    }
}

with open(f"{EXPERIMENTS_FOLDER}/{experiment_name}/dataset/dataset_stats_ILR.json", "w") as f:
    json.dump(dataset_stats, f, indent=4)

# Model
# CNN hyperparameters
num_classes = config.num_symbols
num_channels = 1
nodes_linear = 54

formula = f_pl
data = image_seq_dataset
classifier = CNN_NME(num_channels, num_classes, nodes_linear, mutually_exc=config.mutually_exclusive_symbols)
classifier.load_state_dict(torch.load(f"{MODELS_FOLDER}/{config.cnn_model}", weights_only=True))
logical_nn = LogicalNetwork(classifier, 
                            formula, 
                            data=data, 
                            symbolic_dataset=symbolic_dataset, 
                            lr=config.hyperparameters["learning_rate"],
                            batch_size=config.hyperparameters["batch_size"],
                            seq_max_len=config.max_length_traces,
                            mutex=config.mutually_exclusive_symbols,
                            schedule=0.1,
                            max_iterations_lrl=1)


loss_list, train_image_classification_accuracy_list, test_image_classification_accuracy_list, time_list = logical_nn.train_classifier(config.hyperparameters["num_epochs"])

# Save results
save_results(loss_list,
              train_image_classification_accuracy_list,
                test_image_classification_accuracy_list, 
                time_list,
                f"{EXPERIMENTS_FOLDER}/{experiment_name}/results/metrics_ILR.json")


# Save model
save_model(logical_nn.nn_layer, f"{EXPERIMENTS_FOLDER}/{experiment_name}/checkpoints/model_ILR.pth")
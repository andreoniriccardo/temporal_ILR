import torchvision
from flloat.parser.ltlf import LTLfParser
import sys
import os
from LTL_grounding import LTL_grounding
import json
import pickle
import torch
import random
import numpy as np

import yaml
import argparse
from threading import Thread
import time

# Get absolute path to src directory
current_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')
utils_path = os.path.join(src_path, 'utils')

# Add to Python path
sys.path.insert(0, src_path)
sys.path.insert(0, utils_path)

from configs.global_config import DATA_FOLDER, MODELS_FOLDER
from save_results import plot_metrics, save_results
from save_model import save_model
from image_normalization import normalize_image_sequences_pytorch
from create_dataset import create_image_sequence_dataset_sampling_NME
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

# Seed
# torch.manual_seed(config.seed)
# torch.random.manual_seed(config.seed)
# np.random.seed(config.seed)
# torch.cuda.manual_seed(config.seed)
# random.seed(config.seed)

# Dataset creation
from threading import Thread, Event

# Timeout for possible long execution time of the DFA generation
class TimeoutBlock:
    def __init__(self, timeout_seconds, timeout_file):
        self.timeout_seconds = timeout_seconds
        self.timeout_file = timeout_file
        self.completed = Event()
        self.timed_out = False

    def __enter__(self):
        self.start_time = time.time()
        # Start timeout monitoring thread
        self.monitor_thread = Thread(target=self._monitor_timeout)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.completed.set()
        return False

    def _monitor_timeout(self):
        while not self.completed.is_set():
            if time.time() - self.start_time > self.timeout_seconds:
                self.timed_out = True
                with open(self.timeout_file, "w") as f:
                    f.write(f"Script terminated due to timeout after {time.time() - self.start_time} seconds")
                print("\nTimeout occurred in critical section - exiting")
                os._exit(1)
            time.sleep(1)

with TimeoutBlock(60*60, f"{EXPERIMENTS_FOLDER}/{experiment_name}/results/DFA_timeout_log.txt"): # 1h timeout
    # Time sensitive code
    parser = LTLfParser()
    ltl_formula_parsed = parser(formula)
    dfa = ltl_formula_parsed.to_automaton()

print("No timeout occurred")

alphabet = ["c" + str(i) for i in range(config.num_symbols)]

# Loading symbolic dataset
with open(f"{EXPERIMENTS_FOLDER}/{experiment_name}/dataset/symbolic_dataset.pickle", "rb") as f:
    symbolic_dataset = pickle.load(f)
train_traces, test_traces, train_acceptance_tr, test_acceptance_tr = symbolic_dataset

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
    },
}

with open(f"{EXPERIMENTS_FOLDER}/{experiment_name}/dataset/dataset_stats_DFA.json", "w") as f:
    json.dump(dataset_stats, f, indent=4)

# Model
ltl_ground =LTL_grounding(formula,dfa, config.mutually_exclusive_symbols, symbolic_dataset, image_seq_dataset, config.num_symbols, 100, config.max_length_traces, train_with_accepted_only=False, num_exp=0, log_dir="",
                          cnn_initialization=f"{MODELS_FOLDER}/{config.cnn_model}",
                          lr=config.hyperparameters["learning_rate"], 
                          batch_size=config.hyperparameters["batch_size"])


loss_list, train_image_classification_accuracy_list, test_image_classification_accuracy_list, time_list = ltl_ground.train_classifier(config.hyperparameters["num_epochs"])

# Save results
save_results(loss_list,
              train_image_classification_accuracy_list,
                test_image_classification_accuracy_list, 
                time_list,
                f"{EXPERIMENTS_FOLDER}/{experiment_name}/results/metrics_DFA.json")

# Save model
save_model(ltl_ground.classifier, f"{EXPERIMENTS_FOLDER}/{experiment_name}/checkpoints/model_DFA.pth")
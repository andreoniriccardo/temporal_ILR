# Temporal Iterative Local Refinement (T-ILR)

## Setup
1. Install dependencies:
It is recommended to use a virtual environment.
``` 
pip install -r requirements.txt
``` 

2. Data:
The MNIST dataset will be downloaded automatically to the `data/` folder upon the first run.

## Running Experiments
This repository supports two main sets of experiments: the baseline experiments and the extended scalability experiments introduced in our paper.

### 1. Baseline Experiments
These experiments replicate the setup from the original benchmark.
1. Open `main_baseline_experiments.py` and configure the `settings` dictionary to select either the `ME` (Mutually Exclusive) or `NME` (Non-Mutually Exclusive) setting by setting its value to `True`.

```
settings = {"ME": True,
            "NME": False
           }
```

2. Run the main script from the project's root directory:
```
python main_baseline_experiments.py
```

### 2. Extended Experiments
These are the new scalability experiments introduced in our paper.
1.  Open `main_extended_experiments.py` and configure the `settings` dictionary to select either the `ME` (Mutually Exclusive) or `NME` (Non-Mutually Exclusive) setting by setting its value to `True`.

```
settings = {"ME": True,
            "NME": False
           }
```
2. Run the main script from the project's root directory:
```
python main_extended_experiments.py
```

## Third-Party Libraries and Baselines
Our work builds upon the following libraries and research:
- [Umili et al. (2023)](https://github.com/whitemech/grounding_LTLf_in_image_sequences)
- [LTLf parser](https://github.com/whitemech/LTLf2DFA)

Please note that the code to run the baseline method is not included in this repository. To replicate the baseline results, please refer to the corresponding official code repository, available at https://github.com/whitemech/grounding_LTLf_in_image_sequences.
import sys
import os
import torch
from classifier import CNN_ME, CNN_NME

# -------------------------------------------------------------------------

# CNN hyperparameters
num_classes = 4
mutex = True

num_channels = 1
nodes_linear = 54

if mutex:
    scenario = 'ME'
    model = CNN_ME
else:
    scenario = 'NME'
    model = CNN_NME

model_name = f"untrained_CNN_{scenario}_state_dict_{num_classes}symb_.pth"


classifier = model(num_channels, num_classes, nodes_linear, mutually_exc=mutex)
torch.save(classifier.state_dict(), f=f"../models/{model_name}")
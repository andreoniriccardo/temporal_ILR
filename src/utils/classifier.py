import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN_ME(nn.Module):
    def __init__(self, channels, classes, nodes_linear, mutually_exc=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 3, 7, stride=2)
        self.conv2 = nn.Conv2d(3, 6, 7, stride=2)
        self.fc1 = nn.Linear(nodes_linear, classes)

        self.classes = classes
        if mutually_exc:
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        return self.activation(x)
    
class CNN_NME(nn.Module):
    def __init__(self, channels, classes, nodes_linear, mutually_exc=False, complete_initialization=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 3, 7, stride=2)
        self.conv2 = nn.Conv2d(3, 6, 7, stride=2)
        self.fc1 = nn.Linear(nodes_linear, classes)
        
        if complete_initialization:
            nn.init.kaiming_normal_(self.fc1.weight, 
                                mode='fan_out', 
                                nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        else:
            nn.init.kaiming_normal_(self.fc1.weight, 
                              mode='fan_out', 
                              nonlinearity='leaky_relu')

        
        self.classes = classes
        self.mutually_exc = mutually_exc
        
        if mutually_exc:
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        
        if self.mutually_exc:
            x = F.relu(self.fc1(x)) 
        else:
            x = self.fc1(x)
            
        return self.activation(x)
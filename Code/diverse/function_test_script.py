# prerequisites
from ast import arg
from tokenize import Double
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy.integrate import odeint


n_neurons = 30

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, g_output_dim)
    
        
        
    # forward method
    def forward(self,y):
        y = torch.tanh(self.fc1(y)) # leaky relu, with slope angle 
        y = torch.tanh(self.fc2(y)) 
        y = torch.tanh(self.fc3(y))
        
        
        #return torch.tanh(self.fc4(x))
        return self.fc4(y) 
    
z_dim = 1
x_dim = 1
y_dim = 1

G = Generator(g_input_dim = 1, g_output_dim = y_dim).to(device)


t_bc = np.array([[0]])
t = Variable(torch.from_numpy(t_bc).float(), requires_grad=True).to(device)

t_data = np.linspace(0,10,30)
t_data = t_data.reshape(30,1)
t_data = Variable(torch.from_numpy(t_data).float(), requires_grad=True).to(device)

print(G((t_data)))


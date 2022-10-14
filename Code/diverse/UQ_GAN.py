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

n_data = 20
timesteps = 500
slope = 0.01
drop = 0.2
criterion = nn.BCELoss() 
lr =  1e-4
np.random.seed(2022)
time_limit = 10
beta = 1 # weigt of ODE loss
#n_epochs = args.n_epochs
#z_dim = args.z_dim
n_epochs = 1
z_dim = 50
col_res = 1000 # collocation point resolution 
col_points = int(time_limit*col_res) # n collocation points

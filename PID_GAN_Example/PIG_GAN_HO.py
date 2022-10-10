import sys
#sys.path.insert(0, '../../../Utilities/')
import argparse
import os
import torch
from collections import OrderedDict
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import seaborn as sns
import pylab as py
import time
from pyDOE import lhs
import warnings
#sys.path.insert(0, '../../../Scripts/')
from models_pde import Generator, Discriminator, Q_Net
from pendulum_PIG_GAN import *
# from ../Scripts/helper import 


# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    
    
num_epochs = 1
lambda_phy = 1
lambda_q = 0.5

noise = 0.1

#architecture for the models
d_hid_dim = 50 
d_num_layer = 2

g_hid_dim = 50
g_num_layer = 4

q_hid_dim = 50
q_num_layer = 4


#%% 

N_u = 1
N_i = 50
N_f = 10000
data = scipy.io.loadmat(r"C:\Users\Ejer\OneDrive - Danmarks Tekniske Universitet\Dokumenter\GitHub\Speciale\PID_GAN_Example\pendulum_data.mat")

t = data['t'].flatten()[:,None]
Exact = data['u_sol'].flatten()[:,None]

X_star = np.hstack((t.flatten()[:,None]))
u_star = Exact.flatten()[:,None] 

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)


# initial conditions t = 0
t0 = t[0]
u0 = Exact[0]

X_f_train = lb + (ub-lb)*lhs(1, N_f)


X_u_train = t0 
u_train = u0


D = Discriminator(in_dim = 1, out_dim = 1, hid_dim = d_hid_dim, num_layers = d_num_layer).to(device)
G = Generator(in_dim = 1, out_dim = 1, hid_dim = g_hid_dim, num_layers = g_num_layer).to(device)
Q = Q_Net(in_dim = 1, out_dim = 1, hid_dim = q_hid_dim, num_layers = q_num_layer).to(device)

burgers = Pendulum_PIG(X_u_train, u_train, X_f_train, X_star, u_star, G, D, Q, device, num_epochs, lambda_phy, noise)
burgers.train()








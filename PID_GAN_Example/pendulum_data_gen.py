# prerequisites
from ast import arg
from tokenize import Double
from matplotlib.cbook import maxdict
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
from scipy.io import savemat
timesteps = 100

def pend(x, t, m, k):
    x1,x2 = x
    dxdt = [x2, -m*x2 - k*np.sin(x1)]
    return dxdt


t = np.linspace(0, 10, timesteps)
   
u_sol = np.zeros((timesteps))

x0 = [1,0]
m = 1    
k = 2
sol = odeint(pend, x0, t, args=(m, k))
u_sol = sol[:,0].T


mdic = {"t": t, "u_sol": u_sol}
savemat("pendulum_data", mdic)



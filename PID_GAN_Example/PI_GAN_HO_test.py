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
from pid import *
# from ../Scripts/helper import *

warnings.filterwarnings('ignore')

np.random.seed(1234)


# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
    
#%% Hyper paramters 

num_epochs = 1
lambda_val = 0.05
lambda_q = 0.5

noise = 0.1




#architecture for the models
d_hid_dim = 50 
d_num_layer = 2

g_hid_dim = 50
g_num_layer = 4

q_hid_dim = 50
q_num_layer = 4


N_u = 100
N_i = 50
N_f = 10000




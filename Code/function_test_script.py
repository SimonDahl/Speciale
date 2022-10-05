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



t = np.array([1])
y = np.array([1])

t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)

z = torch.concat((t,y))


print(z.shape)



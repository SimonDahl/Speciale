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

#%% load data 

N_u = 100
N_i = 50
N_f = 10000
data = scipy.io.loadmat(r"C:\Users\Ejer\OneDrive - Danmarks Tekniske Universitet\Dokumenter\GitHub\Speciale\PID_GAN_Example\burgers_shock.mat")

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)


X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None] 

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)

# initial conditions t = 0
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
uu1 = Exact[0:1,:].T

# boundary conditions x = lb
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
uu2 = Exact[:,0:1]

# boundary conditions, x = ub
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
uu3 = Exact[:,-1:]

X_u_train = np.vstack([xx2, xx3]) 
u_train = np.vstack([uu2, uu3])

X_f_train = lb + (ub-lb)*lhs(2, N_f)
X_f_train = np.vstack([X_f_train, X_u_train, xx1])

# selecting N_u boundary points for training
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx,:]

# selecting N_i initial points for training
idx = np.random.choice(xx1.shape[0], N_i, replace=False)
X_i_train = xx1[idx, :]
u_i_train = uu1[idx, :]

# adding boundary and initial points
X_u_train = np.vstack([X_u_train, X_i_train])
u_train = np.vstack([u_train, u_i_train])


#%% Models 

D = Discriminator(in_dim = 4, out_dim = 1, hid_dim = d_hid_dim, num_layers = d_num_layer).to(device)
G = Generator(in_dim = 3, out_dim = 1, hid_dim = g_hid_dim, num_layers = g_num_layer).to(device)
Q = Q_Net(in_dim = 3, out_dim = 1, hid_dim = q_hid_dim, num_layers = q_num_layer).to(device)

#%% Burgers 


burgers = Burgers_PID(X_u_train, u_train, X_f_train, X_star, u_star, G, D, Q, device, num_epochs, lambda_val, noise)

#%% 
burgers.train()

Xmean = burgers.Xmean
Xstd = burgers.Xstd
X_star_norm = (X_star - Xmean) / Xstd
nsamples = 500
u_pred_list = []
f_pred_list = []
for run in range(nsamples):
    u_pred, f_pred = burgers.predict(X_star_norm)
    u_pred_list.append(u_pred)
    f_pred_list.append(f_pred)

    
u_pred_arr = np.array(u_pred_list)
f_pred_arr = np.array(f_pred_list)
u_pred = u_pred_arr.mean(axis=0)
f_pred = f_pred_arr.mean(axis=0)
u_dev = u_pred_arr.var(axis=0)
f_dev = f_pred_arr.var(axis=0)

error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
print('Error u: %e' % (error_u))                     
print('Residual: %e' % (f_pred**2).mean())
U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
U_dev = griddata(X_star, u_dev.flatten(), (X, T), method='cubic')
Error = np.abs(Exact - U_pred)


""" The aesthetic setting has changed. """

####### Row 0: u(t,x) ##################    
X_u_train_ = X_u_train 
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15) 

ax.plot(
    X_u_train_[:,1], 
    X_u_train_[:,0], 
    'kx', label = 'Data (%d points)' % (u_train.shape[0]), 
    markersize = 4,  # marker size doubled
    clip_on = False,
    alpha=1.0
)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.9, -0.05), 
    ncol=5, 
    frameon=False, 
    prop={'size': 15}
)
ax.set_title('$u(t,x)$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

####### Row 0: u(t,x) ##################    

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15) 

ax.plot(
    X_u_train_[:,1], 
    X_u_train_[:,0], 
    'kx', label = 'Data (%d points)' % (u_train.shape[0]), 
    markersize = 4,  # marker size doubled
    clip_on = False,
    alpha=1.0
)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.9, -0.05), 
    ncol=5, 
    frameon=False, 
    prop={'size': 15}
)
ax.set_title('$u(t,x)$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

plt.show()

####### Row 0: u(t,x) ##################    

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

h = ax.imshow(U_dev.T, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15) 

ax.plot(
    X_u_train_[:,1], 
    X_u_train_[:,0], 
    'kx', label = 'Data (%d points)' % (u_train.shape[0]), 
    markersize = 4,  # marker size doubled
    clip_on = False,
    alpha=1.0
)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.9, -0.05), 
    ncol=5, 
    frameon=False, 
    prop={'size': 15}
)
ax.set_title('$u(t,x)$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

plt.show()

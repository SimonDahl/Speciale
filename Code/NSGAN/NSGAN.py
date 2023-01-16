# prerequisites
from ast import arg
from tkinter import X
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
import time 
import argparse
import scipy
from scipy.interpolate import griddata
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy.integrate import solve_ivp


#np.random.seed(1234)
n_neurons = 20
n_neurons_d = 128
lr = 0.001
criterion_MSE = nn.MSELoss() # loss function
criterion_BCE = nn.BCELoss() 

HPC = False

if HPC == True:
    n_epochs = 5
    N_train = 500
    data = scipy.io.loadmat(r"~/speciale/cylinder_nektar_wake.mat")
if HPC == False: 
    n_epochs = 1
    N_train = 100
    data = scipy.io.loadmat(r"C:\Users\Simon\OneDrive - Danmarks Tekniske Universitet\Speciale\Speciale\Code\NSGAN\cylinder_nektar_wake.mat")


lambda_1 = 1
lambda_2 = 0.01



U_star = data['U_star'] # N x 2 x T
P_star = data['p_star'] # N x T
t_star = data['t'] # T x 1
X_star = data['X_star'] # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data 
XX = np.tile(X_star[:,0:1], (1,T)) # N x T
YY = np.tile(X_star[:,1:2], (1,T)) # N x T
TT = np.tile(t_star, (1,N)).T # N x T

UU = U_star[:,0,:] # N x T
VV = U_star[:,1,:] # N x T
PP = P_star # N x T

x = XX.flatten()[:,None] # NT x 1
y = YY.flatten()[:,None] # NT x 1
t = TT.flatten()[:,None] # NT x 1

u = UU.flatten()[:,None] # NT x 1
v = VV.flatten()[:,None] # NT x 1
p = PP.flatten()[:,None] # NT x 1

 # Training Data    
idx = np.random.choice(N*T, N_train, replace=False)
x_train = x[idx,:]
y_train = y[idx,:]
t_train = t[idx,:]
u_train = u[idx,:]
v_train = v[idx,:]

x_train = Variable(torch.from_numpy(x_train).float(), requires_grad=True).to(device)
y_train = Variable(torch.from_numpy(y_train).float(), requires_grad=True).to(device)
t_train = Variable(torch.from_numpy(t_train).float(), requires_grad=True).to(device)
u_train = Variable(torch.from_numpy(u_train).float(), requires_grad=True).to(device)
v_train = Variable(torch.from_numpy(v_train).float(), requires_grad=True).to(device)


class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, n_neurons)
        self.fc5 = nn.Linear(n_neurons, n_neurons)
        self.fc6 = nn.Linear(n_neurons, n_neurons)
        self.fc7 = nn.Linear(n_neurons, n_neurons)
        self.fc8 = nn.Linear(n_neurons, n_neurons)
        self.fc9 = nn.Linear(n_neurons, g_output_dim)
        
      
    # forward method
    def forward(self,y):
        y = F.silu(self.fc1(y)) 
        y = F.silu(self.fc2(y)) 
        y = F.silu(self.fc3(y))
        y = F.silu(self.fc4(y))
        y = F.silu(self.fc5(y))
        y = F.silu(self.fc6(y))
        y = F.silu(self.fc7(y))
        y = F.silu(self.fc8(y))
        return self.fc9(y) 

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Conv2d(1, 4, (3,3), stride=1)
        self.fc2 = nn.Conv2d(4, 8, (2,2), stride=1)
        self.fc3 = nn.Conv2d(8, 16, (2,2), stride=1)
        self.fc4 = nn.Linear(n_neurons_d,n_neurons_d)
        self.fc5 = nn.Linear(n_neurons_d,n_neurons_d)
        self.fc6 = nn.Linear(n_neurons_d, 1)  # output dim = 1 for binary classification 
    
    # forward method
    def forward(self, d):
        d = torch.relu(self.fc1(d))
        d = torch.relu(self.fc2(d))
        d = torch.relu(self.fc3(d))
        d = torch.relu(self.fc4(d))
        d = torch.relu(self.fc5(d))
        return torch.sigmoid(self.fc6(d))


#  build networks
G = Generator(g_input_dim =3, g_output_dim = 2).to(device)
D = Discriminator(d_input_dim = 2).to(device)

def predict(x,y,t):

    psi_and_p = G(torch.concat((x,y,t),dim=1))
    psi = psi_and_p[:,0:1]
    p = psi_and_p[:,1:2]
    
    u = torch.autograd.grad(psi, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]# 
    v = -torch.autograd.grad(psi, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]#

    return p,u,v

def compute_residuals(x, y, t):

    
    psi_and_p = G(torch.concat((x,y,t),dim=1))
    psi = psi_and_p[:,0:1]
    p = psi_and_p[:,1:2]


    u = torch.autograd.grad(psi, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]# 
    v = -torch.autograd.grad(psi, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]#
    
    u_t = torch.autograd.grad(u, t, torch.ones_like(t), retain_graph=True,create_graph=True)[0]# 
    u_x = torch.autograd.grad(u, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]# 
    u_y = torch.autograd.grad(u, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]# 

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), retain_graph=True,create_graph=True)[0]# 
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), retain_graph=True,create_graph=True)[0]# 

    v_t = torch.autograd.grad(v, t, torch.ones_like(t), retain_graph=True,create_graph=True)[0]# 
    v_x = torch.autograd.grad(v, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]# 
    v_y = torch.autograd.grad(u, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]# 
   
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]# 
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]#  

    p_x = torch.autograd.grad(p, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]# 
    p_y = torch.autograd.grad(p, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]# 
  
    f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
    f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
    
    return u, v, p, f_u, f_v


def rearange(x,y,t,p,u,v):
    img = torch.zeros(3,3)
    img[]



def G_train(x,y,t,p,u,v):
    
    G.zero_grad()
    
    # MSE loss on training points
    p_fake,u_fake,v_fake = predict(x,y,t)
    
    MSE_p = criterion_MSE(p,p_fake)
    MSE_u = criterion_MSE(u,u_fake)
    MSE_v = criterion_MSE(v,v_fake)
    
    

    _, _, _, f_u, f_v = compute_residuals(x,y,t)
    target = torch.zeros_like(f_u)

    MSE_f_u = criterion_MSE(f_u,target)
    MSE_f_v = criterion_MSE(f_v,target)
    
    d_input = torch.cat((t_train,u_fake),dim=1)     
    L_adv = torch.mean(D(d_input)) 
           
    G_loss = L_adv + L_MSE + lambda_phy * L_phy 
    
    G_loss.backward()
    G_optimizer.step()
    return G_loss
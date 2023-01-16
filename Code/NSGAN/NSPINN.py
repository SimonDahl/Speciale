
# denne kode er brugt til sektion NSGAN for at lave pendul løsnig. og er brugt som kode eksempel 

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
lr = 0.001
criterion = nn.MSELoss()

HPC = True

if HPC == True:
    n_epochs = 5
    N_train = 500
    data = scipy.io.loadmat('cylinder_nektar_wake.mat')
if HPC == False: 
    n_epochs = 1
    N_train = 100
    data = scipy.io.loadmat(r"C:\Users\Simon\OneDrive - Danmarks Tekniske Universitet\Speciale\Speciale\Code\NSGAN\PINN\cylinder_nektar_wake.mat")


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





class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()       
        self.fc1 = nn.Linear(3, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons,n_neurons)
        self.fc4 = nn.Linear(n_neurons,n_neurons)
        self.fc5 = nn.Linear(n_neurons,n_neurons)
        self.fc6 = nn.Linear(n_neurons,n_neurons)
        self.fc7 = nn.Linear(n_neurons,n_neurons)
        self.fc8 = nn.Linear(n_neurons,2)
        
       
    # forward method
    def forward(self,y):
        y = torch.tanh(self.fc1(y)) 
        y = torch.tanh(self.fc2(y))
        y = torch.tanh(self.fc3(y))
        y = torch.tanh(self.fc4(y)) 
        y = torch.tanh(self.fc5(y)) 
        y = torch.tanh(self.fc6(y)) 
        y = torch.tanh(self.fc7(y)) 
        return self.fc8(y) 
  
    
   
net = PINN().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)


def predict(x,y,t):

    psi_and_p = net(torch.concat((x,y,t),dim=1))
    psi = psi_and_p[:,0:1]
    p = psi_and_p[:,1:2]
    
    u = torch.autograd.grad(psi, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]# 
    v = -torch.autograd.grad(psi, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]#

    return u,v,p



def compute_residuals(x, y, t):

    
    psi_and_p = net(torch.concat((x,y,t),dim=1))
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

def train(x, y, t, u, v):
    optimizer.zero_grad()
      
   # lb = X.min(0)
    #ub = X.max(0)
            

    u_pred, v_pred, p_pred, f_u_pred, f_v_pred = compute_residuals(x,y,t)

    MSE_u_u = criterion(u,u_pred)
    MSE_u_v = criterion(v,v_pred)

    f_target = torch.zeros_like(f_u_pred)

    MSE_f_u = criterion(f_target,f_u_pred)
    MSE_f_v = criterion(f_target,f_v_pred)

    loss = MSE_u_u + MSE_u_v + MSE_f_u + MSE_f_v

    loss.backward()
    optimizer.step()
    
    return loss.data.item()

losses = []

for epoch in range(1, n_epochs+1):
    
    losses.append(train(x_train, y_train, t_train, u_train, v_train))

    print('[%d/%d]: loss: %.4f' % ((epoch), n_epochs, torch.mean(torch.FloatTensor(losses))))
    

#%% Test 

# Test Data
snap = np.array([100])
x_star = X_star[:,0:1]
y_star = X_star[:,1:2]
t_star = TT[:,snap]

u_star = U_star[:,0,snap]
v_star = U_star[:,1,snap]
p_star = P_star[:,snap]

x_star = Variable(torch.from_numpy(x_star).float(), requires_grad=True).to(device)
y_star = Variable(torch.from_numpy(y_star).float(), requires_grad=True).to(device)
t_star = Variable(torch.from_numpy(t_star).float(), requires_grad=True).to(device)
u_star = Variable(torch.from_numpy(u_star).float(), requires_grad=True).to(device)
v_star = Variable(torch.from_numpy(v_star).float(), requires_grad=True).to(device)
p_star = Variable(torch.from_numpy(p_star).float(), requires_grad=True).to(device)

u_pred, v_pred, p_pred = predict(x_star, y_star, t_star)

# Error u,v,p 


def plot_solution(X_star, u_star, index):
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    plt.colorbar()
   
    if HPC == False: 
        plt.show()
    if HPC == True: 
        plt.savefig('./output/NS/'+'Plot ' +str(index)+'.png')     
   


with torch.no_grad():
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)
        
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))    

     # Predict for plotting
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')

    plot_solution(X_star, p_pred, 3)    
    plot_solution(X_star, p_star, 4)
    plot_solution(X_star, p_star - p_pred, 5)
    